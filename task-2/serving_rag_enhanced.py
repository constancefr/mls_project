import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from queue import Queue
from threading import Thread
import time
from datetime import datetime
from typing import List, Dict, Tuple
import json
import asyncio
from concurrent.futures import Future

app = FastAPI()

# Configuration
MAX_BATCH_SIZE = 8 # increased from 4
MAX_WAITING_TIME = 1.0 # increased from 0.1
PERFORMANCE_LOG_FILE = "rag_performance.log"

# Example documents
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
]

# Models
EMBED_MODEL_PATH = "/home/s2666210/models/multilingual-e5-large-instruct"
CHAT_MODEL_PATH = "/home/s2666210/models/opt-125m"

# Initialize models
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_PATH)
chat_pipeline = pipeline("text-generation", model=CHAT_MODEL_PATH)

# Request queue and processing setup
request_queue = Queue()
processing_thread = None
shutdown_flag = False

# Performance tracking
performance_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "avg_latency": 0,
    "max_batch_size_used": 0,
    "batch_processing_times": []
}

def get_embedding(text: str) -> np.ndarray:
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])

def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    sims = doc_embeddings @ query_emb.T
    top_k_indices = np.argsort(sims.ravel())[::-1][:k]
    return [documents[i] for i in top_k_indices]

def process_batch(batch: List[Tuple[Dict, datetime, Future]]) -> List[Dict]:
    start_time = time.time()
    
    # Batch embeddings
    queries = [item[0]['query'] for item in batch]
    query_inputs = embed_tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        query_outputs = embed_model(**query_inputs)
        query_embs = query_outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    results = []
    for (request, arrival_time, _), query_emb in zip(batch, query_embs):
        try:
            retrieved_docs = retrieve_top_k(query_emb, request['k'])
            context = "\n".join(retrieved_docs)
            prompt = f"Question: {request['query']}\nContext:\n{context}\nAnswer:"
            
            generated = chat_pipeline(prompt, max_length=50, do_sample=True)[0]["generated_text"]
            
            latency = time.time() - arrival_time.timestamp()
            performance_stats['successful_requests'] += 1
            performance_stats['avg_latency'] = (
                performance_stats['avg_latency'] * (performance_stats['successful_requests'] - 1) + latency
            ) / performance_stats['successful_requests']
            
            results.append({
                "query": request['query'],
                "result": generated,
                "latency": latency
            })
        except Exception as e:
            results.append({
                "query": request['query'],
                "error": str(e)
            })
    
    batch_size = len(batch)
    processing_time = time.time() - start_time
    performance_stats['batch_processing_times'].append(processing_time)
    performance_stats['max_batch_size_used'] = max(
        performance_stats['max_batch_size_used'], batch_size)
    
    return results

def batch_processor():
    global shutdown_flag
    while not shutdown_flag or not request_queue.empty():
        batch = []
        start_time = time.time()
        
        while (len(batch) < MAX_BATCH_SIZE and 
               (time.time() - start_time) < MAX_WAITING_TIME):
            try:
                item = request_queue.get(timeout=MAX_WAITING_TIME)
                batch.append(item)
            except:
                pass
        
        if batch:
            results = process_batch(batch)
            for (_, _, future), result in zip(batch, results):
                if future is not None:
                    future.set_result(result)

class QueryRequest(BaseModel):
    query: str
    k: int = 2

@app.post("/rag")
async def predict(payload: QueryRequest):
    performance_stats['total_requests'] += 1
    
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    
    request_queue.put((payload.model_dump(), datetime.now(), future))
    
    try:
        result = await asyncio.wait_for(future, timeout=30)
        return result
    except asyncio.TimeoutError:
        return {"error": "Request timed out"}

@app.get("/performance")
def get_performance():
    return performance_stats

@app.on_event("startup")
def startup_event():
    global processing_thread
    processing_thread = Thread(target=batch_processor, daemon=True)
    processing_thread.start()

@app.on_event("shutdown")
def shutdown_event():
    global shutdown_flag
    shutdown_flag = True
    processing_thread.join()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)