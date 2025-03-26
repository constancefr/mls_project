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

# 1. Load embedding model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)

# Basic Chat LLM
chat_pipeline = pipeline("text-generation", model="facebook/opt-125m")

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

def get_embedding_batch(batch : List[str]) -> torch.Tensor:
    inputs = embed_tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])

def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    sims = doc_embeddings @ query_emb.T
    top_k_indices = np.argsort(sims.ravel())[::-1][:k]
    return [documents[i] for i in top_k_indices]

@torch.jit.script
def d_dot(X: torch.Tensor,Y: torch.Tensor) -> torch.Tensor:
    if X.dim() > 1 and Y.dim() > 1:
        return torch.sum(X*Y, dim=-1)
    return X @ Y

def retrieve_top_k(N,D,A,X,Ks):        
    similarity = torch.mm(X, A.t())
    _, indices = torch.sort(similarity, dim=1, descending=False)
    return [[documents[indices[b,i]] for i in range(len(indices[b])) if i < Ks[b]] for b in range(len(indices))]

def process_batch(batch: List[Tuple[Dict, datetime, Future]]) -> List[Dict]:
    queries = [item[0]['query'] for item in batch]
    ks = np.array([item[0]['k'] for item in batch])
    query_embs = get_embedding_batch(queries)
    N,D = doc_embeddings.shape
    retrieved_docs = retrieve_top_k(N,D,torch.tensor(doc_embeddings),query_embs,ks)
    results = []    
    nl = '\n'
    prompts = [f"Question: {request['query']}\nContext:\n{nl.join(retrieved_doc)}\nAnswer:" 
               for (request, _, _), retrieved_doc in zip(batch, retrieved_docs)]
    results = chat_pipeline(prompts, max_length=50, do_sample=True, batch_size=len(batch))
    return [result[0]["generated_text"] for result in results]

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