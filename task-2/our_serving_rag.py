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

# === FastAPI App ===
app = FastAPI()

# === Configuration ===
MAX_BATCH_SIZE = 16           # Maximum number of requests processed in a batch
MAX_WAITING_TIME = 1.0        # Maximum time (in seconds) to wait before processing a batch

# === Sample Documents for Retrieval ===
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
]

# === Load Embedding Model Locally ===
# Assume that models have been downloaded and cached locally in the corresponding directory 
LOCAL_MODEL_PATH = "../../.cache/huggingface/hub/models--intfloat--multilingual-e5-large-instruct/snapshots/84344a23ee1820ac951bc365f1e91d094a911763"
embed_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
embed_model = AutoModel.from_pretrained(LOCAL_MODEL_PATH)

# === Load Basic Chat Language Model (LLM) for QA generation ===
chat_pipeline = pipeline("text-generation", model="../../.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6")

# === Request Queue and Control Flags ===
request_queue = Queue()
processing_thread = None
shutdown_flag = False

# === Performance Metrics ===
performance_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "avg_latency": 0,
    "max_batch_size_used": 0,
    "batch_processing_times": []
}

# === Embedding Functions ===
def get_embedding(text: str) -> np.ndarray:
    """
    Compute the embedding of a single input text.
    Args:
        text (str): Input string to embed.
    Returns:
        np.ndarray: Embedding vector of shape (1, D), where D is the hidden size of the model.
    """

    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def get_embedding_batch(batch: List[str]) -> torch.Tensor:
    """
    Compute the embeddings for a batch of input strings.
    Args:
        batch (List[str]): List of input strings to embed.
    Returns:
        torch.Tensor: Tensor of shape (B, D), where B is the batch size and D is the hidden size of the model.
    """
    inputs = embed_tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# === Precompute Document Embeddings ===
doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])

# === Top-k Retrieval Function (Vector Similarity) ===
def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    """
    Retrieve the top-k most similar documents to a query embedding using dot product similarity.
    Args:
        query_emb (np.ndarray): Embedding of the query of shape (1, D).
        k (int): Number of top documents to retrieve.

    Returns:
        List[str]: List of the top-k most similar documents (strings).
    """
    sims = doc_embeddings @ query_emb.T
    top_k_indices = np.argsort(sims.ravel())[::-1][:k]
    return [documents[i] for i in top_k_indices]

# === Torch Dot Product (for GPU optimization) ===
@torch.jit.script
def d_dot(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute the dot product between two tensors using TorchScript for potential GPU optimization.
    Args:
        X (torch.Tensor): Tensor of shape (N, D) or (D,).
        Y (torch.Tensor): Tensor of shape (M, D) or (D,).

    Returns:
        torch.Tensor: Dot product result. 
            - If X and Y are 2D: returns (N,) tensor with dot products per row.
            - If both are 1D: returns scalar (1,).
    """
    if X.dim() > 1 and Y.dim() > 1:
        return torch.sum(X * Y, dim=-1)
    return X @ Y

# === Batched Retrieval Function ===
def retrieve_top_k(N: int, D: int, A: torch.Tensor, X: torch.Tensor, Ks) -> List[List[str]]:
    """
    Retrieve the top-k most similar documents for each query in a batch using dot product similarity.
    Args:
        N (int): Number of documents.
        D (int): Dimensionality of embeddings.
        A (torch.Tensor): Document embeddings of shape (N, D).
        X (torch.Tensor): Query embeddings of shape (B, D), where B is batch size.
        Ks (List[int]): List of integers specifying top-k to retrieve per query.
    Returns:
        List[List[str]]: A list of lists where each sublist contains the top-k retrieved documents (strings) for a query.
    """

    similarity = torch.mm(X, A.t())
    _, indices = torch.sort(similarity, dim=1, descending=False)
    return [[documents[indices[b, i]] for i in range(len(indices[b])) if i < Ks[b]] for b in range(len(indices))]

# === Process Batch of Requests ===
def process_batch(batch: List[Tuple[Dict, datetime, Future]]) -> List[Dict]:
    """
    Process a batch of RAG requests: embed queries, retrieve top-k documents, and generate answers using an LLM.

    Args:
        batch (List[Tuple[Dict, datetime, Future]]): 
            Each tuple contains:
                - request (Dict): with keys "query" (str) and "k" (int),
                - timestamp (datetime): when the request was received,
                - future (Future): async future for returning results.

    Returns:
        List[str]: List of generated responses (one per query in the batch).
    """
    queries = [item[0]['query'] for item in batch]
    ks = np.array([item[0]['k'] for item in batch])
    query_embs = get_embedding_batch(queries)

    N, D = doc_embeddings.shape
    retrieved_docs = retrieve_top_k(N, D, torch.tensor(doc_embeddings), query_embs, ks)

    # Build prompt for LLM
    prompts = [
        f"Question: {request['query']}\nContext:\n{'\n'.join(retrieved_doc)}\nAnswer:"
        for (request, _, _), retrieved_doc in zip(batch, retrieved_docs)
    ]

    results = chat_pipeline(prompts, max_length=50, do_sample=True, batch_size=len(batch))
    return [result[0]["generated_text"] for result in results]

# === Background Thread for Processing Requests in Batches ===
def batch_processor():
    """Continuously process requests from the queue in batches."""
    global shutdown_flag
    while not shutdown_flag or not request_queue.empty():
        batch = []
        start_time = time.time()
        
        # Collect batch
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

# === Request Schema ===
class QueryRequest(BaseModel):
    query: str 
    k: int = 2

# === RAG Inference Endpoint ===
@app.post("/rag")
async def predict(payload: QueryRequest):
    """RAG endpoint: Receives a query and returns generated answer using retrieved documents."""
    performance_stats['total_requests'] += 1
    
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    
    request_queue.put((payload.model_dump(), datetime.now(), future))
    
    try:
        result = await asyncio.wait_for(future, timeout=30)
        return result
    except asyncio.TimeoutError:
        return {"error": "Request timed out"}

# === Performance Metrics Endpoint ===
@app.get("/performance")
def get_performance():
    """Returns current performance statistics."""
    return performance_stats

# === Startup Event: Launch Background Thread ===
@app.on_event("startup")
def startup_event():
    global processing_thread
    processing_thread = Thread(target=batch_processor, daemon=True)
    processing_thread.start()

# === Shutdown Event: Graceful Termination ===
@app.on_event("shutdown")
def shutdown_event():
    global shutdown_flag
    shutdown_flag = True
    processing_thread.join()

# === Run App Locally ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
