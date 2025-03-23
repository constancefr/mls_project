import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

app = FastAPI()

# Example documents in memory
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
]

# 1. Load embedding model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct" # pre-trained embedding model from Hugging Face
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME) # converts text into tokens
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME) # computes embeddings for text

# Basic Chat LLM
chat_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct") # will generate responses based on query and retrieved documents


## Hints:

### Step 3.1:
# 1. Initialize a request queue
# 2. Initialize a background thread to process the request (via calling the rag_pipeline function)
# 3. Modify the predict function to put the request in the queue, instead of processing it immediately

### Step 3.2:
# 1. Take up to MAX_BATCH_SIZE requests from the queue or wait until MAX_WAITING_TIME
# 2. Process the batched requests

def get_embedding(text: str) -> np.ndarray:
    """
    Computes a simple average-pool embedding.

    Input: a text string.
    Ourput: a NumPy array representing the embedding of the text.
    """
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True) # tokenize text

    # Pass tokenized input through embedding model to get hidden states,
    # compute the avg of the hidden states along the sequence dimension.
    with torch.no_grad():
        outputs = embed_model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).cpu().numpy() # move to CPU, convert to NumPy

# Precompute document embeddings
doc_embeddings = np.vstack([get_embedding(doc) for doc in documents]) # np.vstack to stack embeddings vertically into single array

### You may want to use your own top-k retrieval method (task 1)
def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    """
    Retrieve top-k docs via dot-product similarity.
    
    Input: query embedding, number of documents to retrieve.
    Output: list of the top-k most relevant documents.
    """
    sims = doc_embeddings @ query_emb.T # dot-prod similarity
    top_k_indices = np.argsort(sims.ravel())[::-1][:k]

    return [documents[i] for i in top_k_indices]

def rag_pipeline(query: str, k: int = 2) -> str:
    '''
    Input: user query, number of docs to retrieve.
    Output: generated response from the LLM.
    '''
    # Step 1: Input embedding
    query_emb = get_embedding(query)
    
    # Step 2: Retrieval
    retrieved_docs = retrieve_top_k(query_emb, k)
    
    # Construct the prompt from query + retrieved docs
    context = "\n".join(retrieved_docs)
    prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
    
    # Step 3: LLM Output
    generated = chat_pipeline(prompt, max_length=50, do_sample=True)[0]["generated_text"]
    return generated

# Define request model using Pydantic
class QueryRequest(BaseModel):
    query: str
    k: int = 2

# Define API endpoint
@app.post("/rag")
def predict(payload: QueryRequest):
    '''
    Input: JSON payload containing query & k (see QueryRequest class).
    Output: JSON response containing original query & generated result.
    '''
    result = rag_pipeline(payload.query, payload.k)
    
    return {
        "query": payload.query,
        "result": result,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)