import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Simple in-memory FAISS setup
dimension = 384
index = faiss.IndexFlatL2(dimension)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Dummy quotes to embed
quotes = [
    "We offer a 10% discount on bulk orders",
    "Shipping will take 7-10 business days",
    "We cannot reduce the price further"
]

# Create embeddings and add to FAISS
embeddings = model.encode(quotes)
index.add(np.array(embeddings))

def search_similar_quotes(query: str, top_k: int = 2):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), top_k)
    results = [quotes[i] for i in indices[0]]
    return results