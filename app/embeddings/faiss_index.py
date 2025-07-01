import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FAISS index
dimension = 384  # All-MiniLM-L6-v2 outputs 384-dim vectors
index = faiss.IndexFlatL2(dimension)
documents = []  # In-memory list for mapping results

def reset_index():
    global index, documents
    index = faiss.IndexFlatL2(dimension)
    documents = []

def add_documents_with_ids(texts: list[str], ids: list[int]):
    global documents
    embeddings = model.encode(texts)
    index.add(np.array(embeddings).astype("float32"))
    documents.extend(ids)

def search_ids(query: str, top_k: int = 3) -> list[int]:
    if index.ntotal == 0:
        return []

    embedding = model.encode([query])
    distances, indices = index.search(np.array(embedding).astype("float32"), top_k)
    return [documents[i] for i in indices[0]]

