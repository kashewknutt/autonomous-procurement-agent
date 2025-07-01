# app/embeddings/chroma_index.py
import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from sqlalchemy.orm import Session
from app.db.models import Quote


model_name = "all-MiniLM-L6-v2"
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

client = chromadb.PersistentClient(path="./chroma-data")
collection = client.get_or_create_collection(
    name="quotes",  
    embedding_function=embed_fn
)

def reset_index():
    client.delete_collection("quotes")
    client.create_collection(name="quotes", embedding_function=embed_fn)

def add_quotes_from_db(db: Session):
    from app.db.models import Quote
    quotes = db.query(Quote).all()

    if not quotes:
        print("No quotes found in DB. Skipping Chroma collection add.")
        return
    documents = [q.content for q in quotes]
    metadatas = [
        {
            "supplier": q.supplier,
            "unit_price": q.unit_price,
            "total_price": q.total_price,
            "quantity": q.quantity,
            "vertical": q.vertical
        } for q in quotes
    ]
    ids = [str(q.id) for q in quotes]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)

def search_quote_ids(query: str, top_k: int = 5, threshold: float = 0.5) -> list[int]:
    """Enhanced search with better similarity scoring."""
    try:
        results = collection.query(query_texts=[query], n_results=min(top_k, 20))
        ids = []
        
        if results["ids"] and results["ids"][0]:
            for i, distance in zip(results["ids"][0], results["distances"][0]):
                # Convert cosine distance to similarity
                similarity = 1 - distance
                if similarity >= threshold:
                    ids.append(int(i))
        
        return ids
    except Exception as e:
        print(f"Search error: {e}")
        return []
