from fastapi import APIRouter
from app.embeddings.faiss_index import search_similar_quotes

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.get("/search")
async def search(query: str):
    result = search_similar_quotes(query)
    return {"result": result}