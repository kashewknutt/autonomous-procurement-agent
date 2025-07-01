from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.db.models import Quote
from app.embeddings.faiss_index import add_documents_with_ids, search_ids, reset_index

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/quote")
def add_quote(
    supplier: str = Body(...),
    content: str = Body(...),
    price: float = Body(...),
    db: Session = Depends(get_db)
):
    quote = Quote(supplier=supplier, content=content, price=price)
    db.add(quote)
    db.commit()
    db.refresh(quote)

    add_documents_with_ids([quote.content], [quote.id])
    return {"id": quote.id}

@router.get("/search")
def search_quotes(query: str, db: Session = Depends(get_db)):
    ids = search_ids(query)
    results = db.query(Quote).filter(Quote.id.in_(ids)).all()
    return {"results": [{"id": q.id, "supplier": q.supplier, "price": q.price, "content": q.content} for q in results]}

@router.post("/reset")
def reset(db: Session = Depends(get_db)):
    reset_index()
    db.query(Quote).delete()
    db.commit()
    return {"message": "FAISS and database reset."}