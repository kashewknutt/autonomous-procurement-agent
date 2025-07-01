## app/api/routes.py
from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.db.models import Quote
from app.agent.core import get_agent
from app.embeddings.chroma_index import (
    reset_index, add_quotes_from_db, search_quote_ids
)
from .agent_route import router as agent_router

router = APIRouter()
router.include_router(agent_router)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---- SYSTEM ----
@router.get("/health")
def health():
    return {"status": "ok"}

# ---- CHROMA ----
@router.post("/reset-chroma")
def reset_chroma():
    reset_index()
    db = SessionLocal()
    add_quotes_from_db(db)
    db.close()
    return {"message": "Chroma index reset and populated."}

# ---- QUOTES ----
@router.post("/quote")
def add_quote(
    supplier: str = Body(...),
    content: str = Body(...),
    unit_price: float = Body(None),
    total_price: float = Body(None),
    quantity: int = Body(None),
    vertical: str = Body("general"),
    db: Session = Depends(get_db)
):
    quote = Quote(
        supplier=supplier, 
        content=content, 
        unit_price=unit_price,
        total_price=total_price,
        quantity=quantity,
        vertical=vertical
    )
    db.add(quote)
    db.commit()
    db.refresh(quote)
    
    # Update Chroma index
    from app.embeddings.chroma_index import collection
    collection.add(
        documents=[quote.content],
        metadatas={
            "supplier": quote.supplier, 
            "unit_price": quote.unit_price,
            "total_price": quote.total_price,
            "quantity": quote.quantity,
            "vertical": quote.vertical
        },
        ids=[str(quote.id)]
    )
    
    return {"id": quote.id}

@router.get("/search")
def search_quotes(query: str, db: Session = Depends(get_db)):
    ids = search_quote_ids(query)
    results = db.query(Quote).filter(Quote.id.in_(ids)).all()
    return {
        "results": [
            {"id": q.id, "supplier": q.supplier, "price": q.price, "content": q.content}
            for q in results
        ]
    }

@router.post("/reset")
def reset(db: Session = Depends(get_db)):
    reset_index()
    db.query(Quote).delete()
    db.commit()
    return {"message": "Chroma and database reset."}

# ---- SUPPLIERS ----
@router.get("/suppliers")
def list_suppliers(db: Session = Depends(get_db)):
    suppliers = db.query(Quote.supplier).distinct().all()
    return {"suppliers": [s[0] for s in suppliers]}

# ---- AGENT ----
@router.post("/agent")
def agent_query(input: str = Body(...)):
    agent = get_agent()
    result = agent.run(input)
    return {"response": result}
