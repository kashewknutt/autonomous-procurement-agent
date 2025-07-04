import os
# from dotenv import load_dotenv
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))



from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.db.init_db import init_db
from app.embeddings.chroma_index import add_quotes_from_db
from app.db.database import SessionLocal


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.on_event("startup")
def on_startup():
    init_db()
    db = SessionLocal()
    add_quotes_from_db(db)
    db.close()