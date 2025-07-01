import os
from fastapi import FastAPI
from app.api.routes import router
from app.db.init_db import init_db
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

app = FastAPI()
app.include_router(router, prefix="/api")

@app.on_event("startup")
def on_startup():
    init_db()
