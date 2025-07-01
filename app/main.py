from fastapi import FastAPI
from app.api.routes import router
from app.db.init_db import init_db

app = FastAPI()
app.include_router(router, prefix="/api")

@app.on_event("startup")
def on_startup():
    init_db()
