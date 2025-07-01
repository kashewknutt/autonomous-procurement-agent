from sqlalchemy import Column, Integer, String, Float
from app.db.database import Base

class Quote(Base):
    __tablename__ = "quotes"
    id = Column(Integer, primary_key=True, index=True)
    supplier = Column(String, index=True)
    content = Column(String)
    price = Column(Float)
