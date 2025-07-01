from app.agent.core import get_agent
from fastapi import APIRouter, Body

router = APIRouter()

@router.post("/agent/invoke")
def invoke_agent(query: str = Body(..., embed=True)):
    agent = get_agent()
    response = agent.run(query)
    return {"response": response}
