## app/api/agent_route.py
from typing import Optional

from pydantic import BaseModel
from app.agent.core import get_agent
from fastapi import APIRouter

router = APIRouter()

class AgentRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

@router.post("/agent/invoke")
def invoke_agent(request: AgentRequest):
    session_id = request.session_id or "default_procurement"
    agent = get_agent(session_id)
    print("🔍 Loaded memory:", agent.memory.load_memory_variables({}))

    result = agent.run(request.query)
    return {"response": result, "session_id": session_id}
