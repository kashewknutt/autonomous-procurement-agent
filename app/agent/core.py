## app/agent/core.py
from langchain.agents import initialize_agent, AgentType
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration
import httpx
from pydantic import Field
from app.agent.tools import find_best_quote_for_quantity, procurement_requirements_form, procurement_knowledge_tool, vendor_management_tool, search_quotes_by_vertical, search_quotes_tool, search_best_quote, check_procurement_policy
from app.agent.memory import get_memory
from app.agent.prompts import DEFAULT_AGENT_PREFIX, DEFAULT_AGENT_SUFFIX
import os

class GitHubChatModel(BaseChatModel):
    temperature: float = Field(default=0.3)
    max_tokens: int = Field(default=256)

    def _convert_message(self, m):
        if isinstance(m, HumanMessage):
            return {"role": "user", "content": m.content}
        elif isinstance(m, AIMessage):
            return {"role": "assistant", "content": m.content}
        else:
            return {"role": "system", "content": m.content}

    def _call(self, messages, **kwargs):
        payload = {
            "model": "openai/gpt-4.1-mini",
            "messages": [self._convert_message(m) for m in messages],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        headers = {
            "Authorization": f"Bearer {os.environ['GITHUB_API_TOKEN']}",
            "Content-Type": "application/json",
        }
        response = httpx.post("https://models.github.ai/inference/chat/completions", json=payload, headers=headers, timeout=30.0)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _generate(self, messages, stop=None, **kwargs):
        content = self._call(messages, **kwargs)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    @property
    def _llm_type(self) -> str:
        return "github-chat-model"


def get_agent(session_id: str) -> BaseChatModel:
    llm = GitHubChatModel(temperature=0.3, max_tokens=256)

    tools = [
        search_quotes_by_vertical,
        find_best_quote_for_quantity, 
        search_best_quote,
        search_quotes_tool,
        check_procurement_policy,
        procurement_knowledge_tool, 
        vendor_management_tool,     
        procurement_requirements_form
    ]
    
    memory = get_memory(session_id=session_id)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": DEFAULT_AGENT_PREFIX,
            "suffix": DEFAULT_AGENT_SUFFIX,
        },
    )
    return agent
