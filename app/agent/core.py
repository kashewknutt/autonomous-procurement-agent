from langchain.agents import initialize_agent, AgentType
from langchain_huggingface import HuggingFaceEndpoint  # Updated import path
from app.agent.tools import search_quotes_tool, get_best_quote_under_budget, check_procurement_policy
from app.agent.memory import get_memory
from app.agent.prompts import DEFAULT_AGENT_PREFIX, DEFAULT_AGENT_SUFFIX
import os


def get_agent():
    llm = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        temperature=0.3,
        max_new_tokens=512
    )

    tools = [search_quotes_tool, get_best_quote_under_budget, check_procurement_policy]
    memory = get_memory()

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
