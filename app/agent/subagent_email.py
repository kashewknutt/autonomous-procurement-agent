from langchain.agents import initialize_agent, AgentType
from app.agent.core import GitHubChatModel
from app.agent.subtools.gmail_email_tool import send_procurement_email

def get_email_agent():
    llm = GitHubChatModel()
    tools = [send_procurement_email]

    email_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": """You are an email assistant. You can ONLY send procurement-related emails using the tool below.

Tool:
- send_procurement_email: Sends a Gmail message to a supplier. Input should include 'to', 'subject', and 'body' as JSON string.

If you are asked to send an email, fill in the required fields and invoke the tool.

Example:

User: Email supplier@tools.com about the missing invoice
Thought: I need to send an email
Action: send_procurement_email
Action Input: {"to": "supplier@tools.com", "subject": "Missing Invoice", "body": "Hi, we haven\'t received the invoice for PO#1234."}
""",
            "suffix": """
Question: {input}
Thought: {agent_scratchpad}"""
        }
    )
    return email_agent
