import os
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory

def get_memory(session_id: str):
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    chat_history = RedisChatMessageHistory(
        session_id=session_id,
        url=redis_url
    )

    return ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=chat_history,
        return_messages=True,
        k=4
    )
