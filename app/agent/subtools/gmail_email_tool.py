from langchain.tools import tool
from app.gmail.email_client import send_email

@tool
def send_procurement_email(input: str) -> str:
    """
    Sends a procurement-related email using Gmail.
    Input format (JSON as string): 
    {
        "to": "email@example.com", 
        "subject": "Quote Request",
        "body": "Dear supplier, we need quotes for..."
    }
    """
    import json
    try:
        payload = json.loads(input)
        return send_email(payload["to"], payload["subject"], payload["body"])
    except Exception as e:
        return f"Failed to send email: {str(e)}"
