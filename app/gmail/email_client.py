import base64
from email.mime.text import MIMEText
from googleapiclient.discovery import build
from app.gmail.auth import get_gmail_credentials

def create_message(to: str, subject: str, body: str) -> dict:
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {"raw": raw}

def send_email(to: str, subject: str, body: str) -> str:
    creds = get_gmail_credentials()
    service = build("gmail", "v1", credentials=creds)
    message = create_message(to, subject, body)
    result = service.users().messages().send(userId="me", body=message).execute()
    return f"Email sent to {to}, ID: {result['id']}"
