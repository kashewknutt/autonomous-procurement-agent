from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import os
import json

def get_gmail_credentials():
    token_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "credentials.json")
    creds = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, ["https://www.googleapis.com/auth/gmail.send"])
    
    # Refresh if expired
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    return creds
