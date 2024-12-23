import os
import base64
import csv
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from bs4 import BeautifulSoup

# Define the scopes for Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Authenticate and build the Gmail service
def gmail_authenticate():
    creds = None
    # Load credentials if they exist
    if os.path.exists('token2.json'):
        creds = Credentials.from_authorized_user_file('token2.json', SCOPES)
    # Request login if necessary
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret_399928458301-36igi2gajktulcp5506qd7eaqc4k9dfv.apps.googleusercontent.com.json', SCOPES)
            creds = flow.run_local_server(port=3000)
        # Save credentials
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

# Export all email details to CSV for spam detection
def export_emails_to_csv():
    service = gmail_authenticate()
    # Open CSV file in write mode
    with open('emails1.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write CSV header
        writer.writerow(['Subject', 'From', 'To', 'Date', 'Message-ID', 'Thread-ID', 'Body', 'Labels'])

        # Retrieve messages with pagination
        next_page_token = None
        while True:
            results = service.users().messages().list(userId='me', labelIds=['INBOX'], pageToken=next_page_token).execute()
            messages = results.get('messages', [])

            for message in messages:
                msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()

                # Extract email details
                payload = msg['payload']
                headers = payload['headers']

                # Get basic headers like Subject, From, To, Date, Message-ID, and Thread-ID
                subject = next((header['value'] for header in headers if header['name'] == 'Subject'), "No Subject")
                sender = next((header['value'] for header in headers if header['name'] == 'From'), "Unknown Sender")
                recipient = next((header['value'] for header in headers if header['name'] == 'To'), "Unknown Recipient")
                date = next((header['value'] for header in headers if header['name'] == 'Date'), "No Date")
                message_id = next((header['value'] for header in headers if header['name'] == 'Message-ID'), "No Message-ID")
                thread_id = next((header['value'] for header in headers if header['name'] == 'Thread-ID'), "No Thread-ID")
                labels = msg.get('labelIds', [])

                # Get email body
                body = "No Body Content"
                if 'parts' in payload:
                    part = payload['parts'][0]
                    if 'body' in part and 'data' in part['body']:
                        data = part['body']["data"]
                        decoded_data = base64.urlsafe_b64decode(data).decode("utf-8")
                        soup = BeautifulSoup(decoded_data, "html.parser")
                        body = soup.get_text()
                elif 'body' in payload and 'data' in payload['body']:
                    data = payload['body']["data"]
                    decoded_data = base64.urlsafe_b64decode(data).decode("utf-8")
                    soup = BeautifulSoup(decoded_data, "html.parser")
                    body = soup.get_text()

                # Write email data as a row in the CSV
                writer.writerow([subject, sender, recipient, date, message_id, thread_id, body, labels])

            # Check if there are more pages of results
            next_page_token = results.get('nextPageToken')
            if not next_page_token:
                break  # Exit loop if no more pages are available

# Run the CSV export function
export_emails_to_csv()