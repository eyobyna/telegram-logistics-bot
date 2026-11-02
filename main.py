import os
import base64
import json
import requests
import gspread
from google.oauth2.service_account import Credentials
import google.auth.transport.requests
from flask import Flask, request

app = Flask(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GOOGLE_SHEET_NAME = os.environ.get("GOOGLE_SHEET_NAME", "Logistics_Tracking")

def get_google_credentials():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/generative-language"
    ]
    creds_json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if creds_json_str:
        creds_dict = json.loads(creds_json_str)
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    else:
        creds = Credentials.from_service_account_file("credentials.json", scopes=scope)
    return creds

def get_oauth_token():
    creds = get_google_credentials()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    return creds.token

def get_google_sheet():
    creds = get_google_credentials()
    client = gspread.authorize(creds)
    return client.open(GOOGLE_SHEET_NAME).sheet1

@app.route(f"/{TELEGRAM_BOT_TOKEN}", methods=["POST"])
def telegram_webhook():
    update = request.get_json()
    
    if "message" in update and "voice" in update["message"]:
        message = update["message"]
        chat_id = message["chat"]["id"]
        file_id = message["voice"]["file_id"]
        
        try:
            # 1. Download voice note from Telegram
            file_info_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}"
            file_path_resp = requests.get(file_info_url).json()
            file_path = file_path_resp["result"]["file_path"]
            
            voice_file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
            voice_bytes = requests.get(voice_file_url).content
            
            # 2. Convert audio bytes to Base64
            audio_base64 = base64.b64encode(voice_bytes).decode("utf-8")
            
            # 3. Get fresh OAuth token with correct scope and call Gemini API via REST
            access_token = get_oauth_token()
            generate_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "audio/ogg",
                                    "data": audio_base64
                                }
                            },
                            {
                                "text": (
                                    "Listen to this voice note from a logistics agent. Extract the following fields strictly as a text summary "
                                    "separated by commas: Container Number, Current Location/Port, Status Update. "
                                    "Do not add extra conversational text."
                                )
                            }
                        ]
                    }
                ]
            }
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            gen_resp = requests.post(generate_url, headers=headers, json=payload).json()
            
            if "candidates" not in gen_resp:
                raise Exception(f"Gemini API Error: {gen_resp}")
                
            extracted_data = gen_resp["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            # 4. Log to Google Sheets
            sheet = get_google_sheet()
            sheet.append_row([str(chat_id), extracted_data])
            
            # 5. Reply back on Telegram
            reply_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(reply_url, json={
                "chat_id": chat_id, 
                "text": f"Successfully logged to sheet:\n{extracted_data}"
            })
            
        except Exception as e:
            print(f"Error processing voice note: {e}")
            reply_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(reply_url, json={
                "chat_id": chat_id, 
                "text": f"Error processing update: {str(e)}"
            })
            
    return "OK", 200

@app.route("/", methods=["GET"])
def index():
    return "Bot is running live!", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
