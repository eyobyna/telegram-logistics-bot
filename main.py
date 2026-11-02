import os
import base64
import json
import gspread
from google.oauth2.service_account import Credentials
from flask import Flask, request
from google import genai
from google.genai import types
import requests

app = Flask(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GOOGLE_SHEET_NAME = os.environ.get("GOOGLE_SHEET_NAME", "Logistics_Tracking")

# Check multiple potential environment variable names to prevent None errors
gemini_key = (
    os.environ.get("CUSTOM_GEMINI_API_KEY") or 
    os.environ.get("GEMINI_API_KEY") or 
    os.environ.get("CUSTOM_GEMINI_KEY")
)

if not gemini_key:
    print("CRITICAL WARNING: No Gemini API key found in environment variables!")
    print("Available environment variables:", list(os.environ.keys()))

# Initialize the official Gemini client safely
client = genai.Client(api_key=gemini_key)

def get_google_credentials():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds_json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if creds_json_str:
        creds_dict = json.loads(creds_json_str)
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    else:
        creds = Credentials.from_service_account_file("credentials.json", scopes=scope)
    return creds

def get_google_sheet():
    creds = get_google_credentials()
    gspread_client = gspread.authorize(creds)
    return gspread_client.open(GOOGLE_SHEET_NAME).sheet1

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
            
            # 2. Call Gemini using the official SDK (passing audio bytes directly)
            prompt_text = (
                "Listen to this voice note from a logistics agent. Extract the following fields strictly as a text summary "
                "separated by commas: Container Number, Current Location/Port, Status Update. "
                "Do not add extra conversational text."
            )
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    types.Part.from_bytes(
                        data=voice_bytes,
                        mime_type='audio/ogg',
                    ),
                    prompt_text
                ]
            )
            
            extracted_data = response.text.strip()
            
            # 3. Log to Google Sheets
            sheet = get_google_sheet()
            sheet.append_row([str(chat_id), extracted_data])
            
            # 4. Reply back on Telegram
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
