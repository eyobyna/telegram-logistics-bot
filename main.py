import os
import requests
import gspread
from google.oauth2.service_account import Credentials
from flask import Flask, request

app = Flask(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GOOGLE_SHEET_NAME = os.environ.get("GOOGLE_SHEET_NAME", "Logistics_Tracking")

def get_google_sheet():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds_json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if creds_json_str:
        import json
        creds_dict = json.loads(creds_json_str)
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    else:
        creds = Credentials.from_service_account_file("credentials.json", scopes=scope)
        
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
            
            temp_audio_path = "voice_note.ogg"
            with open(temp_audio_path, "wb") as f:
                f.write(voice_bytes)
                
            # 2. Upload file to Gemini using Bearer token authentication
            headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
            file_size = os.path.getsize(temp_audio_path)
            
            init_url = "https://generativelanguage.googleapis.com/upload/v1beta/files"
            init_headers = {
                **headers,
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(file_size),
                "X-Goog-Upload-Header-Content-Type": "audio/ogg",
                "Content-Type": "application/json"
            }
            init_resp = requests.post(init_url, headers=init_headers, json={"file": {"display_name": "voice_note.ogg"}}).headers
            upload_url = init_resp.get("X-Goog-Upload-URL")
            
            upload_headers = {
                **headers,
                "X-Goog-Upload-Command": "upload, finalize",
                "X-Goog-Upload-Offset": "0",
                "Content-Length": str(file_size)
            }
            with open(temp_audio_path, "rb") as audio_f:
                upload_resp = requests.post(upload_url, headers=upload_headers, data=audio_f).json()
            
            file_name = upload_resp.get("file", {}).get("name")
            
            # 3. Generate content via direct REST API
            prompt = (
                "Listen to this voice note from a logistics agent. Extract the following fields strictly as a text summary "
                "separated by commas: Container Number, Current Location/Port, Status Update. "
                "Do not add extra conversational text."
            )
            generate_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"file_data": {"file_uri": f"https://generativelanguage.googleapis.com/v1beta/{file_name}", "mime_type": "audio/ogg"}},
                            {"text": prompt}
                        ]
                    }
                ]
            }
            gen_resp = requests.post(generate_url, headers={**headers, "Content-Type": "application/json"}, json=payload).json()
            extracted_data = gen_resp["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
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
