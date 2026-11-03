"""
Telegram logistics voice-note bot.

Flow: Telegram webhook -> download voice note -> Gemini extracts
(container number, location/port, status) -> append row to Google Sheet
-> confirmation reply via Telegram.

Required env vars:
    TELEGRAM_BOT_TOKEN       Bot token from @BotFather
    GEMINI_API_KEY           Gemini API key (AQ...)
    GOOGLE_CREDENTIALS_JSON  Full service-account JSON (as a single string)
    GOOGLE_SHEET_ID          ID of the target Google Sheet (from its URL)

Optional env vars:
    GOOGLE_SHEET_TAB         Worksheet/tab name (defaults to the first tab)
    TELEGRAM_WEBHOOK_SECRET  Secret token set via setWebhook (recommended)
    ALLOWED_CHAT_IDS         Comma-separated chat IDs allowed to use the bot
    GEMINI_MODEL             Defaults to gemini-3.8-flash
"""

import json
import logging
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import gspread
import requests
from flask import Flask, abort, jsonify, request
from google import genai
from google.genai import types
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("logistics-bot")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON", "")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID", "")
GOOGLE_SHEET_TAB = os.environ.get("GOOGLE_SHEET_TAB", "")
WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
ALLOWED_CHAT_IDS = {
    c.strip() for c in os.environ.get("ALLOWED_CHAT_IDS", "").split(",") if c.strip()
}

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
TELEGRAM_FILE_API = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}"

SHEET_HEADERS = [
    "Timestamp (UTC)",
    "Telegram Chat ID",
    "Container Number",
    "Current Location/Port",
    "Status Update",
]

EXTRACTION_PROMPT = """You are a logistics data-entry assistant.
Listen to the attached voice note from a logistics operator and extract
strictly these three fields:

1. container_number: The shipping container number (typically 4 letters
   followed by 7 digits, e.g. MSKU1234567). Remove spaces and hyphens and
   use uppercase. If several containers are mentioned, list them separated
   by commas.
2. location: The current location or port mentioned (e.g. "Port of Djibouti",
   "Modjo dry port").
3. status_update: A clean, concise text summary of the status update
   (what is happening with the container: customs clearance, loaded,
   delayed, in transit, delivered, etc.). Write it as 1-3 plain sentences.

Rules:
- Use only information actually spoken in the audio. Never guess or invent.
- If a field is not mentioned, return exactly "Not mentioned".
- The audio may be in any language; always write the output in English.
"""


class LogisticsData(BaseModel):
    container_number: str
    location: str
    status_update: str


# --------------------------------------------------------------------------- #
# Lazy, cached clients
# --------------------------------------------------------------------------- #
_client_lock = threading.Lock()
_gemini_client = None
_worksheet = None


def get_gemini_client():
    global _gemini_client
    with _client_lock:
        if _gemini_client is None:
            _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        return _gemini_client


def get_worksheet():
    """Open the worksheet once and make sure the header row exists."""
    global _worksheet
    with _client_lock:
        if _worksheet is None:
            creds = json.loads(GOOGLE_CREDENTIALS_JSON)
            gc = gspread.service_account_from_dict(creds)
            sheet = gc.open_by_key(GOOGLE_SHEET_ID)
            ws = sheet.worksheet(GOOGLE_SHEET_TAB) if GOOGLE_SHEET_TAB else sheet.sheet1
            if not ws.acell("A1").value:
                ws.append_row(SHEET_HEADERS, value_input_option="RAW")
            _worksheet = ws
        return _worksheet


# --------------------------------------------------------------------------- #
# Telegram helpers
# --------------------------------------------------------------------------- #
def send_message(chat_id, text):
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text[:4000]},
            timeout=15,
        )
        if not resp.ok:
            log.error("sendMessage failed: %s %s", resp.status_code, resp.text)
    except requests.RequestException:
        log.exception("sendMessage request error")


def download_telegram_file(file_id):
    """Return the raw bytes of a Telegram file."""
    meta = requests.get(
        f"{TELEGRAM_API}/getFile", params={"file_id": file_id}, timeout=15
    )
    meta.raise_for_status()
    file_path = meta.json()["result"]["file_path"]
    content = requests.get(f"{TELEGRAM_FILE_API}/{file_path}", timeout=60)
    content.raise_for_status()
    return content.content


# --------------------------------------------------------------------------- #
# Gemini extraction
# --------------------------------------------------------------------------- #
def extract_logistics_data(audio_bytes):
    response = get_gemini_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            EXTRACTION_PROMPT,
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg"),
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=LogisticsData,
        ),
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty response")

    data = json.loads(response.text)
    return {
        "container_number": str(data.get("container_number") or "Not mentioned").strip(),
        "location": str(data.get("location") or "Not mentioned").strip(),
        "status_update": str(data.get("status_update") or "Not mentioned").strip(),
    }


# --------------------------------------------------------------------------- #
# Sheet logging
# --------------------------------------------------------------------------- #
def log_to_sheet(chat_id, data):
    row = [
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        str(chat_id),
        data["container_number"],
        data["location"],
        data["status_update"],
    ]
    # RAW stores values as plain text (no formula interpretation).
    get_worksheet().append_row(row, value_input_option="RAW")


# --------------------------------------------------------------------------- #
# Update processing (runs in a background thread)
# --------------------------------------------------------------------------- #
def process_update(update):
    message = update.get("message") or update.get("channel_post") or {}
    chat_id = (message.get("chat") or {}).get("id")
    if chat_id is None:
        return

    if ALLOWED_CHAT_IDS and str(chat_id) not in ALLOWED_CHAT_IDS:
        log.warning("Ignoring message from unauthorized chat %s", chat_id)
        return

    voice = message.get("voice")
    audio = message.get("audio")
    media = voice or audio

    if not media:
        text = (message.get("text") or "").strip()
        if text.startswith("/start"):
            send_message(
                chat_id,
                "Hi! Send me a voice note with the container number, "
                "current location/port and status, and I'll log it.",
            )
        elif text:
            send_message(chat_id, "Please send a voice note (not text) to log an update.")
        return

    try:
        send_message(chat_id, "🎧 Got it, processing your voice note...")
        audio_bytes = download_telegram_file(media["file_id"])
        data = extract_logistics_data(audio_bytes)
        log_to_sheet(chat_id, data)
        send_message(
            chat_id,
            "✅ Logged successfully\n\n"
            f"Container: {data['container_number']}\n"
            f"Location/Port: {data['location']}\n"
            f"Status: {data['status_update']}",
        )
    except Exception:
        log.exception("Failed to process voice note for chat %s", chat_id)
        send_message(
            chat_id,
            "❌ Sorry, I couldn't process that voice note. Please try again.",
        )


# --------------------------------------------------------------------------- #
# Flask app
# --------------------------------------------------------------------------- #
app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)

# Telegram retries webhooks it thinks failed; remember recent update IDs
# so the same voice note is never logged twice.
_seen_updates = OrderedDict()
_seen_lock = threading.Lock()
_SEEN_MAX = 1000


def already_seen(update_id):
    if update_id is None:
        return False
    with _seen_lock:
        if update_id in _seen_updates:
            return True
        _seen_updates[update_id] = True
        while len(_seen_updates) > _SEEN_MAX:
            _seen_updates.popitem(last=False)
        return False


@app.get("/")
def health():
    return jsonify(status="ok"), 200


@app.post("/webhook")
def webhook():
    if WEBHOOK_SECRET:
        if request.headers.get("X-Telegram-Bot-Api-Secret-Token") != WEBHOOK_SECRET:
            abort(403)

    update = request.get_json(silent=True)
    if not update:
        return jsonify(ok=True), 200

    if not already_seen(update.get("update_id")):
        # Acknowledge immediately; do the slow work in the background.
        executor.submit(process_update, update)

    return jsonify(ok=True), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
