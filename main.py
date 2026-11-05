"""
Telegram logistics bot (voice notes, text messages, photos/scans, PDFs).

Flow: Telegram webhook -> read the input (voice / text / photo / image or PDF
document) -> Gemini extracts (container number, location/port, status)
-> append row(s) to Google Sheet -> confirmation reply via Telegram.

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
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import List

import gspread
import requests
from flask import Flask, abort, jsonify, request
from google import genai
from google.genai import errors as genai_errors
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
GEMINI_FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "")
ALLOWED_CHAT_IDS = {
    c.strip() for c in os.environ.get("ALLOWED_CHAT_IDS", "").split(",") if c.strip()
}

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
TELEGRAM_FILE_API = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}"

MAX_FILE_BYTES = 18 * 1024 * 1024  # keep under Gemini's ~20 MB inline limit
MAX_RECORDS = 50

SHEET_HEADERS = [
    "Timestamp (UTC)",
    "Telegram Chat ID",
    "Container Number",
    "Current Location/Port",
    "Status Update",
    "Input Type",
]

EXTRACTION_PROMPT = """You are a logistics data-entry assistant.
The input is one of: a voice note, a typed message, or a photo/scan/PDF of a
paper document (for example a delivery order, bill of lading, gate pass,
inspection sheet, handwritten note, or shipping update).

Extract strictly these three fields for each container:

1. container_number: The shipping container number (typically 4 letters
   followed by 7 digits, e.g. MSKU1234567). Remove spaces and hyphens and
   use uppercase.
2. location: The current location or port mentioned (e.g. "Port of Djibouti",
   "Modjo dry port"). For documents, use the current/last known location,
   or the port named on the document if no current location is given.
3. status_update: A clean, concise text summary of the status update
   (customs clearance, loaded, delayed, in transit, delivered, inspection,
   etc.). Write it as 1-3 plain sentences.

Rules:
- Use only information actually present in the input. Never guess or invent.
- If a field is not present for a container, return exactly "Not mentioned".
- If several containers appear, return one record per container.
- If the input mentions a status or location but no container number,
  return one record with container_number "Not mentioned".
- If the input contains no logistics information at all, return an empty
  records list.
- The input may be in any language; always write the output in English.
"""


class LogisticsRecord(BaseModel):
    container_number: str
    location: str
    status_update: str


class LogisticsExtraction(BaseModel):
    records: List[LogisticsRecord]


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
# Input classification
# --------------------------------------------------------------------------- #
def classify_message(message):
    """
    Work out what the user sent. Returns a dict with:
      kind:    Voice | Audio | Photo | Document | Text | Unsupported
      file_id, mime, size, caption (for file inputs) or text (for text input)
    or None when there is nothing to process.
    """
    caption = (message.get("caption") or "").strip()

    if message.get("voice"):
        m = message["voice"]
        return {
            "kind": "Voice",
            "file_id": m["file_id"],
            "mime": "audio/ogg",
            "size": m.get("file_size"),
            "caption": caption,
        }

    if message.get("audio"):
        m = message["audio"]
        return {
            "kind": "Audio",
            "file_id": m["file_id"],
            "mime": m.get("mime_type") or "audio/mpeg",
            "size": m.get("file_size"),
            "caption": caption,
        }

    if message.get("photo"):
        m = message["photo"][-1]  # largest available size
        return {
            "kind": "Photo",
            "file_id": m["file_id"],
            "mime": "image/jpeg",
            "size": m.get("file_size"),
            "caption": caption,
        }

    if message.get("document"):
        m = message["document"]
        mime = (m.get("mime_type") or "").lower()
        if mime.startswith("image/") or mime == "application/pdf":
            return {
                "kind": "Document",
                "file_id": m["file_id"],
                "mime": mime,
                "size": m.get("file_size"),
                "caption": caption,
            }
        return {"kind": "Unsupported"}

    text = (message.get("text") or "").strip()
    if text:
        return {"kind": "Text", "text": text}

    return None


# --------------------------------------------------------------------------- #
# Gemini extraction
# --------------------------------------------------------------------------- #
RETRYABLE_CODES = {429, 500, 502, 503, 504}
RETRY_DELAYS = [3, 8, 15]  # seconds to wait between attempts


def generate_with_retry(contents):
    """
    Call Gemini, retrying temporary errors (e.g. 503 'model is overloaded').
    If GEMINI_FALLBACK_MODEL is set, it is tried after the main model.
    """
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=LogisticsExtraction,
    )
    models = [GEMINI_MODEL]
    if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL != GEMINI_MODEL:
        models.append(GEMINI_FALLBACK_MODEL)

    last_exc = None
    for model in models:
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                return get_gemini_client().models.generate_content(
                    model=model, contents=contents, config=config
                )
            except genai_errors.APIError as exc:
                last_exc = exc
                if getattr(exc, "code", None) not in RETRYABLE_CODES:
                    raise
                log.warning(
                    "Gemini %s error on %s (attempt %d)", exc.code, model, attempt + 1
                )
                if attempt < len(RETRY_DELAYS):
                    time.sleep(RETRY_DELAYS[attempt])
    raise last_exc


def extract_logistics_data(inp, media_bytes=None):
    contents = [EXTRACTION_PROMPT]

    if inp["kind"] == "Text":
        contents.append("Message text:\n" + inp["text"])
    else:
        contents.append(types.Part.from_bytes(data=media_bytes, mime_type=inp["mime"]))
        if inp.get("caption"):
            contents.append("Sender's caption: " + inp["caption"])

    response = generate_with_retry(contents)
    if not response.text:
        raise RuntimeError("Gemini returned an empty response")

    raw = json.loads(response.text)
    records = []
    for item in (raw.get("records") or [])[:MAX_RECORDS]:
        records.append(
            {
                "container_number": str(item.get("container_number") or "Not mentioned").strip(),
                "location": str(item.get("location") or "Not mentioned").strip(),
                "status_update": str(item.get("status_update") or "Not mentioned").strip(),
            }
        )
    return records


# --------------------------------------------------------------------------- #
# Sheet logging
# --------------------------------------------------------------------------- #
def log_to_sheet(chat_id, records, input_type):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        [
            timestamp,
            str(chat_id),
            r["container_number"],
            r["location"],
            r["status_update"],
            input_type,
        ]
        for r in records
    ]
    # RAW stores values as plain text (no formula interpretation).
    get_worksheet().append_rows(rows, value_input_option="RAW")


def format_confirmation(records):
    if len(records) == 1:
        r = records[0]
        return (
            "✅ Logged successfully\n\n"
            f"Container: {r['container_number']}\n"
            f"Location/Port: {r['location']}\n"
            f"Status: {r['status_update']}"
        )

    lines = [f"✅ Logged {len(records)} entries\n"]
    for i, r in enumerate(records[:10], start=1):
        lines.append(
            f"{i}. {r['container_number']} | {r['location']} | {r['status_update']}"
        )
    if len(records) > 10:
        lines.append(f"...and {len(records) - 10} more (all saved in the sheet).")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Update processing (runs in a background thread)
# --------------------------------------------------------------------------- #
HELP_TEXT = (
    "Hi! Send me a logistics update and I'll log it to the sheet.\n\n"
    "You can send:\n"
    "• a voice note\n"
    "• a text message\n"
    "• a photo or scan of a paper (tip: send it as a File for the sharpest "
    "result)\n"
    "• a PDF\n\n"
    "I extract the container number, current location/port and status."
)


def process_update(update):
    message = update.get("message") or update.get("channel_post") or {}
    chat_id = (message.get("chat") or {}).get("id")
    if chat_id is None:
        return

    if ALLOWED_CHAT_IDS and str(chat_id) not in ALLOWED_CHAT_IDS:
        log.warning("Ignoring message from unauthorized chat %s", chat_id)
        return

    inp = classify_message(message)
    if inp is None:
        return

    if inp["kind"] == "Unsupported":
        send_message(
            chat_id,
            "Sorry, I can only read voice notes, text, photos/images and PDFs.",
        )
        return

    if inp["kind"] == "Text":
        text = inp["text"]
        if text.startswith("/start") or text.startswith("/help"):
            send_message(chat_id, HELP_TEXT)
            return
        if text.startswith("/"):
            return

    try:
        media_bytes = None
        if inp["kind"] != "Text":
            if inp.get("size") and inp["size"] > MAX_FILE_BYTES:
                send_message(chat_id, "❌ That file is too large (limit is about 18 MB).")
                return
            send_message(chat_id, "⏳ Got it, processing...")
            media_bytes = download_telegram_file(inp["file_id"])
            if len(media_bytes) > MAX_FILE_BYTES:
                send_message(chat_id, "❌ That file is too large (limit is about 18 MB).")
                return
        else:
            send_message(chat_id, "⏳ Got it, processing...")

        records = extract_logistics_data(inp, media_bytes)

        if not records:
            send_message(
                chat_id,
                "I couldn't find any container, location or status information "
                "in that, so nothing was logged. Please try again with a clearer "
                "voice note, message or scan.",
            )
            return

        log_to_sheet(chat_id, records, inp["kind"])
        send_message(chat_id, format_confirmation(records))

    except Exception as exc:
        log.exception("Failed to process %s for chat %s", inp.get("kind"), chat_id)
        if (
            isinstance(exc, genai_errors.APIError)
            and getattr(exc, "code", None) in RETRYABLE_CODES
        ):
            send_message(
                chat_id,
                "⏳ The AI service is very busy right now. "
                "Please send that again in a minute.",
            )
        else:
            send_message(
                chat_id,
                "❌ Sorry, I couldn't process that. Please try again.\n"
                f"(Error type: {type(exc).__name__})",
            )


# --------------------------------------------------------------------------- #
# Flask app
# --------------------------------------------------------------------------- #
app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)

# Telegram retries webhooks it thinks failed; remember recent update IDs
# so the same message is never logged twice.
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
