"""
Telegram logistics & finance bot (voice notes, text messages, photos/scans, PDFs).

Flow: Telegram webhook -> read the input -> Gemini extracts logistics data
and calculates financials (Revenue, VAT, WHT, Expenses, Net Profit) 
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
from typing import List, Optional

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
log = logging.getLogger("logistics-finance-bot")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON", "")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID", "")
GOOGLE_SHEET_TAB = os.environ.get("GOOGLE_SHEET_TAB", "")
WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.1-flash-lite")
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
    "Container/Job Ref",
    "Client Name",
    "Service/Activity Description",
    "Current Location/Port",
    "Gross Revenue Amount",
    "Calculated VAT",
    "Calculated WHT",
    "Total Expenses",
    "Net Profit",
    "Status Update",
    "Input Type",
]

EXTRACTION_PROMPT = """You are an expert logistics accountant and data-entry assistant for Solomon Zewdu Shipping & Freight Forwarding Company.
The input is one of: a voice note, a typed message, or a photo/scan/PDF of a paper document (such as an invoice, receipt, waybill, delivery order, bill of lading, gate pass, or financial update).

For each job/container mentioned or found in the document, extract and calculate the following fields accurately:

1. container_number: The shipping container number or job reference (e.g. MSKU1234567, or a job reference string). Use uppercase, remove unnecessary spaces. If none, return "Not mentioned".
2. client_name: The client or company being billed or served. If not mentioned, return "Not mentioned".
3. service_description: Short description of the service/activity or expense item.
4. location: The current location or port mentioned (e.g. "Port of Djibouti", "Modjo dry port", "Addis Ababa"). If none, return "Not mentioned".
5. gross_revenue: Total gross revenue amount as a numeric value (e.g., 50000). Put 0 if it's purely an expense receipt or not mentioned.
6. vat: Calculated VAT amount based on standard local tax laws or stated values. Put 0 if not applicable.
7. witholding_tax: Calculated Withholding Tax (WHT) amount. Put 0 if not applicable.
8. total_expenses: Total expenses incurred (extracted from receipts/text). Put 0 if none.
9. net_profit: Net profit calculated as (gross_revenue minus total_expenses minus vat and withholding tax where applicable). Put 0 if it's strictly an expense entry.
10. status_update: A clean, concise 1-3 sentence summary of the status, financial note, or operational milestone.

Rules:
- Use only information actually present in or calculable from the input. Never guess or invent numbers.
- If multiple entries or containers appear, return a record for each.
- The input may be in any language; always write the output in English.
"""


class LogisticsFinancialRecord(BaseModel):
    container_number: str
    client_name: str
    service_description: str
    location: str
    gross_revenue: float
    vat: float
    witholding_tax: float
    total_expenses: float
    net_profit: float
    status_update: str


class FinancialExtraction(BaseModel):
    records: List[LogisticsFinancialRecord]


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
            _gemini_client = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options=types.HttpOptions(
                    timeout=60_000,  # 60 s per request
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
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
        m = message["photo"][-1]
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
RETRY_DELAYS = [3, 8]


def generate_with_retry(contents):
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=FinancialExtraction,
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
                if getattr(exc, "code", None) not in RETRYABLE_CODES:
                    if model == GEMINI_MODEL:
                        raise
                    log.warning("Fallback model %s failed: %s", model, exc)
                    break
                last_exc = exc
                log.warning(
                    "Gemini %s error on %s (attempt %d)", exc.code, model, attempt + 1
                )
                if attempt < len(RETRY_DELAYS):
                    time.sleep(RETRY_DELAYS[attempt])
    raise last_exc


def extract_logistics_financial_data(inp, media_bytes=None):
    contents = [EXTRACTION_PROMPT]

    if inp["kind"] == "Text":
        contents.append("Message text:\n" + inp["text"])
    else:
        contents.append(types.Part.from_bytes(data=media_bytes, mime_type=inp["mime"]))
        if inp.get("caption"):
            contents.append("Sender's caption / notes: " + inp["caption"])

    response = generate_with_retry(contents)
    if not response.text:
        raise RuntimeError("Gemini returned an empty response")

    raw = json.loads(response.text)
    records = []
    for item in (raw.get("records") or [])[:MAX_RECORDS]:
        records.append(
            {
                "container_number": str(item.get("container_number") or "Not mentioned").strip(),
                "client_name": str(item.get("client_name") or "Not mentioned").strip(),
                "service_description": str(item.get("service_description") or "Not mentioned").strip(),
                "location": str(item.get("location") or "Not mentioned").strip(),
                "gross_revenue": float(item.get("gross_revenue") or 0.0),
                "vat": float(item.get("vat") or 0.0),
                "witholding_tax": float(item.get("witholding_tax") or 0.0),
                "total_expenses": float(item.get("total_expenses") or 0.0),
                "net_profit": float(item.get("net_profit") or 0.0),
                "status_update": str(item.get("status_update") or "Not mentioned").strip(),
            }
        )
    return records


# --------------------------------------------------------------------------- #
# Sheet logging & formatting
# --------------------------------------------------------------------------- #
def log_to_sheet(chat_id, records, input_type):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        [
            timestamp,
            str(chat_id),
            r["container_number"],
            r["client_name"],
            r["service_description"],
            r["location"],
            r["gross_revenue"],
            r["vat"],
            r["witholding_tax"],
            r["total_expenses"],
            r["net_profit"],
            r["status_update"],
            input_type,
        ]
        for r in records
    ]
    get_worksheet().append_rows(rows, value_input_option="RAW")


def format_confirmation(records):
    if len(records) == 1:
        r = records[0]
        return (
            "✅ Financial & Logistics Record Logged\n\n"
            f"Ref/Container: {r['container_number']}\n"
            f"Client: {r['client_name']}\n"
            f"Service: {r['service_description']}\n"
            f"Location: {r['location']}\n"
            f"Revenue: {r['gross_revenue']:,.2f}\n"
            f"VAT: {r['vat']:,.2f} | WHT: {r['witholding_tax']:,.2f}\n"
            f"Expenses: {r['total_expenses']:,.2f}\n"
            f"Net Profit: {r['net_profit']:,.2f}\n"
            f"Status: {r['status_update']}"
        )

    lines = [f"✅ Logged {len(records)} financial entries\n"]
    for i, r in enumerate(records[:10], start=1):
        lines.append(
            f"{i}. Ref: {r['container_number']} | Client: {r['client_name']} | Net Profit: {r['net_profit']:,.2f}"
        )
    if len(records) > 10:
        lines.append(f"...and {len(records) - 10} more rows saved in sheet.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Update processing
# --------------------------------------------------------------------------- #
HELP_TEXT = (
    "Solomon Zewdu Shipping - Finance & Operations Bot\n\n"
    "Send me a voice note, text update, receipt photo, scan, or invoice PDF.\n"
    "I will automatically extract container numbers, calculate VAT, WHT, "
    "expenses, and net profit, and record it in the central finance sheet."
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
            "Unsupported format. Please send text, voice notes, images/receipts, or PDFs.",
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
                send_message(chat_id, "❌ File is too large (limit is 18 MB).")
                return
            send_message(chat_id, "⏳ Processing financial record...")
            media_bytes = download_telegram_file(inp["file_id"])
            if len(media_bytes) > MAX_FILE_BYTES:
                send_message(chat_id, "❌ File is too large (limit is 18 MB).")
                return
        else:
            send_message(chat_id, "⏳ Processing financial record...")

        records = extract_logistics_financial_data(inp, media_bytes)

        if not records:
            send_message(
                chat_id,
                "Could not extract any logistics or financial details from that input. Please try again.",
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
                "⏳ AI service is busy. Please send your update again shortly.",
            )
        else:
            send_message(
                chat_id,
                "❌ Error processing input. Please try again.\n"
                f"(Error type: {type(exc).__name__})",
            )


# --------------------------------------------------------------------------- #
# Flask app
# --------------------------------------------------------------------------- #
app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)

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
        executor.submit(process_update, update)

    return jsonify(ok=True), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
