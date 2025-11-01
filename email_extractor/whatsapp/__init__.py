import os
import json
import logging
import requests
import azure.functions as func
from typing import Any, Dict

# Try to import bot logic (answer_query) from sibling module
answer_query = None
try:
    # Prefer importing from parent folder (email_extractor)
    from ..bot import answer_query as _answer_query  # type: ignore
    answer_query = _answer_query
except Exception:
    try:
        # Fallback if package import fails
        from bot import answer_query as _answer_query  # type: ignore
        answer_query = _answer_query
    except Exception:
        answer_query = None


def _send_whatsapp_text(to_number: str, text: str, phone_number_id: str | None = None) -> Dict[str, Any]:
    token = os.getenv("META_WABA_TOKEN") or os.getenv("WHATSAPP_TOKEN") or ""
    pnid = phone_number_id or os.getenv("META_PHONE_NUMBER_ID") or ""
    if not token or not pnid:
        return {"error": "missing META_WABA_TOKEN/WHATSAPP_TOKEN or META_PHONE_NUMBER_ID"}
    url = f"https://graph.facebook.com/v20.0/{pnid}/messages"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": text[:4096]},
    }
    try:
        r = requests.post(url, headers=headers, json=data, timeout=15)
        return {"status": r.status_code, "response": r.text}
    except Exception as e:
        return {"error": str(e)}


def main(req: func.HttpRequest) -> func.HttpResponse:
    method = req.method.upper()

    # Verification (GET)
    if method == "GET":
        mode = req.params.get("hub.mode")
        token = req.params.get("hub.verify_token")
        challenge = req.params.get("hub.challenge")
        verify_token = (
            os.getenv("META_VERIFY_TOKEN")
            or os.getenv("WHATSAPP_VERIFY_TOKEN")
            or os.getenv("VERIFY_TOKEN")
        )
        if mode == "subscribe" and token and verify_token and token == verify_token:
            return func.HttpResponse(challenge or "", status_code=200, mimetype="text/plain")
        return func.HttpResponse("forbidden", status_code=403)

    # Events (POST)
    try:
        payload = req.get_json()
    except ValueError:
        payload = {}

    # Accept only WABA events and acknowledge quickly to avoid retries
    if payload.get("object") != "whatsapp_business_account":
        return func.HttpResponse("EVENT_RECEIVED", status_code=200)

    entries = payload.get("entry", [])
    for entry in entries:
        changes = entry.get("changes", [])
        for change in changes:
            value = change.get("value", {})
            metadata = value.get("metadata", {})
            phone_number_id = metadata.get("phone_number_id")
            messages = value.get("messages", [])
            for msg in messages:
                from_number = msg.get("from")
                msg_type = msg.get("type")
                if msg_type == "text":
                    text_body = (msg.get("text", {}).get("body") or "").strip()
                else:
                    text_body = ""
                if not from_number or not text_body:
                    continue

                # Default business selection
                business_id = "default"

                # Generate reply
                try:
                    if callable(answer_query):
                        reply = answer_query(text_body, business_id=business_id)  # type: ignore
                    else:
                        reply = "Thanks for your message. We'll get back to you shortly."
                except Exception as e:
                    logging.exception("answer_query error: %s", e)
                    reply = "Maaf, sistem mengalami masalah sebentar. Cuba lagi nanti."

                # Send reply via Graph API
                _send_whatsapp_text(from_number, reply, phone_number_id=phone_number_id)

    # Always acknowledge receipt to Meta
    return func.HttpResponse("EVENT_RECEIVED", status_code=200)
