import os
import json
import logging
import azure.functions as func
from typing import Any, Dict, List

# Import the core bot logic (RAG+LLM) from bot.py
import bot  # type: ignore


app = func.FunctionApp()


@app.route(route="health", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def health(_: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(
        body=json.dumps({"status": "ok"}),
        mimetype="application/json",
        status_code=200,
    )


def _get_body_text_from_meta(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract sender WA ID and text body from Meta WhatsApp Cloud API webhook payload.
    Returns list of dicts: {"from": waid, "text": body, "phone_number_id": pnid}
    """
    messages: List[Dict[str, str]] = []
    try:
        entries = payload.get("entry", [])
        for e in entries:
            for change in e.get("changes", []):
                value = change.get("value", {})
                metadata = value.get("metadata", {})
                phone_number_id = metadata.get("phone_number_id")
                for m in value.get("messages", []) or []:
                    waid = m.get("from")
                    text_body = (m.get("text", {}) or {}).get("body") if m.get("type") == "text" else None
                    if waid and text_body:
                        messages.append({
                            "from": str(waid),
                            "text": str(text_body),
                            "phone_number_id": str(phone_number_id) if phone_number_id else "",
                        })
    except Exception:
        # If structure unexpected, just return empty
        pass
    return messages


def _send_meta_reply(phone_number_id: str, to_waid: str, text: str) -> None:
    """Send a reply via WhatsApp Cloud API if credentials are configured."""
    import requests  # local import to avoid dependency unless used
    token = os.getenv("WHATSAPP_CLOUD_ACCESS_TOKEN") or os.getenv("META_WHATSAPP_ACCESS_TOKEN")
    api_ver = os.getenv("META_GRAPH_API_VERSION", "v20.0")
    if not token or not phone_number_id:
        return
    url = f"https://graph.facebook.com/{api_ver}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_waid,
        "type": "text",
        "text": {"body": text[:4096]},
    }
    try:
        requests.post(url, headers=headers, json=payload, timeout=10)
    except Exception:
        logging.exception("Failed to send WhatsApp Cloud API reply")


@app.route(route="webhook", methods=["GET", "POST"], auth_level=func.AuthLevel.ANONYMOUS)
def whatsapp_webhook(req: func.HttpRequest) -> func.HttpResponse:
    # Meta verification (GET)
    if req.method == "GET":
        mode = req.params.get("hub.mode")
        token = req.params.get("hub.verify_token")
        challenge = req.params.get("hub.challenge")
        verify = os.getenv("WHATSAPP_VERIFY_TOKEN") or os.getenv("META_VERIFY_TOKEN")
        if mode == "subscribe" and token and verify and token == verify:
            return func.HttpResponse(body=str(challenge or ""), status_code=200)
        return func.HttpResponse(status_code=403)

    # Message reception (POST)
    try:
        payload = req.get_json()
    except ValueError:
        payload = None

    if not isinstance(payload, dict):
        return func.HttpResponse(status_code=400, body="invalid payload")

    messages = _get_body_text_from_meta(payload)
    for m in messages:
        text = m.get("text") or ""
        waid = m.get("from") or ""
        pnid = m.get("phone_number_id") or ""
        # Route to a business tenant if you wish; default for now
        business_id = os.getenv("DEFAULT_BUSINESS_ID", "default")
        try:
            reply = bot.answer_query(text, business_id=business_id)
            # Attempt to send reply back to WhatsApp Cloud API (optional)
            _send_meta_reply(pnid, waid, reply)
        except Exception:
            logging.exception("Failed to process incoming WhatsApp message")

    # Respond 200 to acknowledge receipt
    return func.HttpResponse(body="EVENT_RECEIVED", status_code=200)