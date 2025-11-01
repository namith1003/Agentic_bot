from ast import Dict
import json
import os
from typing import Any, List
import azure.functions as func
import logging

import bot

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="http_trigger")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
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


@app.route(route="whatsapp")
def whatsapp(req: func.HttpRequest) -> func.HttpResponse:
    """Single route handling both GET (verification) and POST (messages)."""
    if req.method == "GET":
        mode = req.params.get("hub.mode")
        token = req.params.get("hub.verify_token")
        challenge = req.params.get("hub.challenge")
        verify = os.getenv("WHATSAPP_VERIFY_TOKEN") or os.getenv("META_VERIFY_TOKEN")
        if mode == "subscribe" and token and verify and token == verify:
            return func.HttpResponse(body=str(challenge or ""), status_code=200)
        return func.HttpResponse(status_code=403)

    # POST: Message reception
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
        business_id = os.getenv("DEFAULT_BUSINESS_ID", "default")
        try:
            reply = bot.answer_query(text, business_id=business_id)
            _send_meta_reply(pnid, waid, reply)
        except Exception:
            logging.exception("Failed to process incoming WhatsApp message")

    return func.HttpResponse(body="EVENT_RECEIVED", status_code=200)

@app.route(route="http_trigger")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
    

@app.route(route="health")
def health(_: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(body=json.dumps({"status": "ok"}), mimetype="application/json", status_code=200)