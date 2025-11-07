from ast import Dict
import json
import os
from typing import Any, List, Tuple
import azure.functions as func
import logging
from openai import OpenAI

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


def _download_meta_media(media_id: str) -> Tuple[bytes, str]:
    """Fetch media (bytes, mime_type) from Meta Cloud API using media_id."""
    import requests
    token = os.getenv("WHATSAPP_CLOUD_ACCESS_TOKEN") or os.getenv("META_WHATSAPP_ACCESS_TOKEN")
    api_ver = os.getenv("META_GRAPH_API_VERSION", "v20.0")
    if not token or not media_id:
        raise RuntimeError("Missing access token or media_id")

    meta = requests.get(
        f"https://graph.facebook.com/{api_ver}/{media_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    meta.raise_for_status()
    info = meta.json()
    url = info.get("url")
    mime = info.get("mime_type", "application/octet-stream")
    if not url:
        raise RuntimeError("No media URL returned by Meta")

    binary = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    binary.raise_for_status()
    return binary.content, mime


def _suffix_for_mime(mime: str) -> str:
    m = (mime or "").lower()
    if "audio/ogg" in m:
        return ".ogg"
    if "audio/mpeg" in m or "/mp3" in m:
        return ".mp3"
    if "audio/mp4" in m or "/m4a" in m:
        return ".m4a"
    if "audio/aac" in m:
        return ".aac"
    if "audio/amr" in m:
        return ".amr"
    return ".ogg"


def _transcribe_audio_bytes(data: bytes, mime_type: str) -> str:
    """Transcribe audio to text using OpenAI Whisper (whisper-1)."""
    if not data:
        return ""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not set; cannot transcribe audio")
        return ""
    client = OpenAI(api_key=api_key)
    import tempfile
    suffix = _suffix_for_mime(mime_type)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        with open(tmp.name, "rb") as f:
            try:
                resp = client.audio.transcriptions.create(model="whisper-1", file=f)
                return (getattr(resp, "text", None) or "").strip()
            except Exception:
                logging.exception("OpenAI transcription failed")
                return ""


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

    # Handle audio/voice messages (type=audio)
    try:
        entries = payload.get("entry", [])
    except Exception:
        entries = []
    for e in entries or []:
        for change in (e.get("changes", []) or []):
            value = change.get("value", {}) or {}
            metadata = value.get("metadata", {}) or {}
            phone_number_id = metadata.get("phone_number_id") or ""
            for m in (value.get("messages", []) or []):
                if (m.get("type") or "").lower() != "audio":
                    continue
                waid = m.get("from") or ""
                audio = m.get("audio", {}) or {}
                media_id = audio.get("id")
                mime = audio.get("mime_type", "audio/ogg")
                if not media_id:
                    continue
                # Download and transcribe
                transcript = ""
                try:
                    content, mt = _download_meta_media(media_id)
                    transcript = _transcribe_audio_bytes(content, mt or mime)
                except Exception:
                    logging.exception("Failed to download/transcribe WhatsApp audio")
                    transcript = ""
                if not transcript:
                    try:
                        _send_meta_reply(phone_number_id, waid, "Sorry, I couldn't process that voice message.")
                    except Exception:
                        logging.exception("Failed to send audio fallback reply")
                    continue
                # Answer via RAG
                try:
                    business_id = os.getenv("DEFAULT_BUSINESS_ID", "default")
                    reply = bot.answer_query(transcript, business_id=business_id)
                    _send_meta_reply(phone_number_id, waid, reply)
                except Exception:
                    logging.exception("Failed to answer from transcript")

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