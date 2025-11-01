import os
import json
from typing import Any, Dict
import requests
import azure.functions as func

# Import core logic from existing module, but do NOT hard-fail if env vars are missing.
# If the import fails (e.g., missing Pinecone/OpenAI/HF keys), we still want the host
# to load and expose at least /api/health so the Function shows up in Portal.
BOT_READY = False
BOT_IMPORT_ERROR = None
try:
    from bot import (
        answer_query,
        _embed_and_upsert,
        BUSINESS_CONFIG,
        save_business_config,
        ADMIN_API_KEY,
        PINECONE_INDEX_NAME,
        PINECONE_HOST,
        index,
    )
    BOT_READY = True
except Exception as e:
    # Defer failure to request time; surface helpful error via endpoints.
    BUSINESS_CONFIG = {}
    ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "")
    PINECONE_HOST = os.getenv("PINECONE_HOST", "")
    index = None
    BOT_IMPORT_ERROR = str(e)


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.route(route="health", methods=["GET"])  # GET /api/health
def health(req: func.HttpRequest) -> func.HttpResponse:
    body = {"status": "ok"}
    if not BOT_READY and BOT_IMPORT_ERROR:
        body["bot_ready"] = False
        body["error"] = BOT_IMPORT_ERROR
    else:
        body["bot_ready"] = True
    return func.HttpResponse(json.dumps(body), mimetype="application/json", status_code=200)


@app.route(route="simulate", methods=["POST"])  # POST /api/simulate
def simulate(req: func.HttpRequest) -> func.HttpResponse:
    if not BOT_READY:
        return func.HttpResponse(
            json.dumps({"error": "bot not initialized", "detail": BOT_IMPORT_ERROR}),
            mimetype="application/json",
            status_code=503,
        )
    try:
        data = req.get_json()
    except ValueError:
        data = {}
    business_id = req.params.get("business_id") or data.get("business_id", "default")
    message = (data.get("message") or "").strip()
    if not message:
        return func.HttpResponse(
            json.dumps({"error": "message is required"}),
            mimetype="application/json",
            status_code=400,
        )
    try:
        reply = answer_query(message, business_id=business_id)
        return func.HttpResponse(
            json.dumps({"reply": reply, "business_id": business_id}),
            mimetype="application/json",
            status_code=200,
        )
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500,
        )


@app.route(route="whatsapp", methods=["GET", "POST"])  # Meta WhatsApp Cloud API webhook -> /api/whatsapp
def whatsapp(req: func.HttpRequest) -> func.HttpResponse:
    # Verification (GET)
    if req.method == "GET":
        mode = req.params.get("hub.mode")
        token = req.params.get("hub.verify_token")
        challenge = req.params.get("hub.challenge")
        verify_token = os.getenv("META_VERIFY_TOKEN") or os.getenv("WHATSAPP_VERIFY_TOKEN")
        if mode == "subscribe" and token and verify_token and token == verify_token:
            return func.HttpResponse(challenge or "", status_code=200, mimetype="text/plain")
        return func.HttpResponse("forbidden", status_code=403)

    # Events (POST)
    if not BOT_READY:
        # Acknowledge to avoid webhook retries but inform missing setup
        return func.HttpResponse("EVENT_RECEIVED", status_code=200)
    try:
        payload = req.get_json()
    except ValueError:
        payload = {}

    # Helper to send a WhatsApp text via Graph API
    def send_whatsapp_text(to_number: str, text: str, phone_number_id: str | None = None) -> Dict[str, Any]:
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

    try:
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

                    # Business selection via prefix (optional)
                    business_id = "default"
                    low = text_body.lower()
                    if low.startswith("biz:"):
                        try:
                            prefix, rest = text_body.split(None, 1)
                            business_id = prefix[4:]
                            text_body = rest
                        except Exception:
                            pass

                    try:
                        reply = answer_query(text_body, business_id=business_id)
                    except Exception:
                        reply = "Maaf, sistem mengalami masalah sebentar. Cuba lagi nanti."

                    # Send reply back to user via Graph API
                    send_whatsapp_text(from_number, reply, phone_number_id=phone_number_id)

        return func.HttpResponse("EVENT_RECEIVED", status_code=200)
    except Exception:
        return func.HttpResponse("EVENT_RECEIVED", status_code=200)


def _is_admin(req: func.HttpRequest) -> bool:
    if not ADMIN_API_KEY:
        return True
    return (req.headers.get("x-api-key") or req.headers.get("X-API-Key")) == ADMIN_API_KEY


@app.route(route="config", methods=["POST"])  # POST /api/config?business_id=default
def set_config(req: func.HttpRequest) -> func.HttpResponse:
    if not BOT_READY:
        return func.HttpResponse(
            json.dumps({"error": "bot not initialized", "detail": BOT_IMPORT_ERROR}),
            mimetype="application/json",
            status_code=503,
        )
    if not _is_admin(req):
        return func.HttpResponse(
            json.dumps({"error": "unauthorized"}), mimetype="application/json", status_code=401
        )
    try:
        data = req.get_json()
    except ValueError:
        data = {}
    business_id = (req.params.get("business_id") or "default").strip() or "default"
    instructions = data.get("instructions")
    if not instructions:
        return func.HttpResponse(
            json.dumps({"error": "instructions required"}), mimetype="application/json", status_code=400
        )
    BUSINESS_CONFIG[business_id] = BUSINESS_CONFIG.get(business_id, {})
    BUSINESS_CONFIG[business_id]["instructions"] = instructions
    save_business_config(BUSINESS_CONFIG)
    return func.HttpResponse(
        json.dumps({"ok": True, "business_id": business_id}), mimetype="application/json", status_code=200
    )


@app.route(route="ingest", methods=["POST"])  # POST /api/ingest?business_id=default
def ingest(req: func.HttpRequest) -> func.HttpResponse:
    if not BOT_READY:
        return func.HttpResponse(
            json.dumps({"error": "bot not initialized", "detail": BOT_IMPORT_ERROR}),
            mimetype="application/json",
            status_code=503,
        )
    if not _is_admin(req):
        return func.HttpResponse(
            json.dumps({"error": "unauthorized"}), mimetype="application/json", status_code=401
        )
    business_id = (req.params.get("business_id") or "default").strip() or "default"
    try:
        payload = req.get_json()
    except ValueError:
        payload = None
    if not payload or "documents" not in payload:
        return func.HttpResponse(
            json.dumps({"error": "expected JSON with 'documents'"}), mimetype="application/json", status_code=400
        )
    docs = payload.get("documents", [])
    if not isinstance(docs, list) or not docs:
        return func.HttpResponse(
            json.dumps({"error": "documents must be a non-empty list"}), mimetype="application/json", status_code=400
        )
    try:
        _embed_and_upsert(docs, namespace=business_id)
        return func.HttpResponse(
            json.dumps({"ok": True, "ingested": len(docs), "business_id": business_id}),
            mimetype="application/json",
            status_code=200,
        )
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}), mimetype="application/json", status_code=500
        )


@app.route(route="debug/stats", methods=["GET"])  # GET /api/debug/stats
def debug_stats(req: func.HttpRequest) -> func.HttpResponse:
    if not BOT_READY or index is None:
        data = {"bot_ready": BOT_READY, "error": BOT_IMPORT_ERROR}
    else:
        try:
            stats = index.describe_index_stats()
            try:
                ns = stats.get("namespaces", {})
            except Exception:
                ns = getattr(stats, "namespaces", {}) if hasattr(stats, "namespaces") else {}
            data = {
                "index": PINECONE_INDEX_NAME,
                "host": PINECONE_HOST,
                "namespace_counts": ns,
            }
        except Exception as e:
            data = {"error": str(e)}
    return func.HttpResponse(json.dumps(data), mimetype="application/json", status_code=200)
