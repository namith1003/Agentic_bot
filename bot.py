
import os
import json
from typing import List, Dict, Any
from flask import Flask, request, Response, jsonify
from twilio.twiml.messaging_response import MessagingResponse
import openai
import pinecone
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from threading import Lock

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "malaysia-rag")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
BUSINESS_CONFIG_PATH = os.getenv("BUSINESS_CONFIG_PATH", "business_config.json")
TWILIO_NUMBER_TO_BUSINESS = os.getenv("TWILIO_NUMBER_TO_BUSINESS", "{}")

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_ENV:
	raise RuntimeError("Please set OPENAI_API_KEY, PINECONE_API_KEY and PINECONE_ENV in your environment")

openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
# Create index if missing (dimension 1536 for text-embedding-3-small)
if PINECONE_INDEX_NAME not in pinecone.list_indexes():
	try:
		pinecone.create_index(PINECONE_INDEX_NAME, dimension=1536)
	except Exception:
		# In some environments, index creation may require console setup; continue if can't create
		pass

index = pinecone.Index(PINECONE_INDEX_NAME)

app = Flask(__name__)
_config_lock = Lock()


def _load_business_map() -> Dict[str, str]:
	try:
		return json.loads(TWILIO_NUMBER_TO_BUSINESS)
	except Exception:
		return {}


def load_business_config() -> Dict[str, Any]:
	if not os.path.exists(BUSINESS_CONFIG_PATH):
		# Create a default config
		default = {
			"default": {
				"instructions": (
					"You are a helpful Malaysia-focused sales assistant. Answer questions about products, prices, availability, and local considerations (MYR currency, Malaysian context)."
					" Be concise, friendly, and transparent. If information is missing, ask a clarifying question and suggest next steps.")
			}
		}
		with open(BUSINESS_CONFIG_PATH, "w", encoding="utf-8") as f:
			json.dump(default, f, ensure_ascii=False, indent=2)
		return default

	try:
		with open(BUSINESS_CONFIG_PATH, "r", encoding="utf-8") as f:
			return json.load(f)
	except Exception:
		return {"default": {"instructions": "You are a helpful Malaysia-focused sales assistant."}}


def save_business_config(cfg: Dict[str, Any]):
	with _config_lock:
		with open(BUSINESS_CONFIG_PATH, "w", encoding="utf-8") as f:
			json.dump(cfg, f, ensure_ascii=False, indent=2)


BUSINESS_CONFIG = load_business_config()


def get_embedding(text: str):
	# Uses OpenAI embeddings API
	resp = openai.Embedding.create(model="text-embedding-3-small", input=text)
	return resp["data"][0]["embedding"]


def query_pinecone(query: str, top_k: int = 4, namespace: str = "default"):
	emb = get_embedding(query)
	# Pinecone query in business namespace
	resp = index.query(vector=emb, top_k=top_k, include_metadata=True, namespace=namespace)
	results = []
	for match in resp.get("matches", []):
		metadata = match.get("metadata", {})
		score = match.get("score")
		text = metadata.get("text") or metadata.get("content") or ""
		source = metadata.get("source")
		results.append({"score": score, "text": text, "source": source, "metadata": metadata})
	return results


def build_prompt(user_question: str, contexts: list, instructions: str):
	# Business-specific system instruction and context
	system = instructions

	context_texts = []
	for i, c in enumerate(contexts, start=1):
		context_texts.append(f"[{i}] {c.get('text','')}")

	context_block = "\n\n".join(context_texts) if context_texts else ""

	messages = [
		{"role": "system", "content": system},
	]

	if context_block:
		messages.append({"role": "system", "content": f"Relevant documents from the Malaysia knowledge base:\n\n{context_block}"})

	messages.append({"role": "user", "content": user_question})
	return messages


def answer_query(question: str, business_id: str = "default"):
	# Retrieve contexts from Pinecone under business namespace
	contexts = query_pinecone(question, top_k=4, namespace=business_id)
	instructions = BUSINESS_CONFIG.get(business_id, {}).get("instructions") or \
		BUSINESS_CONFIG.get("default", {}).get("instructions", "You are a helpful Malaysia-focused sales assistant.")
	messages = build_prompt(question, contexts, instructions)

	# Query OpenAI ChatCompletion
	resp = openai.ChatCompletion.create(
		model=os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"),
		messages=messages,
		max_tokens=600,
		temperature=0.2,
	)
	answer = resp["choices"][0]["message"]["content"].strip()

	# Optionally append sources
	sources = []
	for c in contexts:
		if c.get("source"):
			sources.append(c["source"])

	if sources:
		answer += "\n\nSources: " + ", ".join(dict.fromkeys(sources))

	return answer


def require_admin_api_key() -> bool:
	if not ADMIN_API_KEY:
		return True  # if no admin key set, keep open for local testing
	api_key = request.headers.get("X-API-Key")
	return api_key == ADMIN_API_KEY


@app.route("/health", methods=["GET"])
def health():
	return jsonify({"status": "ok"})


@app.route("/config", methods=["POST"])
def set_config():
	if not require_admin_api_key():
		return jsonify({"error": "unauthorized"}), 401

	business_id = request.args.get("business_id", "default").strip() or "default"
	data = request.get_json(silent=True) or {}
	instructions = data.get("instructions")
	if not instructions:
		return jsonify({"error": "instructions required"}), 400

	BUSINESS_CONFIG[business_id] = BUSINESS_CONFIG.get(business_id, {})
	BUSINESS_CONFIG[business_id]["instructions"] = instructions
	save_business_config(BUSINESS_CONFIG)
	return jsonify({"ok": True, "business_id": business_id})


def _embed_and_upsert(docs: List[Dict[str, Any]], namespace: str):
	vectors = []
	for i, d in enumerate(docs):
		content = d.get("content") or d.get("text") or ""
		if not content:
			continue
		emb = get_embedding(content)
		doc_id = d.get("id") or f"doc-{i}"
		metadata = d.get("metadata") or {}
		metadata.setdefault("text", content)
		if d.get("source"):
			metadata.setdefault("source", d["source"])
		vectors.append((doc_id, emb, metadata))

	if vectors:
		index.upsert(vectors=vectors, namespace=namespace)


@app.route("/ingest", methods=["POST"])
def ingest():
	if not require_admin_api_key():
		return jsonify({"error": "unauthorized"}), 401

	business_id = request.args.get("business_id", "default").strip() or "default"
	payload = request.get_json(silent=True)
	if not payload or "documents" not in payload:
		return jsonify({"error": "expected JSON with 'documents'"}), 400
	docs = payload.get("documents", [])
	if not isinstance(docs, list) or not docs:
		return jsonify({"error": "documents must be a non-empty list"}), 400

	try:
		_embed_and_upsert(docs, namespace=business_id)
		return jsonify({"ok": True, "ingested": len(docs), "business_id": business_id})
	except Exception as e:
		app.logger.exception("ingest failed")
		return jsonify({"error": str(e)}), 500


@app.route("/simulate", methods=["POST"])
def simulate():
	data = request.get_json(silent=True) or {}
	business_id = request.args.get("business_id", data.get("business_id", "default"))
	message = data.get("message")
	if not message:
		return jsonify({"error": "message is required"}), 400
	try:
		reply = answer_query(message, business_id=business_id)
		return jsonify({"reply": reply, "business_id": business_id})
	except Exception as e:
		app.logger.exception("simulate failed")
		return jsonify({"error": str(e)}), 500


@app.route("/whatsapp", methods=["POST", "GET"])
def whatsapp_webhook():
	# Twilio will POST 'Body' and 'From'
	body = request.values.get("Body", "").strip()
	from_number = request.values.get("From", "")
	to_number = request.values.get("To", "")

	# Optional prefix to select business for testing: "biz:myshop How much is..."
	business_id = "default"
	if body.lower().startswith("biz:"):
		try:
			prefix, rest = body.split(None, 1)
			business_id = prefix[4:]
			body = rest
		except Exception:
			pass
	else:
		# Map by Twilio destination number
		biz_map = _load_business_map()
		business_id = biz_map.get(to_number, "default")

	if not body:
		resp = MessagingResponse()
		resp.message("Hi — I didn't receive any message. Please send a question about Malaysia and I'll try to help.")
		return Response(str(resp), mimetype="application/xml")

	try:
		reply = answer_query(body, business_id=business_id)
	except Exception as e:
		reply = "Sorry, I had trouble processing your request. Please try again later."
		app.logger.exception("Error while answering query")

	tw_resp = MessagingResponse()
	tw_resp.message(reply)
	return Response(str(tw_resp), mimetype="application/xml")


if __name__ == "__main__":
	port = int(os.getenv("PORT", 5000))
	app.run(host="0.0.0.0", port=port, debug=True)
