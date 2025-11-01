
import os
import json
from typing import List, Dict, Any
from flask import Flask, request, Response, jsonify
try:
	from twilio.twiml.messaging_response import MessagingResponse
except Exception:
	MessagingResponse = None  # Twilio optional; not required for Meta WhatsApp
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import time
import socket
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from threading import Lock
from langdetect import detect as detect_lang
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from typing import Dict, Any

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
# Hardcode Pinecone index details per user request
PINECONE_INDEX_NAME = "malaysia-rag2"
PINECONE_HOST = os.getenv("PINECONE_HOST", "https://malaysia-rag2-4ipjuy3.svc.aped-4627-b74a.pinecone.io")
PINECONE_CLOUD = "aws"  # serverless cloud
PINECONE_REGION = "us-east-1"  # serverless region
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
BUSINESS_CONFIG_PATH = os.getenv("BUSINESS_CONFIG_PATH", "business_config.json")
TWILIO_NUMBER_TO_BUSINESS = os.getenv("TWILIO_NUMBER_TO_BUSINESS", "{}")

# Model/provider toggles
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "hf").lower()  # default HF for Malay/English
HF_MODEL_ID = os.getenv("HF_CHAT_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/distiluse-base-multilingual-cased-v1")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()  # default to OpenAI per user request

if not PINECONE_API_KEY or not PINECONE_ENV:
	raise RuntimeError("Please set PINECONE_API_KEY and PINECONE_ENV in your environment")

if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
	raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY not set")
if LLM_PROVIDER == "hf" and not HF_API_TOKEN:
	raise RuntimeError("LLM_PROVIDER=hf but HUGGINGFACEHUB_API_TOKEN not set")

oa_client = None
if OPENAI_API_KEY:
	# OpenAI Python SDK v1
	oa_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize embeddings (HF or OpenAI)
_hf_embedder = None
_embedding_dim = 512  # will adjust based on provider

def _init_embedder():
	global _hf_embedder, _embedding_dim
	if EMBEDDING_PROVIDER == "hf":
		_hf_embedder = SentenceTransformer(EMBEDDING_MODEL_ID)
		_embedding_dim = int(_hf_embedder.get_sentence_embedding_dimension())
	else:
		# Using OpenAI text-embedding-3-small (1536 dims)
		_embedding_dim = 1536

_init_embedder()

def _parse_pinecone_env(env: str):
	# Expect formats like "us-west1-gcp" -> (gcp, us-west1)
	try:
		if env and "-" in env:
			parts = env.split("-")
			cloud = parts[-1]
			region = "-".join(parts[:-1])
			return cloud, region
	except Exception:
		pass
	return None, None

def _ensure_pinecone_index(name: str, dim: int):
	pc_local = Pinecone(api_key=PINECONE_API_KEY)
	_cloud = PINECONE_CLOUD
	_region = PINECONE_REGION
	if not (_cloud and _region):
		parsed_cloud, parsed_region = _parse_pinecone_env(PINECONE_ENV)
		_cloud = _cloud or parsed_cloud or "aws"
		_region = _region or parsed_region or "us-east-1"

	print(f"Pinecone: ensuring index '{name}' (dim={dim}, cloud={_cloud}, region={_region})")
	try:
		names = [i.name for i in pc_local.list_indexes()]
		if name not in names:
			print("Pinecone: index not found, creating...")
			tried = []
			def _try_create(cloud, region):
				tried.append(f"{cloud}:{region}")
				pc_local.create_index(
					name=name,
					dimension=dim,
					metric="cosine",
					spec=ServerlessSpec(cloud=cloud, region=region),
				)

			try:
				_try_create(_cloud, _region)
			except Exception as ce:
				msg = str(ce)
				print(f"Pinecone: create failed for {_cloud}:{_region} -> {msg}")
				# Try common fallbacks
				fallbacks = [("aws","us-east-1"),("aws","us-west-2"),("gcp","us-central1")]
				created = False
				for (fc, fr) in fallbacks:
					if f"{fc}:{fr}" in tried:
						continue
					try:
						print(f"Pinecone: trying fallback region {fc}:{fr}...")
						_try_create(fc, fr)
						_cloud, _region = fc, fr
						created = True
						break
					except Exception as fe:
						print(f"Pinecone: fallback {fc}:{fr} failed -> {fe}")
				if not created:
					raise

			# wait until ready
			for _ in range(60):
				try:
					desc = pc_local.describe_index(name)
					status = getattr(desc, "status", None) or {}
					ready = bool(getattr(status, "ready", False) or status.get("ready") if isinstance(status, dict) else False)
				except Exception:
					ready = False
				if ready:
					break
				time.sleep(1)
			print(f"Pinecone: index is ready in cloud={_cloud}, region={_region}.")
	except Exception as e:
		print("Pinecone: failed to ensure index. Error:", e)
		raise RuntimeError(
			"Couldn't ensure Pinecone index. Verify API key, region/cloud match your Pinecone project, or create the index manually "
			f"in console with name='{name}', dimension={dim}, metric=cosine, serverless cloud={_cloud}, region={_region}.")

	return pc_local.Index(name)

# Initialize Pinecone index handle
pc = Pinecone(api_key=PINECONE_API_KEY)
try:
	# Connect directly to the existing index via provided host
	index = pc.Index(PINECONE_INDEX_NAME, host=PINECONE_HOST)
	print(f"Pinecone: connected to existing index '{PINECONE_INDEX_NAME}' at host {PINECONE_HOST}")
except Exception:
	# Fallback: attempt ensuring index (will try serverless create if missing)
	index = _ensure_pinecone_index(PINECONE_INDEX_NAME, _embedding_dim)

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
	text = (text or "").strip()
	if not text:
		return [0.0] * _embedding_dim
	if EMBEDDING_PROVIDER == "hf":
		return _hf_embedder.encode(text, normalize_embeddings=True).tolist()
	# OpenAI (SDK v1)
	if not oa_client:
		raise RuntimeError("OPENAI_API_KEY not set but EMBEDDING_PROVIDER=openai")
	resp = oa_client.embeddings.create(
		model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
		input=text,
	)
	return resp.data[0].embedding


def query_pinecone(query: str, top_k: int = 4, namespace: str = "default"):
	emb = get_embedding(query)
	# Pinecone query in business namespace
	resp = index.query(vector=emb, top_k=top_k, include_metadata=True, namespace=namespace)
	matches = []
	# Be resilient if SDK returns object-like structure
	try:
		matches = resp.get("matches", [])
	except Exception:
		try:
			matches = getattr(resp, "matches", [])
		except Exception:
			matches = []

	results = []
	for match in matches:
		metadata = match.get("metadata", {})
		score = match.get("score")
		text = metadata.get("text") or metadata.get("content") or ""
		source = metadata.get("source")
		results.append({"score": score, "text": text, "source": source, "metadata": metadata})
	return results


def build_prompt(user_question: str, contexts: list, instructions: str, reply_lang_hint: str = "auto", tone: str = "formal", short_answer: bool = False):
	# Business-specific system instruction and context
	system = (
		instructions
		+ " Always and only use the provided 'Relevant documents' as your source of truth."
		+ " If the answer is not explicitly supported by those documents, say you don't know and suggest checking the stock list or asking a more specific question."
		+ " Never invent items, prices, or policies. The business sells pet fish (e.g., Arowana and other aquarium fish) and does not sell food."
		+ " Do not include citations, file names, or sources in your reply—answer naturally as a sales assistant."
	)

	context_texts = []
	for i, c in enumerate(contexts, start=1):
		context_texts.append(f"[{i}] {c.get('text','')}")

	context_block = "\n\n".join(context_texts) if context_texts else ""

	# Language hint and regional style
	if reply_lang_hint == "ms":
		system += " Respond strictly in Bahasa Malaysia (Bahasa Melayu Malaysia), not Indonesian Malay. Use local wording and RM (MYR) currency style."
	elif reply_lang_hint == "en":
		system += " Respond in English."

	# Tone/style adaptation and human touch
	if tone == "informal":
		system += (
			" Match a friendly Malaysian tone. It's okay to use light, respectful colloquial terms like 'boss' when appropriate, "
			"but stay helpful and sales-focused. Keep answers concise and natural. Never say or imply you are a bot."
		)
	else:
		system += (
			" Keep a professional, courteous tone (no slang). Never say or imply you are a bot."
		)

	# Avoid meta-offers in any language
	system += " Avoid meta-offers like 'let me know' or 'I can provide more information' in any language."

	# Brevity guidance for short questions
	if short_answer:
		system += (
			" For brief questions, respond in 1-2 short sentences. Be direct and avoid filler like 'let me know if you need more info'."
		)

	messages = [
		{"role": "system", "content": system},
	]

	if context_block:
		messages.append({"role": "system", "content": f"Relevant documents:\n\n{context_block}"})
	else:
		messages.append({
			"role": "system",
			"content": (
				"No relevant document snippets were found. You must respond that the information isn't available in our documents and avoid speculation."
			)
		})

	messages.append({"role": "user", "content": user_question})
	return messages


def _detect_lang_code(text: str) -> str:
	try:
		code = detect_lang(text)
		# Map common aliases
		if code.startswith("ms"):
			return "ms"
		if code.startswith("en"):
			return "en"
		return code
	except Exception:
		return "auto"


def _hf_generate(messages: List[Dict[str, str]], max_new_tokens: int = 600) -> str:
	# Simple prompt composer for instruction-tuned models
	# Convert chat to a single prompt
	system_parts = [m["content"] for m in messages if m["role"] == "system"]
	user_parts = [m["content"] for m in messages if m["role"] == "user"]
	prompt = "\n\n".join([*(f"[SYSTEM]\n{s}" for s in system_parts), *(f"[USER]\n{u}" for u in user_parts)]) + "\n\n[ASSISTANT]\n"
	client = InferenceClient(model=HF_MODEL_ID, token=HF_API_TOKEN)
	# temperature low for factual answers
	result = client.text_generation(prompt=prompt, max_new_tokens=max_new_tokens, temperature=0.2, do_sample=False)
	return result.strip()

def _infer_formality(text: str) -> str:
	"""Heuristic: return 'formal' or 'informal' based on common Malaysian cues."""
	t = (text or "").lower()
	informal_cues = [" boss", "bos", "bro", "sis", "bang", "lah", "lol", "haha", "😀", "😂", " bro ", "boss"]
	formal_cues = ["encik", "puan", "tuan", "yang berhormat", "dear", "regards", "salam sejahtera"]
	if any(cue in t for cue in informal_cues) and not any(cue in t for cue in formal_cues):
		return "informal"
	if any(cue in t for cue in formal_cues):
		return "formal"
	# default slightly formal to avoid over-casual tone
	return "formal"


def _refusal_message(lang: str, tone: str) -> str:
	if lang == "ms":
		if tone == "informal":
			return (
				"Maaf bos, saya tak pasti tentang perkara tu. Boleh jelaskan sikit lagi atau tanya tentang stok, harga, atau spesifikasi?"
			)
		return (
			"Maaf, saya tidak pasti tentang perkara itu. Boleh jelaskan sedikit lagi atau tanya tentang stok, harga, atau spesifikasi?"
		)
	# English
	if tone == "informal":
		return (
			"Sorry boss, I’m not sure about that. Could you clarify a bit, or ask about stock, prices, or specs?"
		)
	return (
		"Apologies, I’m not sure about that. Could you clarify, or ask about stock, prices, or specifications?"
	)


def answer_query(question: str, business_id: str = "default"):
	# Retrieve contexts from Pinecone under business namespace (default for this business)
	top_k = int(os.getenv("RAG_TOP_K", "8"))
	results = query_pinecone(question, top_k=top_k, namespace=business_id)
	# Use Pinecone strictly as the RAG source of truth: if no matches at all, refuse.
	# Otherwise, use whatever Pinecone returned (no extra threshold gating).
	contexts = results or []
	instructions = BUSINESS_CONFIG.get(business_id, {}).get("instructions") or \
		BUSINESS_CONFIG.get("default", {}).get("instructions", "You are a helpful Malaysia-focused sales assistant.")
	# detect language (Malay vs English) and hint model to reply in the same language
	lang_code = _detect_lang_code(question)
	reply_hint = "ms" if lang_code == "ms" else ("en" if lang_code == "en" else "auto")
	tone = _infer_formality(question)
	# Simple heuristic: keep very short questions concise
	q = (question or "").strip()
	short_answer = (len(q) <= 60) or (len(q.split()) <= 10)

	# If Pinecone returned no matches, refuse gracefully in the user's language
	if not contexts:
		return _refusal_message(reply_hint if reply_hint in ("ms", "en") else "en", tone)

	messages = build_prompt(question, contexts, instructions, reply_lang_hint=reply_hint, tone=tone, short_answer=short_answer)

	if LLM_PROVIDER == "hf":
		answer = _hf_generate(messages, max_new_tokens=(120 if short_answer else 400))
	else:
		# OpenAI Chat Completions (SDK v1)
		if not oa_client:
			raise RuntimeError("OPENAI_API_KEY not set but LLM_PROVIDER=openai")
		resp = oa_client.chat.completions.create(
			model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
			messages=messages,
			max_tokens=(160 if short_answer else 600),
			temperature=0.2,
		)
		answer = (resp.choices[0].message.content or "").strip()

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
		payload = [
			{"id": vid, "values": vec, "metadata": meta}
			for (vid, vec, meta) in vectors
		]
		index.upsert(vectors=payload, namespace=namespace)


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

	if MessagingResponse is not None:
		tw_resp = MessagingResponse()
		tw_resp.message(reply)
		return Response(str(tw_resp), mimetype="application/xml")
	# Fallback plain text (for environments without Twilio)
	return Response(reply, mimetype="text/plain")


if __name__ == "__main__":
	def _namespace_stats() -> Dict[str, Any]:
		try:
			stats = index.describe_index_stats()  # should include namespaces map
			# Normalize to dict
			try:
				ns = stats.get("namespaces", {})
			except Exception:
				ns = getattr(stats, "namespaces", {}) if hasattr(stats, "namespaces") else {}
			return {"namespaces": ns}
		except Exception as e:
			return {"error": str(e)}

	@app.get("/debug/stats")
	def debug_stats():
		business_id = request.args.get("business_id", "default")
		stats = _namespace_stats()
		# extract count if present
		ns_counts = {}
		try:
			ns_counts = stats.get("namespaces", {}) or {}
		except Exception:
			pass
		return jsonify({
			"index": PINECONE_INDEX_NAME,
			"host": PINECONE_HOST,
			"namespace_counts": ns_counts,
			"requested_namespace": business_id
		})
	def _get_lan_ip() -> str:
		try:
			s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
			s.connect(("8.8.8.8", 80))
			ip = s.getsockname()[0]
			s.close()
			return ip
		except Exception:
			return "127.0.0.1"

	port = int(os.getenv("PORT", 5000))
	local_url = f"http://localhost:{port}"
	lan_url = f"http://{_get_lan_ip()}:{port}"

	print("================ Server starting ================")
	print(f"Local URL:    {local_url}")
	print(f"LAN URL:      {lan_url}")
	print(f"Health Check: {local_url}/health")
	print(f"Simulate API: POST {local_url}/simulate?business_id=default  (body: {{\"message\": \"Hello\"}})")
	print(f"WhatsApp Webhook: POST {local_url}/whatsapp")
	# Log Pinecone basics and namespace stats for quick verification
	try:
		ns_stats = index.describe_index_stats()
		try:
			ns_map = ns_stats.get("namespaces", {})
		except Exception:
			ns_map = getattr(ns_stats, "namespaces", {}) if hasattr(ns_stats, "namespaces") else {}
		print(f"Pinecone: index='{PINECONE_INDEX_NAME}', host='{PINECONE_HOST}', namespaces={ns_map}")
	except Exception as e:
		print(f"Pinecone: failed to fetch stats: {e}")
	print("=================================================")

	app.run(host="0.0.0.0", port=port, debug=True)
