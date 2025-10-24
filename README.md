# Agentic Bot — Malaysia RAG WhatsApp Bot

This repository contains a small Python Flask app that demonstrates a multi-business RAG (retrieval-augmented generation) chatbot focused on Malaysia. It uses OpenAI for embeddings and chat completions and Pinecone for vector storage. The bot is wired to receive WhatsApp messages via Twilio's webhook and supports per-business knowledge bases via Pinecone namespaces.

Files
- `bot.py` — Flask app implementing a Twilio WhatsApp webhook. Queries Pinecone by business namespace, calls OpenAI, and replies via TwiML. Also exposes ingestion and configuration endpoints.
- `index_documents.py` — Script to index `.txt`, `.md`, and `.json` files into Pinecone. Chunks documents and stores embeddings with metadata. Supports JSON Q/A lists.
- `.env.example` — Example environment variables.
- `requirements.txt` — Python dependencies.
- `business_config.json` — Per-business instructions (system prompt). Auto-created if missing.

Quick setup (PowerShell on Windows)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in the values (OpenAI & Pinecone keys, Twilio credentials if using WhatsApp). Map Twilio numbers to business IDs for multi-tenant routing:

```powershell
copy .env.example .env
# then edit .env in an editor
```

4. Index documents (optional) — put Malaysia-related docs in `./docs` and run (per business):

```powershell
# default business
python index_documents.py .\docs --business-id default

# another business namespace (e.g., myshop)
python index_documents.py .\docs --business-id myshop
```

5. Run the Flask app locally:

```powershell
python bot.py
```

6. Expose your local server to the internet for Twilio webhook testing (e.g., using ngrok). Configure your Twilio WhatsApp sandbox webhook to `https://{your-tunnel}.ngrok.io/whatsapp` and enable inbound messages in the Twilio console. Use `TWILIO_NUMBER_TO_BUSINESS` to route each Twilio number to a business namespace.

API endpoints (local admin)

- POST `/ingest?business_id={id}` — Ingest documents as JSON (requires `X-API-Key` if `ADMIN_API_KEY` set).

	Body:

	```json
	{
		"documents": [
			{ "content": "Product: Nasi Lemak, Price: 8 MYR", "source": "price-list-2025.md", "metadata": {"type":"price_list"} }
		]
	}
	```

- POST `/config?business_id={id}` — Set business-specific instructions/system prompt.

	Body:

	```json
	{ "instructions": "You are a sales assistant for MyShop in Kuala Lumpur..." }
	```

- POST `/simulate?business_id={id}` — Chat locally without Twilio.

	Body:

	```json
	{ "message": "Do you have chicken rice and how much?" }
	```

WhatsApp routing and local testing

- The webhook `/whatsapp` maps incoming messages to a business ID by the Twilio destination number (`To`) using the `TWILIO_NUMBER_TO_BUSINESS` map. For quick local tests, you can prefix the message with `biz:{id} `. Example: `biz:myshop What's today's promo?`.

Notes & next steps
- This is a minimal reference implementation. For production, secure your keys, add request verification for Twilio, add rate limiting, caching, and robust error handling.
- Consider using retries and exponential backoff for network calls.
- For better relevance, tailor the document ingestion to include structured metadata (e.g., categories, official sources) and improve chunking using token-aware libraries.
- Add a persistence store (SQL/NoSQL) for multi-tenant configs and usage analytics. Add user auth for a business self-serve portal.
