# Agentic Bot — Malaysia RAG WhatsApp Bot

This repository contains a small Python Flask app that demonstrates a multi-business RAG (retrieval-augmented generation) chatbot focused on Malaysia. It supports Hugging Face multilingual models (for Bahasa Melayu and English) and/or OpenAI, and uses Pinecone for vector storage. The bot receives WhatsApp messages via Twilio's webhook and routes to per-business knowledge bases via Pinecone namespaces.

Files
- `bot.py` — Flask app implementing a Twilio WhatsApp webhook. Queries Pinecone by business namespace, calls OpenAI or Hugging Face (configurable), and replies via TwiML. Exposes ingestion/config endpoints. Auto-detects Malay vs English and responds in the same language. Enforces strict RAG guardrails (answers only from retrieved documents; refuses when unsupported).
- `index_documents.py` — Script to index `.pdf`, `.txt`, `.md`, and `.json` files into Pinecone. Chunks documents and stores embeddings with metadata. Supports JSON Q/A lists and PDF ingestion via `pypdf`. Uses sentence-transformers multilingual embeddings by default.
- `ingest_default_pdfs.py` — Helper to purge and ingest the three included PDFs into the `default` namespace in one go.
- `.env.example` — Example environment variables (includes `RAG_MIN_SCORE`, `MAX_FILE_SIZE_MB`, `PDF_PAGE_MAX_CHARS`).
- `requirements.txt` — Python dependencies.
- `business_config.json` — Per-business instructions (system prompt). Auto-created if missing.
- `Dockerfile` — Container image for local/ECS/EB deployment with gunicorn.

Quick setup (PowerShell on Windows)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies (PowerShell):

```powershell
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in the values.

	Choose providers:
	- For Hugging Face (Malay/English): set `LLM_PROVIDER=hf`, `HUGGINGFACEHUB_API_TOKEN`, `HF_CHAT_MODEL_ID` (e.g. `meta-llama/Llama-3.1-8B-Instruct`).
	- For OpenAI: set `LLM_PROVIDER=openai` and `OPENAI_API_KEY`.
	- Embeddings: default to `EMBEDDING_PROVIDER=hf` with `intfloat/multilingual-e5-base` (good multilingual embedding including Malay). You can switch to OpenAI if preferred.

	Also map Twilio numbers to business IDs for multi-tenant routing via `TWILIO_NUMBER_TO_BUSINESS`.

```powershell
copy .env.example .env
# then edit .env in an editor
```

4. Index documents — you can ingest the provided three PDFs into the `default` namespace with purge:

```powershell
python ingest_default_pdfs.py
```

Or put your Malaysia-related docs (PDFs, price lists, descriptions, sample Q/A) in `./docs` and run (per business):

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

6. Test locally without Twilio using the simulate endpoint:

```powershell
curl -s -X POST http://localhost:5000/simulate?business_id=default -H "Content-Type: application/json" -d '{"message":"Berapa harga ikan kembung?"}' | jq
```

7. Expose your local server to the internet for Twilio webhook testing (e.g., using ngrok). Configure your Twilio WhatsApp sandbox webhook to `https://{your-tunnel}.ngrok.io/whatsapp` and enable inbound messages in the Twilio console. Use `TWILIO_NUMBER_TO_BUSINESS` to route each Twilio number to a business namespace.

Hugging Face notes
- Default embedding model: `intfloat/multilingual-e5-base` (dimension auto-detected). The Pinecone index is created with the correct dimension on first run.
- Example chat models: `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.2`, or any multilingual instruction model exposed on the HF Inference API.
- Set `LLM_PROVIDER=hf` and `HUGGINGFACEHUB_API_TOKEN` to use HF Inference API.

AWS deployment (overview)
1) Build the container locally:

```powershell
docker build -t agentic-bot:latest .
```

2) Push to Amazon ECR and run on ECS Fargate or Elastic Beanstalk (Docker platform):
- Create an ECR repo, login, tag and push the image.
- For ECS Fargate: create a task definition (CPU/RAM), set env vars (secrets in AWS Secrets Manager), service + ALB target group to port 8080.
- For Elastic Beanstalk: create a Docker app, upload the image or a Dockerrun JSON, set env vars in EB config.

3) Configure Twilio WhatsApp webhook to your public load balancer URL `/whatsapp`.

Minimum env vars on AWS
- Pinecone: `PINECONE_API_KEY`, `PINECONE_ENV`, `PINECONE_INDEX_NAME`
- LLM: either `OPENAI_API_KEY` or `HUGGINGFACEHUB_API_TOKEN` + `HF_CHAT_MODEL_ID`, and provider toggles
- Twilio: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_NUMBER_TO_BUSINESS`
- App: `PORT=8080`, `ADMIN_API_KEY`, `BUSINESS_CONFIG_PATH`

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

Strict RAG behavior
- The bot only answers from retrieved document snippets. If nothing relevant is found above a similarity threshold, it refuses and suggests asking about the covered topics (ornamental fish like arowana, stock, prices, specs) — both in Malay or English.
- Tune the similarity threshold via `RAG_MIN_SCORE` (default `0.75`). Higher = stricter; lower = more permissive.
- PDF indexing controls: `MAX_FILE_SIZE_MB` (default `2`), `PDF_PAGE_MAX_CHARS` to cap per-page characters; chunking via `CHUNK_SIZE`/`CHUNK_OVERLAP` in `index_documents.py`.

Notes & next steps
- This is a minimal reference implementation. For production, secure your keys (AWS Secrets Manager/Parameter Store), add request verification for Twilio, add rate limiting, caching, and robust error handling.
- Consider using retries and exponential backoff for network calls.
- For better relevance, tailor the document ingestion to include structured metadata (e.g., categories, official sources) and improve chunking using token-aware libraries.
- Add a persistence store (SQL/NoSQL) for multi-tenant configs and usage analytics. Add user auth for a business self-serve portal.
