import os
import glob
import json
import re
from typing import List, Dict, Any, Iterable
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
# Hardcoded Pinecone index details per user request
PINECONE_INDEX_NAME = "malaysia-rag2"
PINECONE_HOST = os.getenv("PINECONE_HOST", "https://malaysia-rag2-4ipjuy3.svc.aped-4627-b74a.pinecone.io")
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/distiluse-base-multilingual-cased-v1")
# Default to OpenAI per user request; we'll fit vectors to 512 dims to match the index
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "2"))  # allow up to 2MB
PDF_PAGE_MAX_CHARS = int(os.getenv("PDF_PAGE_MAX_CHARS", "50000"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

if not PINECONE_API_KEY or not PINECONE_ENV:
    raise RuntimeError("Please set PINECONE_API_KEY and PINECONE_ENV in your environment")

oa_client = None
if OPENAI_API_KEY:
    oa_client = OpenAI(api_key=OPENAI_API_KEY)

def _parse_pinecone_env(env: str):
    try:
        if env and "-" in env:
            parts = env.split("-")
            cloud = parts[-1]
            region = "-".join(parts[:-1])
            return cloud, region
    except Exception:
        pass
    return None, None

pc = Pinecone(api_key=PINECONE_API_KEY)


def _ensure_pinecone_index(name: str, dim: int):
    pc_local = Pinecone(api_key=PINECONE_API_KEY)
    cloud = PINECONE_CLOUD
    region = PINECONE_REGION
    if not (cloud and region):
        parsed_cloud, parsed_region = _parse_pinecone_env(PINECONE_ENV)
        cloud = cloud or parsed_cloud or "aws"
        region = region or parsed_region or "us-east-1"

    print(f"Pinecone: ensuring index '{name}' (dim={dim}, cloud={cloud}, region={region})")
    try:
        names = [i.name for i in pc_local.list_indexes()]
        if name not in names:
            print("Pinecone: index not found, creating...")
            tried = []

            def _try_create(c, r):
                tried.append(f"{c}:{r}")
                pc_local.create_index(
                    name=name,
                    dimension=dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=c, region=r),
                )

            try:
                _try_create(cloud, region)
            except Exception as ce:
                print(f"Pinecone: create failed for {cloud}:{region} -> {ce}")
                fallbacks = [("aws", "us-east-1"), ("aws", "us-west-2"), ("gcp", "us-central1")]
                created = False
                for (fc, fr) in fallbacks:
                    if f"{fc}:{fr}" in tried:
                        continue
                    try:
                        print(f"Pinecone: trying fallback region {fc}:{fr}...")
                        _try_create(fc, fr)
                        cloud, region = fc, fr
                        created = True
                        break
                    except Exception as fe:
                        print(f"Pinecone: fallback {fc}:{fr} failed -> {fe}")
                if not created:
                    raise

            # wait until ready
            import time as _time
            for _ in range(60):
                try:
                    desc = pc_local.describe_index(name)
                    status = getattr(desc, "status", None) or {}
                    ready = bool(getattr(status, "ready", False) or status.get("ready") if isinstance(status, dict) else False)
                except Exception:
                    ready = False
                if ready:
                    break
                _time.sleep(1)
            print(f"Pinecone: index is ready in cloud={cloud}, region={region}.")
    except Exception as e:
        print("Pinecone: failed to ensure index. Error:", e)
        raise RuntimeError(
            "Couldn't ensure Pinecone index. Verify API key, region/cloud match your Pinecone project, or create the index manually "
            f"in console with name='{name}', dimension={dim}, metric=cosine, serverless cloud={cloud}, region={region}.")

    return pc_local.Index(name)


try:
    # Connect directly to the existing index via provided host
    index = pc.Index(PINECONE_INDEX_NAME, host=PINECONE_HOST)
    print(f"Pinecone: connected to existing index '{PINECONE_INDEX_NAME}' at host {PINECONE_HOST}")
except Exception:
    # Fallback: ensure or create the index if missing
    # Note: embedding dimension is initialized below, we'll temporarily pass 512 and adjust once embedder loads
    index = None  # placeholder; will be set after embedder init

_hf_embedder = None
_embedding_dim = 1536
if EMBEDDING_PROVIDER == "hf":
    _hf_embedder = SentenceTransformer(EMBEDDING_MODEL_ID)
    _embedding_dim = int(_hf_embedder.get_sentence_embedding_dimension())

# If index wasn't connected earlier, ensure/create it now with the correct embedding dim
if 'index' not in globals() or index is None:
    index = _ensure_pinecone_index(PINECONE_INDEX_NAME, _embedding_dim)
    print(f"Pinecone: ensured index '{PINECONE_INDEX_NAME}' (dim={_embedding_dim})")


def read_files_from_dir(path: str, patterns: List[str] = ["**/*.pdf"]):
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(path, p), recursive=True))
    return files


def read_files_from_list(paths: Iterable[str]) -> List[str]:
    files: List[str] = []
    for p in paths:
        p = p.strip().strip('"')
        if not p:
            continue
        if os.path.isdir(p):
            files.extend(read_files_from_dir(p, patterns=["**/*.pdf"]))
        elif os.path.isfile(p):
            files.append(p)
    return files


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Chunk text with safe stepping to avoid infinite loops and huge lists.
    Ensures at least step=1 and clamps overlap.
    """
    if not isinstance(text, str):
        text = str(text or "")
    length = len(text)
    if length == 0:
        return []
    if chunk_size is None or chunk_size <= 0:
        return [text]
    ov = max(0, overlap or 0)
    step = max(1, chunk_size - ov)
    out = []
    start = 0
    while start < length:
        end = min(start + chunk_size, length)
        out.append(text[start:end])
        if end >= length:
            break
        start += step
    return out


def get_embedding(text: str):
    text = (text or "").strip()
    if not text:
        return [0.0] * _embedding_dim
    if EMBEDDING_PROVIDER == "hf":
        return _hf_embedder.encode(text, normalize_embeddings=True).tolist()
    if not oa_client:
        raise RuntimeError("OPENAI_API_KEY not set but EMBEDDING_PROVIDER=openai")
    resp = oa_client.embeddings.create(model="text-embedding-3-small", input=text)
    vec = resp.data[0].embedding
    # Fit to index dimension (slice or zero-pad) to avoid dimension mismatch
    if len(vec) > _embedding_dim:
        vec = vec[:_embedding_dim]
    elif len(vec) < _embedding_dim:
        vec = vec + [0.0] * (_embedding_dim - len(vec))
    return vec


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    texts = [t.strip() if isinstance(t, str) else "" for t in texts]
    if EMBEDDING_PROVIDER == "hf":
        return _hf_embedder.encode(texts, batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True).tolist()
    if not oa_client:
        raise RuntimeError("OPENAI_API_KEY not set but EMBEDDING_PROVIDER=openai")
    resp = oa_client.embeddings.create(model="text-embedding-3-small", input=texts)
    out = []
    for d in resp.data:
        vec = d.embedding
        if len(vec) > _embedding_dim:
            vec = vec[:_embedding_dim]
        elif len(vec) < _embedding_dim:
            vec = vec + [0.0] * (_embedding_dim - len(vec))
        out.append(vec)
    return out


def _read_json(fp: str) -> Any:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_pdf(fp: str) -> List[Dict[str, Any]]:
    """Return a list of {text, page} extracted from a PDF file."""
    pages: List[Dict[str, Any]] = []
    try:
        reader = PdfReader(fp)
        for i, page in enumerate(reader.pages, start=1):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt.strip():
                pages.append({"text": txt, "page": i})
    except Exception as e:
        print(f"Skipping PDF {fp}: {e}")
    return pages


def index_inputs(inputs: List[str], namespace: str = "default", purge: bool = False):
    files = read_files_from_list(inputs)
    print(f"Found {len(files)} files to index from inputs")

    if purge:
        try:
            index.delete(delete_all=True, namespace=namespace)
            print(f"Purged existing vectors in namespace '{namespace}'")
        except Exception as e:
            msg = str(e)
            # Pinecone returns 404/Namespace not found if namespace hasn't been created yet — that's benign
            if "Namespace not found" in msg or "404" in msg:
                print(f"No existing namespace '{namespace}' to purge (will be created on first upsert).")
            else:
                print(f"Warning: failed to purge namespace '{namespace}': {e}")

    upserts = []
    counter = 0
    batch_size = 100
    for fp in files:
        try:
            size_ok = os.path.getsize(fp) <= MAX_FILE_SIZE_MB * 1024 * 1024
        except Exception:
            size_ok = True
        if not size_ok:
            print(f"Skipping {fp}: file size exceeds {MAX_FILE_SIZE_MB} MB limit")
            continue
        lower = fp.lower()
        if lower.endswith(".json"):
            data = _read_json(fp)
            if isinstance(data, list):
                for i, item in enumerate(data):
                    # Support Q/A pairs or generic items
                    if isinstance(item, dict) and ("question" in item and "answer" in item):
                        content = f"Q: {item['question']}\nA: {item['answer']}"
                        meta = {"source": fp, "type": "qa", "category": item.get("category", "general"), "text": content}
                    else:
                        content = json.dumps(item, ensure_ascii=False)
                        meta = {"source": fp, "type": "json", "text": content}

                    emb = get_embedding(content)
                    doc_id = f"{os.path.basename(fp)}-json-{i}"
                    upserts.append((doc_id, emb, meta))
                    counter += 1
            else:
                print(f"Skipping {fp}: unsupported JSON structure")
        elif lower.endswith(".pdf"):
            pdf_pages = _read_pdf(fp)
            for pg in pdf_pages:
                raw = pg["text"] or ""
                # Trim early to avoid huge memory from splitting/joining giant strings
                limited = raw[: max(PDF_PAGE_MAX_CHARS, 10000)]  # process at least 10k, up to cap
                # Normalize whitespace without creating massive intermediate lists
                normalized = re.sub(r"\s+", " ", limited).strip()
                text = normalized[:PDF_PAGE_MAX_CHARS]
                page = pg["page"]
                chunks = chunk_text(text)
                if not chunks:
                    continue
                # Batch-embed page chunks
                embs = get_embeddings_batch(chunks)
                for i, (chunk, emb) in enumerate(zip(chunks, embs)):
                    doc_id = f"{os.path.basename(fp)}-p{page}-{i}"
                    metadata = {"source": f"{fp}#page={page}", "text": chunk, "page": page}
                    upserts.append((doc_id, emb, metadata))
                    counter += 1
                    if len(upserts) >= batch_size:
                        payload = [
                            {"id": u[0], "values": u[1], "metadata": u[2]}
                            for u in upserts
                        ]
                        index.upsert(vectors=payload, namespace=namespace)
                        upserts = []
        else:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"Skipping {fp}: {e}")
                continue

            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                doc_id = f"{os.path.basename(fp)}-{i}"
                emb = get_embedding(chunk)
                metadata = {"source": fp, "text": chunk, "namespace": namespace}
                upserts.append((doc_id, emb, metadata))
                counter += 1

        # send batches of 100
        if len(upserts) >= batch_size:
            payload = [
                {"id": u[0], "values": u[1], "metadata": u[2]}
                for u in upserts
            ]
            index.upsert(vectors=payload, namespace=namespace)
            upserts = []
            print(f"Upserted {counter} vectors so far...")

    if upserts:
        payload = [
            {"id": u[0], "values": u[1], "metadata": u[2]}
            for u in upserts
        ]
        index.upsert(vectors=payload, namespace=namespace)

    print(f"Indexing complete. Total vectors upserted: {counter}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index documents/PDFs into Pinecone for RAG")
    parser.add_argument("paths", nargs="*", help="Files or directories to index (supports .pdf/.txt/.md/.json)")
    parser.add_argument("--business-id", dest="namespace", default="default", help="Business ID / Pinecone namespace to use")
    parser.add_argument("--purge", action="store_true", help="Delete all vectors in the namespace before indexing")
    args = parser.parse_args()
    if not args.paths:
        # fallback to current directory
        args.paths = [os.getcwd()]
    index_inputs(args.paths, namespace=args.namespace, purge=args.purge)
