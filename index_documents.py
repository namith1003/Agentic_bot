import os
import glob
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
import openai
import pinecone

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "malaysia-rag")

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_ENV:
    raise RuntimeError("Please set OPENAI_API_KEY, PINECONE_API_KEY and PINECONE_ENV in your environment")

openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX_NAME)


def read_files_from_dir(path: str, patterns: List[str] = ["**/*.txt", "**/*.md", "**/*.json"]):
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(path, p), recursive=True))
    return files


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    tokens = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        tokens.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return tokens


def get_embedding(text: str):
    resp = openai.Embedding.create(model="text-embedding-3-small", input=text)
    return resp["data"][0]["embedding"]


def _read_json(fp: str) -> Any:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def index_directory(directory: str, namespace: str = "malaysia"):
    files = read_files_from_dir(directory)
    print(f"Found {len(files)} files to index in {directory}")

    upserts = []
    counter = 0
    for fp in files:
        if fp.lower().endswith(".json"):
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
        if len(upserts) >= 100:
            to_upsert = [(u[0], u[1], u[2]) for u in upserts]
            index.upsert(vectors=to_upsert, namespace=namespace)
            upserts = []
            print(f"Upserted {counter} vectors so far...")

    if upserts:
        to_upsert = [(u[0], u[1], u[2]) for u in upserts]
        index.upsert(vectors=to_upsert, namespace=namespace)

    print(f"Indexing complete. Total vectors upserted: {counter}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index documents into Pinecone for Malaysia RAG")
    parser.add_argument("directory", help="Directory containing documents to index")
    parser.add_argument("--business-id", dest="namespace", default="malaysia", help="Business ID / Pinecone namespace to use")
    args = parser.parse_args()

    index_directory(args.directory, namespace=args.namespace)
