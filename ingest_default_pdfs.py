import os
from index_documents import index_inputs

PDFS = [
    "AI AGENT.pdf",
    "AI AGENT LIVE STOCK LIST.pdf",
    "AI AGENT CONSIDERATION.pdf",
]

if __name__ == "__main__":
    # Validate files exist in current directory
    missing = [p for p in PDFS if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"Missing files: {missing}. Place the PDFs in the repo root or update PDFS list.")

    # Use namespace 'default' and purge anything previously ingested
    index_inputs(PDFS, namespace="default", purge=True)
    print("Done. Ingested PDFs into Pinecone namespace 'default'.")
