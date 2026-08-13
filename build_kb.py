"""
Run this once (and again whenever documents change) to build the knowledge base:

    python build_kb.py

It reads every file in data/documents/, cleans + chunks the text, embeds it,
and persists it into the local Chroma vector store.
"""
from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from ingestion.vectorstore import build_vectorstore
from config import DOCUMENTS_DIR


def main():
    print(f"Loading documents from {DOCUMENTS_DIR} ...")
    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        print("No documents found. Add PDF/DOCX/TXT/MD files to data/documents/ and re-run.")
        return
    print(f"Loaded {len(documents)} document(s): {[d['source'] for d in documents]}")

    print("Chunking ...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding + storing in Chroma ...")
    build_vectorstore(chunks)
    print("Done. Knowledge base is ready.")


if __name__ == "__main__":
    main()
