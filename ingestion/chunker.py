"""
Splits loaded documents into overlapping chunks, preserving source + chunk index
metadata so the agent can always cite where an answer came from.
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    documents: [{"source": str, "text": str}, ...]
    returns: [{"source": str, "chunk_id": int, "text": str}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append({
                "source": doc["source"],
                "chunk_id": i,
                "text": piece,
            })
    return chunks
