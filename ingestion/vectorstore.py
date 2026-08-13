"""
Thin wrapper around Chroma for building and querying the knowledge base.
"""
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, VECTORSTORE_DIR, COLLECTION_NAME, TOP_K
)


def get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)


def build_vectorstore(chunks: list[dict]) -> Chroma:
    """
    Embeds and stores chunks. `chunks`: [{"source", "chunk_id", "text"}, ...]
    Persists to disk at VECTORSTORE_DIR.
    """
    docs = [
        Document(
            page_content=c["text"],
            metadata={"source": c["source"], "chunk_id": c["chunk_id"]},
        )
        for c in chunks
    ]

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=VECTORSTORE_DIR,
    )
    return vectordb


def load_vectorstore() -> Chroma:
    """Loads an existing persisted Chroma collection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=VECTORSTORE_DIR,
    )


def similarity_search(vectordb: Chroma, query: str, k: int = TOP_K):
    """Returns list of (Document, score) tuples, best first."""
    return vectordb.similarity_search_with_relevance_scores(query, k=k)
