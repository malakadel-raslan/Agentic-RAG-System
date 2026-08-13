"""
Tool #1: search_knowledge_base
Lets the agent decide WHEN to retrieve, and with WHAT query (it can rewrite
the user's question into a better search query before calling this).
"""
from langchain_core.tools import tool
from ingestion.vectorstore import load_vectorstore, similarity_search

_vectordb = None


def _get_db():
    global _vectordb
    if _vectordb is None:
        _vectordb = load_vectorstore()
    return _vectordb


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the document knowledge base for information relevant to `query`.
    Use this whenever you need facts, definitions, or details that might be
    contained in the ingested documents. Returns the top matching passages,
    each tagged with its source file and a relevance score, so you can cite
    them in your final answer.
    """
    db = _get_db()
    results = similarity_search(db, query)

    if not results:
        return "No relevant results found in the knowledge base."

    formatted = []
    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "?")
        formatted.append(
            f"[Source: {source} | chunk {chunk_id} | relevance: {score:.2f}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)
