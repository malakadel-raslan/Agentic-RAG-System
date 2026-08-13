"""
Central configuration for the Agentic RAG System.
"""
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Models ---
CHAT_MODEL = "gpt-4o-mini"       # fast + cheap, good for agent planning + tool calls
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "data", "documents")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore", "chroma_db")
COLLECTION_NAME = "agentic_rag_kb"

# --- Chunking ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# --- Retrieval ---
TOP_K = 5

# --- Agent ---
MAX_AGENT_STEPS = 8  # safety limit on tool-calling loop iterations
