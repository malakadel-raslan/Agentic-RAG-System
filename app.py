"""
Streamlit UI for the Agentic RAG System.

Run with:
    streamlit run app.py
"""
import os
import streamlit as st

from config import DOCUMENTS_DIR, VECTORSTORE_DIR

st.set_page_config(page_title="Agentic RAG System", page_icon="🧠", layout="wide")

st.title("🧠 Agentic RAG System")
st.caption(
    "An AI agent that plans, decides its retrieval strategy, uses tools "
    "(search / summarize / calculate / analyze data), and answers complex "
    "questions from your knowledge base — with sources."
)

kb_ready = os.path.isdir(VECTORSTORE_DIR) and len(os.listdir(VECTORSTORE_DIR)) > 0

with st.sidebar:
    st.header("Knowledge Base")
    if os.path.isdir(DOCUMENTS_DIR):
        docs = [f for f in os.listdir(DOCUMENTS_DIR) if not f.startswith(".")]
    else:
        docs = []

    if docs:
        st.write(f"**{len(docs)} document(s)** in `data/documents/`:")
        for d in docs:
            st.write(f"- {d}")
    else:
        st.warning("No documents found in `data/documents/`. Add files there.")

    if kb_ready:
        st.success("Vector store is built and ready.")
    else:
        st.error("Vector store not built yet. Run `python build_kb.py` first.")

    st.divider()
    st.header("Tools available to the agent")
    st.markdown(
        "- 🔎 **search_knowledge_base**\n"
        "- 📝 **summarize_text**\n"
        "- 🧮 **calculate**\n"
        "- 📊 **analyze_data** (CSV files)"
    )
    show_debug = st.checkbox("Show agent plan & sources", value=True)

if not kb_ready:
    st.info(
        "Add documents to `data/documents/` and run `python build_kb.py` "
        "from the terminal, then reload this page."
    )
    st.stop()

# Lazy import so the app can render the sidebar even if the vector store
# isn't ready yet (avoids crashing on import before the KB exists).
from agent.agent_graph import answer_question  # noqa: E402

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if show_debug:
            if turn.get("sources"):
                st.caption("Sources: " + ", ".join(turn["sources"]))
            with st.expander("Agent plan"):
                st.write(turn.get("plan", ""))

question = st.chat_input("Ask a complex, multi-step question about your documents...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Planning and retrieving..."):
            result = answer_question(question)

        st.write(result["answer"])
        if show_debug:
            if result["sources"]:
                st.caption("Sources: " + ", ".join(result["sources"]))
            if not result["grounded"]:
                st.warning("The agent flagged this answer as weakly grounded in the documents.")
            with st.expander("Agent plan"):
                st.write(result["plan"])

    st.session_state.history.append(result | {"question": question})
