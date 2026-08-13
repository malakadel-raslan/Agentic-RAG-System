"""
Tool #2: summarize_text
Useful when retrieved context is long/spans multiple chunks and the agent
wants a condensed version before reasoning over it or before citing it.
"""
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, CHAT_MODEL

_llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0)


@tool
def summarize_text(text: str) -> str:
    """
    Summarize a block of text (e.g. long retrieved passages) into a short,
    dense summary that preserves key facts, numbers, and named entities.
    Use this when retrieved content is too long to reason over directly.
    """
    if len(text) < 400:
        return text  # already short enough, no need to summarize

    prompt = (
        "Summarize the following text into a concise summary (5-8 sentences max). "
        "Preserve all specific facts, numbers, names, and dates. Do not add opinions.\n\n"
        f"TEXT:\n{text}"
    )
    response = _llm.invoke(prompt)
    return response.content
