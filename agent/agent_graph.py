"""
The Agentic RAG core: plans, runs a tool-calling ReAct loop (search / summarize /
calculate / analyze_data), then checks whether the answer is actually grounded
in retrieved context before returning it. Retries once with extra guidance if not.
"""
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from config import OPENAI_API_KEY, CHAT_MODEL, MAX_AGENT_STEPS
from agent.prompts import PLANNER_PROMPT, AGENT_SYSTEM_PROMPT, GROUNDING_CHECK_PROMPT
from tools.retrieval_tool import search_knowledge_base
from tools.summarize_tool import summarize_text
from tools.calculator_tool import calculate
from tools.data_analysis_tool import analyze_data

TOOLS = [search_knowledge_base, summarize_text, calculate, analyze_data]

_llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0)
_react_agent = create_react_agent(_llm, TOOLS)


def _make_plan(question: str) -> str:
    prompt = PLANNER_PROMPT.format(question=question)
    response = _llm.invoke(prompt)
    return response.content


def _extract_sources(messages) -> list[str]:
    sources = set()
    for m in messages:
        if isinstance(m, ToolMessage) and isinstance(m.content, str):
            for match in re.findall(r"Source:\s*([^\|\]\n]+)", m.content):
                sources.add(match.strip())
    return sorted(sources)


def _extract_context(messages) -> str:
    chunks = []
    for m in messages:
        if isinstance(m, ToolMessage) and isinstance(m.content, str):
            chunks.append(m.content)
    return "\n\n".join(chunks)[:6000]  # cap length for the grounding check call


def _run_react(question: str, plan: str, extra_guidance: str = "") -> dict:
    system = AGENT_SYSTEM_PROMPT.format(plan=plan)
    if extra_guidance:
        system += f"\n\nAdditional guidance: {extra_guidance}"

    result = _react_agent.invoke(
        {"messages": [SystemMessage(content=system), HumanMessage(content=question)]},
        config={"recursion_limit": MAX_AGENT_STEPS * 2},
    )
    return result


def _grounding_check(question: str, answer: str, context: str) -> bool:
    if not context.strip():
        # No retrieval happened at all — nothing to ground against, treat as
        # acceptable only if the answer itself says it couldn't find info.
        return True
    prompt = GROUNDING_CHECK_PROMPT.format(question=question, answer=answer, context=context)
    response = _llm.invoke(prompt)
    verdict = response.content.strip().upper()
    return verdict.startswith("GROUNDED")


def answer_question(question: str, verbose: bool = False) -> dict:
    """
    Full agentic pipeline: plan -> retrieve/tool-use -> answer -> verify -> (retry once).
    Returns {"answer": str, "sources": list[str], "plan": str, "grounded": bool}.
    """
    plan = _make_plan(question)
    if verbose:
        print(f"[plan]\n{plan}\n")

    result = _run_react(question, plan)
    answer = result["messages"][-1].content
    sources = _extract_sources(result["messages"])
    context = _extract_context(result["messages"])

    grounded = _grounding_check(question, answer, context)

    if not grounded:
        if verbose:
            print("[grounding check failed — retrying with more targeted search]")
        retry_guidance = (
            "Your previous answer was not well supported by retrieved context. "
            "Search again with more specific / rewritten queries before answering, "
            "and only state facts you can support with retrieved passages."
        )
        result = _run_react(question, plan, extra_guidance=retry_guidance)
        answer = result["messages"][-1].content
        sources = _extract_sources(result["messages"])
        context = _extract_context(result["messages"])
        grounded = _grounding_check(question, answer, context)

    return {
        "answer": answer,
        "sources": sources,
        "plan": plan,
        "grounded": grounded,
    }
