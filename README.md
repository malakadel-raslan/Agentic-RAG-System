# Agentic RAG System

An AI agent that **plans**, **decides its own retrieval strategy**, **uses tools**,
and answers complex, multi-step questions from a document knowledge base —
with citations. Built with LangChain + LangGraph, OpenAI models, Chroma, and
Streamlit.

## How this differs from Standard RAG

| Standard RAG | Agentic RAG (this project) |
|---|---|
| Fixed pipeline: always retrieve → always generate | Agent **decides** whether/when/how many times to retrieve |
| One search per question | Can issue **multiple, refined searches** for multi-part questions |
| No tools besides retrieval | Has **search, summarize, calculate, analyze_data** tools and picks among them |
| No self-check | Runs a **grounding check** on its own answer and retries with a better query if the answer isn't supported |

## Architecture

```
User question
      │
      ▼
 ┌─────────┐   short plan: does this need retrieval? multiple
 │ Planner │   searches? calculation? which tools?
 └────┬────┘
      ▼
 ┌────────────────────────────┐
 │ ReAct Agent (LangGraph)    │  loops, calling tools as needed:
 │  - search_knowledge_base   │──▶ Chroma vector search over your docs
 │  - summarize_text          │──▶ condenses long retrieved passages
 │  - calculate                │──▶ safe arithmetic evaluator
 │  - analyze_data             │──▶ pandas stats over CSV files
 └────────────┬───────────────┘
              ▼
      Draft answer + sources
              ▼
      ┌───────────────┐
      │ Grounding check│  is every claim supported by retrieved context?
      └──────┬────────┘
        no │      │ yes
           ▼      ▼
     retry once   final answer, shown with sources + the agent's plan
     (better query)
```

## Project layout

```
agentic-rag/
├── config.py                  # models, paths, chunk size, etc.
├── build_kb.py                 # run once to build the knowledge base
├── app.py                      # Streamlit UI
├── data/documents/             # put your source documents here (pdf/docx/txt/md/csv)
├── ingestion/
│   ├── loader.py                # extracts + cleans text from files
│   ├── chunker.py                # splits into overlapping chunks
│   └── vectorstore.py            # embeds + stores/queries Chroma
├── tools/
│   ├── retrieval_tool.py         # search_knowledge_base
│   ├── summarize_tool.py         # summarize_text
│   ├── calculator_tool.py        # calculate
│   └── data_analysis_tool.py     # analyze_data (CSV stats)
└── agent/
    ├── prompts.py                 # planner / agent / grounding-check prompts
    └── agent_graph.py              # plan → ReAct tool loop → verify → (retry)
```

## Setup

1. **Install dependencies** (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your OpenAI API key**:
   ```bash
   cp .env.example .env
   # then edit .env and paste your key
   ```

3. **Add documents**: drop PDFs, DOCX, TXT, MD, or CSV files into `data/documents/`.
   A sample handbook (`.txt`) and a sample revenue CSV are already included so
   you can try it immediately.

4. **Build the knowledge base**:
   ```bash
   python build_kb.py
   ```
   Re-run this any time you add/change documents.

5. **Launch the app**:
   ```bash
   streamlit run app.py
   ```

## Try these example questions

- *"What is INSTANT's remote work policy?"* — single retrieval
- *"Compare InstantInsight and InstantForecast revenue growth in 2023."* —
  multi-step: two searches + reasoning
- *"What was the percentage growth in InstantForecast revenue from Q1 to Q4 2023?"*
  — triggers `analyze_data` and/or `calculate` on `quarterly_revenue.csv`
- *"Summarize the engineering practices section."* — retrieval + `summarize_text`
- *"What is the CEO's favorite color?"* — should say the knowledge base has no
  relevant information rather than making something up

## Extending it

- **More tools**: add a new `@tool`-decorated function under `tools/` and add
  it to the `TOOLS` list in `agent/agent_graph.py`.
- **Different vector store**: swap `Chroma` in `ingestion/vectorstore.py` for
  Pinecone/Weaviate/FAISS — the rest of the app is unaffected.
- **Different LLM**: change `CHAT_MODEL` in `config.py`.
