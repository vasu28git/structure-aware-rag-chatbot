"""
ask.py — Phase B: Query Pipeline Entry Point

This is the top-level wiring module that ties together:
  retriever.classify_query()  ->  retriever.retrieve()  ->  generate.generate_answer()

Public API:
  ask(query: str) -> str
    Full end-to-end RAG pipeline. Returns the grounded answer string.

Usage:
  python query/ask.py                  # interactive terminal REPL
  python query/ask.py "Your question"  # single-shot from command line

Optional FastAPI endpoint:
  uvicorn ask:app --reload             # starts POST /ask on port 8000
"""

import sys
from retriever import classify_query, retrieve
from generate import generate_answer

# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def ask(query: str) -> str:
    """
    Run the full RAG pipeline for a single customer query.

    Steps:
      1. classify_query  — embed query, find best-matching category (or None).
      2. retrieve        — metadata-filtered cosine similarity search via Supabase RPC.
      3. generate_answer — build grounded prompt, call Gemini Flash, return answer.

    Args:
        query: Raw customer question string.

    Returns:
        str: Grounded answer, or a safe fallback if retrieval returns nothing.
    """
    query = query.strip()
    if not query:
        return "Please enter a question so I can help you."

    # Step 1: Classify
    category = classify_query(query)

    # Step 2: Retrieve (filter by category if confident, else global search)
    chunks = retrieve(query, category=category, top_k=3)

    # Step 3: Generate (generate_answer handles empty chunks with a fallback)
    answer = generate_answer(query, chunks)

    return answer


# ---------------------------------------------------------------------------
# Optional FastAPI endpoint
# ---------------------------------------------------------------------------
# Only imported/exposed if fastapi is installed. ask.py works without it.

try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(
        title="ShopRAG Customer Support API",
        description=(
            "Structure-aware, metadata-filtered RAG chatbot for e-commerce "
            "customer support. Answers are grounded in retrieved company data only."
        ),
        version="1.0.0",
    )

    class AskRequest(BaseModel):
        query: str

    class AskResponse(BaseModel):
        query: str
        category: str | None
        answer: str
        chunks_retrieved: int

    @app.post("/ask", response_model=AskResponse)
    def ask_endpoint(request: AskRequest):
        """
        POST /ask

        Body: { "query": "Your question here" }
        Returns: { "query": ..., "category": ..., "answer": ..., "chunks_retrieved": ... }
        """
        query = request.query.strip()
        if not query:
            return AskResponse(
                query=query,
                category=None,
                answer="Please provide a non-empty question.",
                chunks_retrieved=0,
            )

        category = classify_query(query)
        chunks = retrieve(query, category=category, top_k=3)
        answer = generate_answer(query, chunks)

        return AskResponse(
            query=query,
            category=category,
            answer=answer,
            chunks_retrieved=len(chunks),
        )

except ImportError:
    # FastAPI not installed — terminal mode only, which is fine per spec.
    app = None


# ---------------------------------------------------------------------------
# Terminal REPL
# ---------------------------------------------------------------------------

def _run_repl() -> None:
    """
    Interactive terminal loop. Type a question and press Enter to get an answer.
    Type 'exit' or 'quit' (or press Ctrl-C / Ctrl-D) to stop.
    """
    print("=" * 60)
    print("  ShopRAG — Customer Support Chatbot")
    print("  Type your question and press Enter.")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 60)
    print()

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        print()
        answer = ask(query)
        print(f"Bot: {answer}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single-shot mode: python query/ask.py "my question here"
        single_query = " ".join(sys.argv[1:])
        print(f"\nQuery: {single_query}")
        print(f"\nAnswer: {ask(single_query)}\n")
    else:
        # Interactive REPL mode
        _run_repl()
