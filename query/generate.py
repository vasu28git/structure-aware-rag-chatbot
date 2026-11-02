"""
generate.py — Phase B: Grounded Answer Generation

Responsibility: Given a customer question and a list of retrieved context chunks,
build a grounded prompt and call Gemini Flash to produce an answer.

Grounding contract (enforced in the system instruction):
  - The model MUST answer using ONLY the provided context.
  - It MUST NOT use its own general knowledge.
  - If the context is insufficient, it must say so and suggest contacting support.

IMPORTANT SDK NOTES (from project spec §4):
  - Use `from google import genai` (package google-genai).
  - Verify the LLM model name against what the API key can access.
  - Model used: "gemini-2.0-flash"
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from google import genai

# ---------------------------------------------------------------------------
# Environment & client initialisation
# ---------------------------------------------------------------------------

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini Flash model for answer generation
GENERATION_MODEL: str = "gemini-2.0-flash"

# Safe fallback message returned when context is empty or generation fails
FALLBACK_MESSAGE: str = (
    "I'm sorry, I wasn't able to find specific information to answer your question. "
    "Please contact our support team directly for assistance."
)

_ai_client: Optional[genai.Client] = None


def _init_client() -> genai.Client:
    """
    Lazily initialise and cache the Gemini client.

    Returns:
        genai.Client: Ready-to-use Gemini client.

    Raises:
        ValueError: If GEMINI_API_KEY is not set.
    """
    global _ai_client
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to the .env file at the project root."
        )
    if _ai_client is None:
        _ai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _ai_client


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

# System instruction enforcing grounded-only answers.
_SYSTEM_INSTRUCTION = """You are a customer support assistant for Tradeify, a proprietary trading firm.

STRICT RULES — follow these without exception:
1. Answer ONLY using the context passages provided below. Do NOT use any outside knowledge or general assumptions.
2. If the provided context does not contain enough information to answer the question, say:
   "I'm not sure based on the information I have. Please contact Tradeify support for accurate assistance."
3. Be concise, clear, and professional. Do not repeat the question back.
4. Do not fabricate numbers, dates, policies, or account parameters that are not explicitly stated in the context.
5. If the question is outside the scope of customer support (e.g., asking for trading advice or investment recommendations), politely decline and redirect to support.
"""


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Construct a grounded prompt combining the system instruction, retrieved
    context chunks, and the customer question.

    Args:
        query:  The raw customer question.
        chunks: List of chunk dicts from retriever.retrieve(), each with keys:
                content, type, category, source_id, similarity.

    Returns:
        str: Full prompt string ready to send to the LLM.
    """
    context_lines = []
    for i, chunk in enumerate(chunks, 1):
        chunk_type = chunk.get("type", "unknown")
        chunk_cat = chunk.get("category", "unknown")
        content = chunk.get("content", "").strip()
        context_lines.append(
            f"--- Context [{i}] | Type: {chunk_type} | Category: {chunk_cat} ---\n{content}"
        )

    context_block = "\n\n".join(context_lines)

    prompt = (
        f"{_SYSTEM_INSTRUCTION}\n\n"
        f"=== CONTEXT ===\n"
        f"{context_block}\n\n"
        f"=== CUSTOMER QUESTION ===\n"
        f"{query}\n\n"
        f"=== YOUR ANSWER ===\n"
    )
    return prompt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Generate a grounded answer for *query* using *chunks* as the sole context.

    If *chunks* is empty, returns the safe fallback message without making an
    LLM call (avoids cost and avoids hallucination on empty context).

    Args:
        query:  The customer question string.
        chunks: Retrieved context chunks from retriever.retrieve().

    Returns:
        str: The generated answer, or the fallback message on failure/empty context.
    """
    if not chunks:
        print(
            "generate_answer: No context chunks provided. Returning fallback message.",
            flush=True,
        )
        return FALLBACK_MESSAGE

    prompt = build_prompt(query, chunks)

    try:
        client = _init_client()
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
        )
        answer = response.text.strip()
        print(
            f"generate_answer: Generated answer ({len(answer)} chars) "
            f"from {len(chunks)} chunk(s).",
            flush=True,
        )
        return answer

    except Exception as exc:
        print(f"generate_answer: LLM call failed - {exc}. Returning fallback message.")
        return FALLBACK_MESSAGE


# ---------------------------------------------------------------------------
# Quick smoke test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== generate.py Smoke Test ===\n")

    # Test 1: With realistic mock chunks
    mock_chunks = [
        {
            "content": (
                "Tradeify does not offer refunds. By completing your purchase, "
                "you agree to our Terms of Service stating all sales are strictly final. "
                "This applies to all evaluation and funded accounts regardless of whether "
                "they have been traded on."
            ),
            "type": "policy",
            "category": "Refund Policy",
            "source_id": "pol_001",
            "similarity": 0.92,
        }
    ]
    test_query = "Can I get a refund if I change my mind?"
    print(f"Query: {test_query}")
    answer = generate_answer(test_query, mock_chunks)
    print(f"Answer:\n{answer}\n")

    # Test 2: Empty chunks — should return fallback
    print("Query: (empty context test)")
    fallback = generate_answer("What is the return policy?", [])
    print(f"Fallback:\n{fallback}\n")
