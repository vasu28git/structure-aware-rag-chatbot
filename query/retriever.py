"""
retriever.py — Phase B: Query Classification + Metadata-Filtered Retrieval

Two responsibilities:
  1. classify_query(text) -> str | None
     Embeds the query and compares it against per-category reference embeddings
     (computed ONCE at module load). The category with the highest cosine similarity
     wins; returns None if the best score falls below the confidence threshold,
     which triggers an unfiltered global search.

  2. retrieve(query_text, category, top_k) -> list[dict]
     Embeds the query, calls the Supabase `match_chunks` RPC with an optional
     category filter, and returns the top-k ranked chunks.

IMPORTANT SDK NOTES (from project spec §4):
  - Use `from google import genai` (package google-genai). Do NOT mix with
    the old google.generativeai package.
  - Model: "gemini-embedding-001", explicit output_dimensionality=768.
  - Embedding value is at response.embeddings[0].values.
  - The same model + dimension MUST be used in ingestion and here, or cosine
    similarity across the two spaces is meaningless.
"""

import os
import math
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from google import genai
from supabase import create_client, Client

# ---------------------------------------------------------------------------
# Environment & client initialisation
# ---------------------------------------------------------------------------

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Confidence threshold: below this score the classifier returns None
# (no filter, search the whole table). Tune between 0.55 – 0.70.
CONFIDENCE_THRESHOLD: float = 0.60

# Top-k chunks to retrieve per query
DEFAULT_TOP_K: int = 3

_supabase_client: Optional[Client] = None
_ai_client: Optional[genai.Client] = None


def _init_clients() -> tuple[Client, genai.Client]:
    """
    Lazily initialise and cache the Supabase and Gemini clients.

    Returns:
        tuple[Client, genai.Client]: Ready-to-use client pair.

    Raises:
        ValueError: If any required environment variable is missing.
    """
    global _supabase_client, _ai_client

    if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
        raise ValueError(
            "Missing credentials. Ensure SUPABASE_URL, SUPABASE_KEY, and "
            "GEMINI_API_KEY are set in the .env file at the project root."
        )

    if _supabase_client is None:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    if _ai_client is None:
        _ai_client = genai.Client(api_key=GEMINI_API_KEY)

    return _supabase_client, _ai_client


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _embed(text: str) -> list[float]:
    """
    Generate a 768-dimensional embedding for *text* using gemini-embedding-001.

    Args:
        text: The string to embed.

    Returns:
        list[float]: 768-element float vector.

    Raises:
        RuntimeError: Wraps any API-level error with a descriptive message.
    """
    _, ai = _init_clients()
    try:
        response = ai.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config={"output_dimensionality": 768},
        )
        return response.embeddings[0].values
    except Exception as exc:
        raise RuntimeError(
            f"Embedding failed for text snippet '{text[:60]}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Cosine similarity (pure Python, no numpy dependency)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two equal-length vectors.

    Returns:
        float in [-1, 1]; returns 0.0 if either vector has zero magnitude.
    """
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Reference embedding bank — built ONCE at module load
# ---------------------------------------------------------------------------
# Category keys MUST exactly match the `category` column values in Supabase.
# Each list contains 2-4 representative questions for that category.
# Adding more diverse examples improves classifier accuracy.

CATEGORY_EXAMPLES: dict[str, list[str]] = {
    "Refund Policy": [
        "Can I get a refund?",
        "Will I get my money back if I cancel?",
        "Are purchases refundable?",
        "What is the refund policy?",
    ],
    "Billing": [
        "Why was my card charged?",
        "How do I download my invoice or receipt?",
        "My payment failed, what should I do?",
        "Why was I charged $1 for a free account?",
        "My card was declined, how do I fix it?",
    ],
    "Payouts": [
        "How do I request a payout?",
        "How long do payouts take to process?",
        "What payout methods are available?",
        "Can I use PayPal to withdraw my earnings?",
        "Is there a minimum number of trading days before I can withdraw?",
    ],
    "Shipping": [
        "How can I track my order?",
        "How long does delivery take?",
        "Where is my package?",
    ],
    "Payments": [
        "What payment methods do you accept?",
        "Can I pay by UPI or credit card?",
        "Do you accept PayPal for purchases?",
    ],
    "KYC & Taxes": [
        "When do I need to complete identity verification?",
        "Do you send 1099 tax forms?",
        "What documents are needed for KYC?",
    ],
    "Promo Codes": [
        "My promo code is not working",
        "How do I apply a discount code?",
        "I saw a code on social media but it does not work",
    ],
    "Trading & Payouts": [
        "If I trade after requesting a payout which cycle do profits count toward?",
        "Profits made after payout approval — which cycle are they for?",
    ],
    "Evaluation Comparison": [
        "What is the difference between SELECT and Growth evaluation?",
        "How do SELECT and Growth accounts compare?",
        "Which evaluation type should I choose?",
    ],
    "Evaluation Rules": [
        "What are the rules for the evaluation?",
        "Does Growth have a consistency rule?",
        "What is the soft breach daily loss limit?",
        "How many trading days do I need to pass evaluation?",
    ],
    "Account Limits": [
        "How many accounts can I hold at the same time?",
        "What is the maximum number of evaluation purchases allowed?",
        "How many resets can I do on my evaluation?",
    ],
    "Digital Delivery": [
        "When will I receive my account after purchase?",
        "How is my account delivered after I buy it?",
    ],
    "Payout Policies": [
        "What is the Select Flex payout policy?",
        "How does the Select Daily payout policy work?",
        "What are the payout caps?",
        "How often can I withdraw profits?",
        "What is the difference between Flex and Daily payout?",
    ],
    "Risk & Scaling": [
        "How does the contract scaling plan work?",
        "When can I increase my contract size?",
        "What are the maximum drawdown limits?",
    ],
    "Account Progression": [
        "How do I qualify for a live account?",
        "What is Tradeify Elite and how do I get it?",
        "How many payouts do I need before getting a live account?",
    ],
    "Evaluation Accounts": [
        "Tell me about the 25K evaluation account",
        "What are the parameters for the 100K account?",
        "What is the profit target for 50K Select?",
    ],
}

# Embed all reference questions at module load time (happens once per process).
# Structure: {category: [embedding_vector, ...]}
print("retriever.py: Building reference embedding bank...", flush=True)
_REFERENCE_EMBEDDINGS: dict[str, list[list[float]]] = {}

for _cat, _questions in CATEGORY_EXAMPLES.items():
    _REFERENCE_EMBEDDINGS[_cat] = []
    for _q in _questions:
        try:
            _vec = _embed(_q)
            _REFERENCE_EMBEDDINGS[_cat].append(_vec)
        except Exception as _err:
            print(f"  Warning: Could not embed reference question '{_q[:60]}': {_err}")

print(
    f"retriever.py: Reference bank ready - "
    f"{sum(len(v) for v in _REFERENCE_EMBEDDINGS.values())} embeddings across "
    f"{len(_REFERENCE_EMBEDDINGS)} categories.",
    flush=True,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_query(query_text: str) -> Optional[str]:
    """
    Classify *query_text* into one of the known categories using cosine
    similarity against pre-computed reference embeddings.

    Algorithm:
      1. Embed the query.
      2. For each category, compute the max cosine similarity across all its
         reference embeddings.
      3. The category with the globally highest max score wins.
      4. If that winning score < CONFIDENCE_THRESHOLD, return None
         (caller should search without a category filter).

    Args:
        query_text: The raw customer question string.

    Returns:
        str | None: Winning category string, or None if below threshold.
    """
    if not query_text or not query_text.strip():
        return None

    try:
        query_vec = _embed(query_text)
    except RuntimeError as err:
        print(
            f"classify_query: Embedding error - {err}. "
            "Falling back to unfiltered search."
        )
        return None

    best_category: Optional[str] = None
    best_score: float = -1.0

    for category, ref_vectors in _REFERENCE_EMBEDDINGS.items():
        if not ref_vectors:
            continue
        # Take the single highest similarity across all reference examples
        category_score = max(
            _cosine_similarity(query_vec, ref_vec) for ref_vec in ref_vectors
        )
        if category_score > best_score:
            best_score = category_score
            best_category = category

    if best_score < CONFIDENCE_THRESHOLD:
        print(
            f"classify_query: Best score {best_score:.3f} < threshold "
            f"{CONFIDENCE_THRESHOLD}. Using unfiltered search.",
            flush=True,
        )
        return None

    print(
        f"classify_query: '{query_text[:60]}' -> '{best_category}' "
        f"(score={best_score:.3f})",
        flush=True,
    )
    return best_category


def retrieve(
    query_text: str,
    category: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Embed *query_text* and call the Supabase `match_chunks` RPC to fetch the
    most semantically similar chunks, optionally filtered by *category*.

    Args:
        query_text: The customer question to search for.
        category:   Category string to filter on, or None for global search.
        top_k:      Maximum number of chunks to return (default 3).

    Returns:
        list[dict]: Each dict has keys: content, type, category, source_id,
                    similarity. Empty list if retrieval fails or no rows found.
    """
    if not query_text or not query_text.strip():
        return []

    db, _ = _init_clients()

    try:
        query_vec = _embed(query_text)
    except RuntimeError as err:
        print(f"retrieve: Embedding error - {err}. Returning empty results.")
        return []

    rpc_params: dict = {
        "query_embedding": query_vec,
        "match_count": top_k,
    }
    if category is not None:
        rpc_params["filter_category"] = category

    try:
        response = db.rpc("match_chunks", rpc_params).execute()
        rows: list[dict] = response.data or []

        if not rows:
            print(
                f"retrieve: No chunks found for query='{query_text[:60]}' "
                f"category='{category}'.",
                flush=True,
            )
        else:
            print(
                f"retrieve: Found {len(rows)} chunk(s) "
                f"[category filter='{category}', top_k={top_k}].",
                flush=True,
            )

        return rows

    except Exception as exc:
        print(f"retrieve: Supabase RPC error - {exc}. Returning empty results.")
        return []


# ---------------------------------------------------------------------------
# Quick smoke test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_QUERIES = [
        "Can I get a refund after buying an account?",
        "How long does it take to get my payout?",
        "What is the difference between Select and Growth evaluation?",
        "My card was declined, what should I do?",
        "How do I track my order?",
    ]

    print("\n=== Retriever Smoke Test ===\n")
    for q in TEST_QUERIES:
        print(f"Query: {q}")
        cat = classify_query(q)
        chunks = retrieve(q, category=cat, top_k=3)
        print(f"  -> Category : {cat}")
        print(f"  -> Chunks   : {len(chunks)} retrieved")
        for i, chunk in enumerate(chunks, 1):
            sim = chunk.get("similarity", 0)
            snippet = chunk.get("content", "")[:80]
            print(f"     [{i}] (sim={sim:.3f}) {snippet}...")
        print()
