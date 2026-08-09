"""
Retriever Module for Structure-Aware RAG Pipeline

Performs vector similarity search against the Supabase `knowledge_chunks` table
using Gemini embeddings. Supports both unfiltered retrieval and category-filtered
retrieval (powered by the classifier).

RPC functions used:
    - match_documents: Unfiltered top-K similarity search
    - match_documents_filtered: Category-filtered top-K similarity search
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from google import genai
from supabase import create_client, Client

# Load environment variables from .env in the project root
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Module-level client singletons
_supabase_client: Optional[Client] = None
_ai_client: Optional[genai.Client] = None


def _get_clients() -> tuple[Client, genai.Client]:
    """
    Initializes and returns Supabase and Gemini AI clients (singleton pattern).

    Returns:
        tuple[Client, genai.Client]: Initialized Supabase and Gemini client instances.
    """
    global _supabase_client, _ai_client

    if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
        raise ValueError(
            "Missing environment credentials. Ensure SUPABASE_URL, SUPABASE_KEY, "
            "and GEMINI_API_KEY are configured in the .env file."
        )

    if _supabase_client is None:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    if _ai_client is None:
        _ai_client = genai.Client(api_key=GEMINI_API_KEY)

    return _supabase_client, _ai_client


def embed(text: str) -> List[float]:
    """
    Generates a 768-dimensional vector embedding for the input text
    using Gemini's gemini-embedding-001 model.

    Args:
        text: Content string to embed.

    Returns:
        List[float]: 768-dimensional float embedding vector.
    """
    _, ai = _get_clients()
    response = ai.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={"output_dimensionality": 768},
    )
    return response.embeddings[0].values


def retrieve(
    query: str,
    categories: Optional[Union[str, List[str]]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieves the most relevant knowledge chunks for a query using vector
    similarity search. Optionally filters results by category.

    Args:
        query: The user's natural language query string.
        categories: Optional category filter.
            - None: Unfiltered search across all chunks.
            - str: Single category name to filter by.
            - List[str]: Multiple category names to filter by.
        top_k: Maximum number of results to return (default: 5).

    Returns:
        List[Dict[str, Any]]: List of matching chunks, each containing:
            - id (str): Chunk UUID
            - content (str): Chunk text content
            - type (str): Source type ("product", "faq", "policy")
            - category (str): Category classification
            - source_id (str): Original document ID
            - similarity (float): Cosine similarity score (0-1)
    """
    db, _ = _get_clients()
    query_embedding = embed(query)

    # Normalize categories to a list (or None for unfiltered)
    if isinstance(categories, str):
        filter_list = [categories]
    elif isinstance(categories, list) and len(categories) > 0:
        filter_list = categories
    else:
        filter_list = None

    if filter_list:
        # Category-filtered retrieval
        response = db.rpc("match_documents_filtered", {
            "query_embedding": query_embedding,
            "filter_categories": filter_list,
            "match_count": top_k,
        }).execute()
    else:
        # Unfiltered retrieval
        response = db.rpc("match_documents", {
            "query_embedding": query_embedding,
            "match_count": top_k,
        }).execute()

def retrieve_top_k_categories(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieves the top K categories for a query based on cosine similarity
    against knowledge_chunks in Supabase.

    Args:
        query: User input query.
        top_k: Number of top categories to return (default: 3).

    Returns:
        List[Dict[str, Any]]: List of top matching categories with similarity scores.
    """
    db, _ = _get_clients()
    query_embedding = embed(query)

    response = db.rpc("classify_query", {
        "query_embedding": query_embedding,
        "match_threshold": 0.0,
    }).execute()

    results = response.data or []
    return [
        {"category": r["category"], "similarity": round(r["avg_similarity"], 4)}
        for r in results[:top_k]
    ]



# ---------------------------------------------------------------------------
# Standalone test — run with: python query/retriever.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Retriever Module — Standalone Test")
    print("=" * 60)

    test_query = "How do I request a payout?"

    # Test 1: Unfiltered retrieval
    print(f"\n--- Unfiltered retrieval for: '{test_query}' ---")
    results = retrieve(test_query, top_k=3)
    for i, chunk in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"    Category:   {chunk['category']}")
        print(f"    Type:       {chunk['type']}")
        print(f"    Source:     {chunk['source_id']}")
        print(f"    Similarity: {chunk['similarity']:.4f}")
        print(f"    Content:    {chunk['content'][:120]}...")

    # Test 2: Filtered retrieval
    print(f"\n--- Filtered retrieval (category='Payouts') for: '{test_query}' ---")
    results = retrieve(test_query, categories="Payouts", top_k=3)
    for i, chunk in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"    Category:   {chunk['category']}")
        print(f"    Type:       {chunk['type']}")
        print(f"    Source:     {chunk['source_id']}")
        print(f"    Similarity: {chunk['similarity']:.4f}")
        print(f"    Content:    {chunk['content'][:120]}...")

    # Test 3: Multi-category retrieval
    print(f"\n--- Multi-category retrieval (Payouts + Billing) for: '{test_query}' ---")
    results = retrieve(test_query, categories=["Payouts", "Billing"], top_k=3)
    for i, chunk in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"    Category:   {chunk['category']}")
        print(f"    Type:       {chunk['type']}")
        print(f"    Source:     {chunk['source_id']}")
        print(f"    Similarity: {chunk['similarity']:.4f}")
        print(f"    Content:    {chunk['content'][:120]}...")
