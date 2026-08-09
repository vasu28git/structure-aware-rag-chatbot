"""
Rule-Based Query Classifier for Structure-Aware RAG Pipeline

Classifies incoming queries into existing categories stored in the Supabase
`knowledge_chunks` table using ONLY embedding cosine similarity and
deterministic threshold rules. Zero LLM / generative model calls.

Classification flow:
    1. Embed the query using Gemini embedding model (gemini-embedding-001)
    2. Call `classify_query` RPC — computes avg cosine similarity per category
    3. Apply threshold rules to determine the result:
       - Clear winner  → single category
       - Ambiguous      → top 2 categories
       - No match       → empty (caller should refuse or search all)
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
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


def _embed(text: str) -> List[float]:
    """
    Generates a 768-dimensional vector embedding for the input text.

    Args:
        text: Content string to embed.

    Returns:
        List[float]: 768-dimensional embedding vector.
    """
    _, ai = _get_clients()
    response = ai.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={"output_dimensionality": 768},
    )
    return response.embeddings[0].values


@dataclass
class ClassificationResult:
    """Holds the result of a query classification."""

    categories: List[str] = field(default_factory=list)
    top_score: float = 0.0
    is_ambiguous: bool = False
    all_scores: List[tuple] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        """True if at least one category was matched above threshold."""
        return len(self.categories) > 0


class QueryClassifier:
    """
    Rule-based query classifier using embedding cosine similarity.

    Classifies queries into categories that already exist in the
    `knowledge_chunks` table. No LLM calls — purely deterministic.

    Attributes:
        THRESHOLD: Minimum average cosine similarity to consider a category relevant.
        AMBIGUITY_MARGIN: Minimum gap between top two categories to declare a clear winner.
    """

    THRESHOLD = 0.50
    AMBIGUITY_MARGIN = 0.05

    def __init__(
        self,
        threshold: float = None,
        ambiguity_margin: float = None,
    ):
        """
        Initialize the classifier.

        Args:
            threshold: Override default similarity threshold.
            ambiguity_margin: Override default ambiguity margin.
        """
        if threshold is not None:
            self.THRESHOLD = threshold
        if ambiguity_margin is not None:
            self.AMBIGUITY_MARGIN = ambiguity_margin

    def get_available_categories(self) -> List[str]:
        """
        Fetches all distinct categories currently stored in `knowledge_chunks`.

        Returns:
            List[str]: Sorted list of unique category names.
        """
        db, _ = _get_clients()
        response = (
            db.table("knowledge_chunks")
            .select("category")
            .execute()
        )
        categories = sorted(set(row["category"] for row in response.data))
        return categories

    def get_top_k_categories(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves the top K categories from the database for a given query,
        ranked by average cosine similarity.

        Args:
            query: User's query string.
            top_k: Number of top categories to return (default: 3).

        Returns:
            List[Dict[str, Any]]: List of dicts containing 'category' and 'similarity'.
        """
        db, _ = _get_clients()
        query_embedding = _embed(query)
        response = db.rpc("classify_query", {
            "query_embedding": query_embedding,
            "match_threshold": 0.0,
        }).execute()

        rankings = response.data or []
        return [
            {"category": r["category"], "similarity": round(r["avg_similarity"], 4)}
            for r in rankings[:top_k]
        ]

    def classify(self, query: str) -> ClassificationResult:
        """
        Classify a query into one or more existing categories.

        Classification rules (fully deterministic):
            1. If no category scores above THRESHOLD → no match
            2. If top score has a clear gap (>= AMBIGUITY_MARGIN) over #2 → single category
            3. If top two scores are within AMBIGUITY_MARGIN → return both (ambiguous)

        Args:
            query: The user's natural language query string.

        Returns:
            ClassificationResult: Contains matched categories, scores, and metadata.
        """
        db, _ = _get_clients()

        # Step 1: Embed the query
        query_embedding = _embed(query)

        # Step 2: Call the classify_query RPC
        response = db.rpc("classify_query", {
            "query_embedding": query_embedding,
            "match_threshold": self.THRESHOLD,
        }).execute()

        rankings = response.data  # [{"category": "X", "avg_similarity": 0.72}, ...]

        # Step 3: Apply deterministic rules
        if not rankings:
            return ClassificationResult()

        all_scores = [
            (r["category"], round(r["avg_similarity"], 4))
            for r in rankings
        ]

        top = rankings[0]
        second = rankings[1] if len(rankings) > 1 else None

        # Check for ambiguity
        if second and (top["avg_similarity"] - second["avg_similarity"]) < self.AMBIGUITY_MARGIN:
            return ClassificationResult(
                categories=[top["category"], second["category"]],
                top_score=round(top["avg_similarity"], 4),
                is_ambiguous=True,
                all_scores=all_scores,
            )

        # Clear winner
        return ClassificationResult(
            categories=[top["category"]],
            top_score=round(top["avg_similarity"], 4),
            is_ambiguous=False,
            all_scores=all_scores,
        )


# ---------------------------------------------------------------------------
# Standalone test — run with: python query/classifier.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Query Classifier — Rule-Based (No LLM)")
    print("=" * 60)

    classifier = QueryClassifier()

    # Show available categories
    categories = classifier.get_available_categories()
    print(f"\nAvailable categories in knowledge_chunks ({len(categories)}):")
    for cat in categories:
        print(f"  • {cat}")

    # Test queries spanning different expected categories
    test_queries = [
        "How do I request a payout?",
        "Why was my card declined?",
        "Can I get a refund on my account?",
        "What is the difference between SELECT and Growth evaluation?",
        "How do I track my order?",
        "What are the contract scaling rules for funded accounts?",
        "Where do I download my invoice?",
        "Does Growth evaluation have a consistency rule?",
        "What are the promo code troubleshooting steps?",
        "Hello how are you doing today?",
    ]

    print(f"\n{'—' * 60}")
    print(f"{'Query':<55} {'Category':<25} {'Score':<8} {'Ambig?'}")
    print(f"{'—' * 60}")

    for query in test_queries:
        result = classifier.classify(query)

        if result.matched:
            cat_str = ", ".join(result.categories)
            print(f"{query:<55} {cat_str:<25} {result.top_score:<8.4f} {result.is_ambiguous}")
        else:
            print(f"{query:<55} {'[NO MATCH]':<25} {'—':<8} —")

        # Print all scores for the last test (off-topic)
        if query == test_queries[-1] and result.all_scores:
            print(f"\n  All scores for off-topic query:")
            for cat, score in result.all_scores:
                print(f"    {cat}: {score:.4f}")
