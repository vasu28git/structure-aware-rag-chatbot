"""
Top-K Category Retriever

Retrieves and displays the top K matching categories from the Supabase database
for any given query using vector embedding similarity.
"""

import sys
from typing import List, Dict, Any
from classifier import QueryClassifier


def get_top_k_categories(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Calls the classifier module to fetch top K matching categories for a query.

    Args:
        query: Input question or search phrase.
        top_k: Number of top categories to return (default: 3).

    Returns:
        List of dicts with 'category' and 'similarity'.
    """
    classifier = QueryClassifier()
    return classifier.get_top_k_categories(query, top_k=top_k)


def main():
    """
    Command Line Interface to display top K categories for a given query.
    """
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "How do I request a payout?"

    top_k = 3

    print("=" * 60)
    print("QUERY CATEGORY CLASSIFIER")
    print("=" * 60)
    print(f"Query: '{query}'")
    print(f"Top K: {top_k}\n")

    categories = get_top_k_categories(query, top_k=top_k)

    print("Top K Categories from Database:")
    print("-" * 60)
    print(f"{'Rank':<6} {'Category':<35} {'Similarity Score':<15}")
    print("-" * 60)

    for rank, item in enumerate(categories, 1):
        print(f"{rank:<6} {item['category']:<35} {item['similarity']:.4f}")

    print("-" * 60)


if __name__ == "__main__":
    main()
