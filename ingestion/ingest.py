"""
Ingestion Pipeline (Phase A) for E-Commerce & Customer Support RAG System

This script reads products, FAQs, and policy datasets, processes them using structure-aware
chunking (`chunker.py`), generates text embeddings using Gemini (`models/text-embedding-004`),
and inserts the resulting vectorized data into Supabase `knowledge_chunks` table.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from supabase import create_client, Client
from chunker import chunk_text

# Load environment variables from .env located in the parent directory
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Global client placeholders
supabase_client: Client = None
ai_client: genai.Client = None


def init_clients() -> tuple[Client, genai.Client]:
    """
    Initializes and returns Supabase and Gemini AI clients using environment credentials.

    Returns:
        tuple[Client, genai.Client]: Initialized Supabase and Gemini client instances.
    """
    global supabase_client, ai_client

    if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
        raise ValueError(
            "Missing environment credentials. Ensure SUPABASE_URL, SUPABASE_KEY, "
            "and GEMINI_API_KEY are configured in the .env file."
        )

    if supabase_client is None:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    if ai_client is None:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)

    return supabase_client, ai_client


def embed(text: str) -> List[float]:
    """
    Generates a 768-dimensional vector embedding for the input text using Gemini's gemini-embedding-001 model.

    Args:
        text (str): Content string to embed.

    Returns:
        List[float]: 768-dimensional float embedding vector.
    """
    _, client = init_clients()
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config={"output_dimensionality": 768},
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"Error generating embedding with Gemini API: {e}")
        raise


def insert_chunk(content: str, type_: str, category: str, source_id: str) -> bool:
    """
    Generates embedding for a chunk and stores it in Supabase `knowledge_chunks` table.

    Args:
        content (str): Chunk text content.
        type_ (str): Type tag ("product", "faq", or "policy").
        category (str): Category classification string.
        source_id (str): Original document ID.

    Returns:
        bool: True if insert succeeded, False otherwise.
    """
    db, _ = init_clients()
    try:
        embedding_vector = embed(content)
        data = {
            "content": content,
            "type": type_,
            "category": category,
            "source_id": source_id,
            "embedding": embedding_vector
        }
        db.table("knowledge_chunks").insert(data).execute()
        print(f"Inserted chunk from {source_id} ({type_})")
        return True
    except Exception as e:
        print(f"Failed to insert chunk from {source_id} ({type_}): {e}")
        return False


def load_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Helper to safely read JSON datasets.

    Args:
        file_path (Path): Path to target JSON file.

    Returns:
        List[Dict[str, Any]]: List of objects loaded from JSON, or empty list if missing/invalid.
    """
    if not file_path.exists():
        print(f"Warning: File not found at {file_path}. Skipping.")
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list) or len(data) == 0:
                print(f"Warning: File at {file_path} is empty or invalid. Skipping.")
                return []
            return data
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}. Skipping.")
        return []


def ingest_products(data_dir: Path) -> int:
    """
    Loads products.json, chunks product text blocks, embeds, and inserts into Supabase.

    Args:
        data_dir (Path): Directory containing data JSON files.

    Returns:
        int: Number of successfully inserted product chunks.
    """
    products = load_json_file(data_dir / "products.json")
    successful_inserts = 0

    for item in products:
        name = item.get("name", "")
        description = item.get("description", "")
        price = item.get("price", "")
        stock = item.get("stock", "")
        category = item.get("category", "General")
        source_id = item.get("id", "unknown_product")

        text_block = f"{name}. {description} Price: ₹{price}. In stock: {stock}."
        chunks = chunk_text(text_block)

        for chunk in chunks:
            if insert_chunk(content=chunk, type_="product", category=category, source_id=source_id):
                successful_inserts += 1

    return successful_inserts


def ingest_faqs(data_dir: Path) -> int:
    """
    Loads faqs.json, chunks Q&A text blocks, embeds, and inserts into Supabase.

    Args:
        data_dir (Path): Directory containing data JSON files.

    Returns:
        int: Number of successfully inserted FAQ chunks.
    """
    faqs = load_json_file(data_dir / "faqs.json")
    successful_inserts = 0

    for item in faqs:
        question = item.get("question", "")
        answer = item.get("answer", "")
        category = item.get("category", "General")
        source_id = item.get("id", "unknown_faq")

        text_block = f"Q: {question} A: {answer}"
        chunks = chunk_text(text_block)

        for chunk in chunks:
            if insert_chunk(content=chunk, type_="faq", category=category, source_id=source_id):
                successful_inserts += 1

    return successful_inserts


def ingest_policies(data_dir: Path) -> int:
    """
    Loads policies.json, chunks policy text blocks, embeds, and inserts into Supabase.

    Args:
        data_dir (Path): Directory containing data JSON files.

    Returns:
        int: Number of successfully inserted policy chunks.
    """
    policies = load_json_file(data_dir / "policies.json")
    successful_inserts = 0

    for item in policies:
        title = item.get("title", "")
        content = item.get("content", "")
        category = item.get("category", "General")
        source_id = item.get("id", "unknown_policy")

        text_block = f"{title}. {content}"
        chunks = chunk_text(text_block)

        for chunk in chunks:
            if insert_chunk(content=chunk, type_="policy", category=category, source_id=source_id):
                successful_inserts += 1

    return successful_inserts


def main():
    """
    Main execution pipeline: sequentially ingests products, FAQs, and policies.
    """
    print("=== Starting Knowledge Base Ingestion Pipeline (Phase A) ===\n")
    data_dir = Path(__file__).resolve().parent.parent / "data"

    total_products = ingest_products(data_dir)
    total_faqs = ingest_faqs(data_dir)
    total_policies = ingest_policies(data_dir)

    total_chunks = total_products + total_faqs + total_policies
    print(
        f"\n=== Ingestion Complete ===\n"
        f"Total chunks successfully inserted: {total_chunks} "
        f"(Products: {total_products}, FAQs: {total_faqs}, Policies: {total_policies})"
    )


if __name__ == "__main__":
    main()
