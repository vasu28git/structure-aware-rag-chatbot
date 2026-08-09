# Multi-Source E-Commerce & Support RAG System

A Retrieval-Augmented Generation (RAG) system built with Supabase Vector (pgvector) and Google Gemini API to index and query products, FAQs, and store policies.

## Project Structure

```text
.
├── data/
│   ├── products.json
│   ├── faqs.json
│   └── policies.json
├── ingestion/
│   ├── chunker.py
│   ├── ingest.py
│   └── requirements.txt
├── query/
│   ├── retriever.py
│   ├── generate.py
│   ├── ask.py
│   └── requirements.txt
├── .env
└── README.md
```

## Setup Instructions

### 1. Environment Configuration

Create a `.env` file in the root directory with the following variables:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_key
GEMINI_API_KEY=your_gemini_api_key
```

### 2. Database Setup (Supabase / pgvector)

Run the following SQL script in your Supabase SQL Editor to enable vector extensions and create the `documents` table with an HNSW index:

```sql
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding VECTOR(768)
);

-- Create HNSW index for fast vector cosine similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx 
ON documents USING hnsw (embedding vector_cosine_ops);
```

### 3. Python Setup

Install dependencies for ingestion and query modules:

```bash
# Ingestion dependencies
pip install -r ingestion/requirements.txt

# Query dependencies
pip install -r query/requirements.txt
```

## Usage

### Ingest Data
Parse JSON datasets (`products.json`, `faqs.json`, `policies.json`), generate Gemini embeddings (`text-embedding-004`), and insert chunks into Supabase pgvector:

```bash
python ingestion/ingest.py
```

### Query System
Ask a natural language question via command line:

```bash
python query/ask.py "What is the return policy for electronics?"
```
