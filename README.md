# Query Classifier & Category Retriever System

A vector-embedding based Query Category Classifier and Retriever built with Supabase Vector (pgvector) and Gemini Embeddings.

Classifies incoming questions into top matching categories stored in the database and retrieves top knowledge chunks without generative LLM calls.

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
│   ├── classifier.py        # Rule-based query category classifier
│   ├── retriever.py         # Category-filtered vector search
│   ├── setup_rpc.sql        # Supabase SQL functions for classification & search
│   ├── ask.py              # CLI entry point to classify & retrieve top categories
│   └── requirements.txt
├── .env
└── README.md
```

## Setup Instructions

### 1. Environment Configuration

Create a `.env` file in the root directory:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_role_key
GEMINI_API_KEY=your_gemini_api_key
```

### 2. Database Setup (Supabase / pgvector)

Run `query/setup_rpc.sql` in your Supabase SQL Editor to enable vector extensions and create the `classify_query` RPC function.

### 3. Python Setup

```bash
pip install -r query/requirements.txt
```

## Usage

### Query Top K Categories

```bash
python query/ask.py "How do I request a payout?"
```

Output:
```text
============================================================
QUERY CATEGORY CLASSIFIER
============================================================
Query: 'How do I request a payout?'
Top K: 3

Top K Categories from Database:
------------------------------------------------------------
Rank   Category                            Similarity Score
------------------------------------------------------------
1      Payouts                             0.6745
2      Trading & Payouts                   0.6696
3      KYC & Taxes                         0.6571
------------------------------------------------------------
```
