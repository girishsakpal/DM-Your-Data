"""
Quick manual test for the Phase 2 semantic search pipeline.
Run scripts/ingest_embeddings.py first, or this will return nothing.

Usage:
    python scripts/test_phase2.py
    python scripts/test_phase2.py "your own query here"
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embeddings.semantic_search import semantic_search

DEFAULT_QUERIES = [
    "customers who were frustrated with their purchase",
    "products with great battery life",
    "items that arrived damaged",
]


def run_query(query: str):
    print(f"\nQuery: {query}")
    result = semantic_search(query, top_k=5)

    if not result["results"]:
        print(f"  No results above the relevance threshold "
              f"({result['raw_result_count']} raw matches were filtered out).")
        return

    for row in result["results"]:
        print(f"  [dist={row['distance']:.3f}] {row['product_name']} ({row['rating']}★): {row['review_text']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_query(" ".join(sys.argv[1:]))
    else:
        print("Running default test queries...")
        for q in DEFAULT_QUERIES:
            run_query(q)
