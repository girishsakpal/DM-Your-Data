"""
Quick manual test for the Phase 3 query router. Tests classification and
end-to-end dispatch across all three route types.

Usage:
    python scripts/test_phase3.py
    python scripts/test_phase3.py "your own question here"
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.nlp.query_router import classify
from app.nlp.sql_executor import generate_and_execute
from app.embeddings.semantic_search import semantic_search
from app.nlp.hybrid import run_hybrid_query

# One example per route, plus a couple of genuinely ambiguous ones to see
# how the heuristic/LLM split behaves.
DEFAULT_QUESTIONS = [
    "What is the average price by category?",                          # sql
    "Find reviews expressing frustration with shipping",                # semantic
    "Frustrated customers in the West region",                          # hybrid
    "Show me reviews about battery life for products under 1500",       # hybrid
    "What's good?",                                                     # ambiguous
]


def run_question(question: str):
    print(f"\nQ: {question}")
    routing = classify(question)
    print(f"  -> route: {routing['route']} (via {routing['method']})")

    if routing["route"] == "sql":
        result = generate_and_execute(question)
        if result["success"]:
            print(f"  SQL: {result['sql']}")
            print(f"  {result['row_count']} row(s) returned")
        else:
            print(f"  FAILED: {result['error']}")

    elif routing["route"] == "semantic":
        result = semantic_search(question, top_k=5)
        print(f"  {result['result_count']} result(s)")
        for r in result["results"][:3]:
            print(f"    [dist={r['distance']:.3f}] {r['review_text'][:80]}")

    else:  # hybrid
        result = run_hybrid_query(question, top_k=5)
        print(f"  filter_sql: {result['filter_sql']}")
        print(f"  semantic_query: {result['semantic_query']}")
        if result["filter_error"]:
            print(f"  filter_error (fell back to unfiltered semantic): {result['filter_error']}")
        print(f"  {result['result_count']} result(s)")
        for r in result["results"][:3]:
            print(f"    [dist={r['distance']:.3f}] {r['review_text'][:80]}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_question(" ".join(sys.argv[1:]))
    else:
        print("Running default test questions across all three routes...")
        for q in DEFAULT_QUESTIONS:
            run_question(q)
