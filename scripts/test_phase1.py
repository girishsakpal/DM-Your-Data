"""
Quick manual test for the Phase 1 text-to-SQL pipeline.
Run this after confirming scripts/test_connection.py passes.

Usage:
    python scripts/test_phase1.py
    python scripts/test_phase1.py "your own question here"
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.nlp.sql_executor import generate_and_execute

DEFAULT_QUESTIONS = [
    "What is the average price by category?",
    "Show me all reviews with a rating below 3",
    "How many reviews came from the West region?",
    "What are the 3 most expensive products?",
]


def run_question(question: str):
    print(f"\nQ: {question}")
    result = generate_and_execute(question)

    if result["success"]:
        print(f"SQL ({result['attempts']} attempt(s)): {result['sql']}")
        print(f"Columns: {result['columns']}")
        for row in result["rows"][:10]:
            print(f"  {row}")
        if result["row_count"] > 10:
            print(f"  ... ({result['row_count']} rows total)")
    else:
        print(f"FAILED after {result['attempts']} attempt(s)")
        print(f"Last SQL tried: {result['sql']}")
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_question(" ".join(sys.argv[1:]))
    else:
        print("Running default test questions...")
        for q in DEFAULT_QUESTIONS:
            run_question(q)
