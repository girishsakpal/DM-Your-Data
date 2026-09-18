"""
Run this after `docker compose up -d` and `ollama pull <model>` to confirm
your Phase 0 setup is working before moving to Phase 1.

Usage:
    python scripts/test_connection.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.connection import test_connection, get_schema_summary


def check_postgres():
    print("Checking PostgreSQL connection...")
    try:
        version = test_connection()
        print(f"  OK - {version}")
        return True
    except Exception as e:
        print(f"  FAILED - {e}")
        print("  -> Is docker compose running? Try: docker compose up -d")
        return False


def check_schema():
    print("Checking schema introspection...")
    try:
        schema = get_schema_summary()
        if not schema:
            print("  WARNING - no tables found. Did init.sql run? (only runs on first container start)")
            return False
        for table, cols in schema.items():
            print(f"  {table}: {len(cols)} columns")
        return True
    except Exception as e:
        print(f"  FAILED - {e}")
        return False


def check_ollama():
    print("Checking Ollama connection...")
    try:
        import ollama
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = ollama.Client(host=host)
        models = client.list()
        names = [m["model"] for m in models.get("models", [])]
        if names:
            print(f"  OK - available models: {', '.join(names)}")
        else:
            print("  Ollama is reachable but no models are pulled yet.")
            print("  -> Try: ollama pull llama3.2:3b")
        return True
    except Exception as e:
        print(f"  FAILED - {e}")
        print("  -> Is Ollama running? Try: ollama serve")
        return False


if __name__ == "__main__":
    results = [check_postgres(), check_schema(), check_ollama()]
    print()
    if all(results):
        print("All checks passed. Ready for Phase 1.")
    else:
        print("Some checks failed - fix the issues above before moving on.")
