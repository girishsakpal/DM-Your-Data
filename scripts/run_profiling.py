"""
Runs automatic data profiling across all tables and saves the report to
data/profile_report.json - this is what schema_context.py reads to enrich
LLM prompts with column stats (Phase 4 -> Phase 1 integration).

Also runs outlier detection on all numeric columns and saves to
data/outlier_report.json.

Re-run this whenever your data changes meaningfully - it's not automatic
yet (see README Phase 4 known limitations).

Usage:
    python scripts/run_profiling.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.profiling.profiler import profile_database
from app.profiling.outliers import detect_all_outliers
from app.db.connection import get_schema_summary

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PROFILE_PATH = os.path.join(DATA_DIR, "profile_report.json")
OUTLIER_PATH = os.path.join(DATA_DIR, "outlier_report.json")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Profiling all tables...")
    profile = profile_database()
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2, default=str)
    print(f"  Saved to {PROFILE_PATH}")

    for table, table_profile in profile.items():
        print(f"  {table}: {table_profile['row_count']} rows, {len(table_profile['columns'])} columns")

    print("\nRunning outlier detection on numeric columns...")
    schema = get_schema_summary()
    outlier_report = {}
    for table in schema:
        if table == "review_embeddings":
            continue
        table_outliers = detect_all_outliers(table)
        if table_outliers:
            outlier_report[table] = table_outliers
            for col, result in table_outliers.items():
                if result["outlier_count"] > 0:
                    print(f"  {table}.{col}: {result['outlier_count']} outlier(s) flagged (z-score)")

    with open(OUTLIER_PATH, "w") as f:
        json.dump(outlier_report, f, indent=2, default=str)
    print(f"  Saved to {OUTLIER_PATH}")

    print("\nDone. Re-run this script after adding new data to keep the profile current.")


if __name__ == "__main__":
    main()
