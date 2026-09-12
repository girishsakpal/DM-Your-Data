"""
Quick manual test for Phase 4: profiling + outlier detection.
Run scripts/run_profiling.py first to generate the cached report used by
the SQL generator's schema context.

Usage:
    python scripts/test_phase4.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.profiling.profiler import profile_table
from app.profiling.outliers import detect_all_outliers, detect_outliers


def print_profile(table: str):
    print(f"\n=== Profile: {table} ===")
    profile = profile_table(table)
    print(f"Row count: {profile['row_count']}")
    for col, stats in profile["columns"].items():
        print(f"\n  {col} ({stats.get('data_type')})")
        if "note" in stats:
            print(f"    {stats['note']}")
            continue
        print(f"    null_rate={stats['null_rate']}, distinct_count={stats['distinct_count']}")
        if "stats" in stats:
            print(f"    {stats['stats']}")


def print_outliers(table: str, method: str = "zscore"):
    print(f"\n=== Outliers in {table} (method={method}) ===")
    results = detect_all_outliers(table, method=method)
    if not results:
        print("  No numeric columns found.")
        return
    for col, result in results.items():
        print(f"\n  {col}: {result['outlier_count']} outlier(s)")
        for row in result["outliers"][:5]:
            print(f"    {dict(row)}")


if __name__ == "__main__":
    print_profile("product_reviews")
    print_outliers("product_reviews", method="zscore")
    print_outliers("product_reviews", method="iqr")
