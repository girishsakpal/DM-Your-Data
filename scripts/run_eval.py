"""
Runs the full Phase 5 evaluation suite: SQL execution accuracy, routing
accuracy, and semantic/hybrid retrieval precision — against the labeled
dataset in tests/eval_dataset.json.

Saves a detailed per-case report to data/eval_runs/ and appends a summary
row to data/eval_history.csv, so you can track accuracy across model or
prompt changes over time rather than just seeing the latest snapshot.

Usage:
    python scripts/run_eval.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.eval.runner import run_full_eval, save_run


def print_report(eval_output: dict):
    summary = eval_output["summary"]
    results = eval_output["results"]

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Timestamp:            {summary['timestamp']}")
    print(f"SQL model:            {summary['sql_model']}")
    print(f"Router model:         {summary['router_model']}")
    print(f"Total cases:          {summary['total_cases']}")
    print(f"Routing accuracy:     {summary['routing_accuracy']}")
    print(f"SQL exec. accuracy:   {summary['sql_execution_accuracy']}")
    print(f"Semantic avg. prec.:  {summary['semantic_avg_precision']}")
    print(f"Hybrid avg. prec.:    {summary['hybrid_avg_precision']}")

    print("\n" + "=" * 60)
    print("PER-CASE RESULTS")
    print("=" * 60)
    for r in results:
        route_mark = "✓" if r["route_correct"] else "✗"
        if r["type"] == "sql":
            correct_mark = "✓" if r["sql_correct"] else "✗"
            print(f"[{r['id']}] route:{route_mark} sql:{correct_mark}  {r['question']}")
            if not r["sql_correct"]:
                print(f"    generated: {r['generated_sql']}")
                print(f"    expected:  {r.get('expected_sql')}")
        else:
            print(f"[{r['id']}] route:{route_mark} precision:{r['precision']} recall:{r['recall']}  {r['question']}")
            if r["precision"] < 1.0:
                print(f"    returned: {r['returned_ids']}  relevant: {r['relevant_ids']}")

    print("\nFailures are expected the first time you run this — use them to")
    print("decide whether to adjust prompts, thresholds, or the model itself.")


if __name__ == "__main__":
    print("Running evaluation suite (this makes real LLM + DB calls per case, may take a minute)...\n")
    eval_output = run_full_eval()
    print_report(eval_output)

    run_path = save_run(eval_output)
    print(f"\nFull report saved to: {run_path}")
    print("Summary appended to: data/eval_history.csv")
