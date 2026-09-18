import os
import json
import csv
from datetime import datetime, timezone

from app.nlp.sql_executor import generate_and_execute, execute_sql
from app.nlp.query_router import classify
from app.embeddings.semantic_search import semantic_search
from app.nlp.hybrid import run_hybrid_query
from app.eval.metrics import result_sets_match, precision_at_k

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(BASE_DIR, "tests", "eval_dataset.json")
RUNS_DIR = os.path.join(BASE_DIR, "data", "eval_runs")
HISTORY_PATH = os.path.join(BASE_DIR, "data", "eval_history.csv")


def load_dataset() -> dict:
    with open(DATASET_PATH, "r") as f:
        return json.load(f)


def eval_sql_case(case: dict) -> dict:
    """
    Execution accuracy: runs the LLM-generated SQL and the hand-written
    expected SQL independently, then compares result sets rather than SQL
    text - a correct query can be phrased many different ways syntactically.
    Also checks routing: did the router send this question down the SQL path?
    """
    question = case["question"]
    routing = classify(question)
    route_correct = routing["route"] == case["expected_route"]

    generated = generate_and_execute(question)
    if not generated["success"]:
        return {
            "id": case["id"], "question": question, "type": "sql",
            "route_correct": route_correct, "actual_route": routing["route"],
            "sql_correct": False, "generated_sql": generated.get("sql"),
            "error": generated.get("error"),
        }

    try:
        expected = execute_sql(case["expected_sql"])
        sql_correct = result_sets_match(generated["rows"], expected["rows"])
    except Exception as e:
        sql_correct = False
        expected = {"rows": [], "error": str(e)}

    return {
        "id": case["id"], "question": question, "type": "sql",
        "route_correct": route_correct, "actual_route": routing["route"],
        "sql_correct": sql_correct,
        "generated_sql": generated["sql"],
        "expected_sql": case["expected_sql"],
        "generated_rows": generated["rows"],
        "expected_rows": expected["rows"],
    }


def eval_semantic_case(case: dict, top_k: int = 5) -> dict:
    """Retrieval evaluation: precision/recall of returned row IDs against hand-labeled relevant IDs."""
    question = case["question"]
    routing = classify(question)
    route_correct = routing["route"] == case["expected_route"]

    result = semantic_search(question, top_k=top_k)
    returned_ids = [r["id"] for r in result["results"]]
    metrics = precision_at_k(returned_ids, case["relevant_ids"], k=top_k)

    return {
        "id": case["id"], "question": question, "type": "semantic",
        "route_correct": route_correct, "actual_route": routing["route"],
        "precision": metrics["precision"], "recall": metrics["recall"],
        "returned_ids": metrics["returned"], "relevant_ids": case["relevant_ids"],
    }


def eval_hybrid_case(case: dict, top_k: int = 5) -> dict:
    """Same retrieval metrics as semantic, but through the full decompose -> filter -> rank pipeline."""
    question = case["question"]
    routing = classify(question)
    route_correct = routing["route"] == case["expected_route"]

    result = run_hybrid_query(question, top_k=top_k)
    returned_ids = [r["id"] for r in result["results"]]
    metrics = precision_at_k(returned_ids, case["relevant_ids"], k=top_k)

    return {
        "id": case["id"], "question": question, "type": "hybrid",
        "route_correct": route_correct, "actual_route": routing["route"],
        "precision": metrics["precision"], "recall": metrics["recall"],
        "returned_ids": metrics["returned"], "relevant_ids": case["relevant_ids"],
        "filter_sql": result.get("filter_sql"),
    }


def run_full_eval() -> dict:
    """Runs every case in the dataset and computes aggregate metrics."""
    dataset = load_dataset()

    sql_results = [eval_sql_case(c) for c in dataset.get("sql_cases", [])]
    semantic_results = [eval_semantic_case(c) for c in dataset.get("semantic_cases", [])]
    hybrid_results = [eval_hybrid_case(c) for c in dataset.get("hybrid_cases", [])]

    all_results = sql_results + semantic_results + hybrid_results
    routing_accuracy = (
        sum(1 for r in all_results if r["route_correct"]) / len(all_results)
        if all_results else 0.0
    )
    sql_accuracy = (
        sum(1 for r in sql_results if r["sql_correct"]) / len(sql_results)
        if sql_results else None
    )
    semantic_avg_precision = (
        sum(r["precision"] for r in semantic_results) / len(semantic_results)
        if semantic_results else None
    )
    hybrid_avg_precision = (
        sum(r["precision"] for r in hybrid_results) / len(hybrid_results)
        if hybrid_results else None
    )

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sql_model": os.getenv("SQL_MODEL"),
        "router_model": os.getenv("ROUTER_MODEL"),
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
        "total_cases": len(all_results),
        "routing_accuracy": round(routing_accuracy, 3),
        "sql_execution_accuracy": round(sql_accuracy, 3) if sql_accuracy is not None else None,
        "semantic_avg_precision": round(semantic_avg_precision, 3) if semantic_avg_precision is not None else None,
        "hybrid_avg_precision": round(hybrid_avg_precision, 3) if hybrid_avg_precision is not None else None,
    }

    return {"summary": summary, "results": all_results}


def save_run(eval_output: dict) -> str:
    """Saves the full per-case results to a timestamped file, and appends the summary row to a running CSV history."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    timestamp_slug = eval_output["summary"]["timestamp"].replace(":", "-")
    run_path = os.path.join(RUNS_DIR, f"eval_{timestamp_slug}.json")

    with open(run_path, "w") as f:
        json.dump(eval_output, f, indent=2, default=str)

    _append_history(eval_output["summary"])
    return run_path


def _append_history(summary: dict):
    """
    Appends one row per run to data/eval_history.csv - this is what lets you
    track accuracy over time as you change prompts, models, or thresholds,
    instead of only ever seeing the latest run's numbers.
    """
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    file_exists = os.path.exists(HISTORY_PATH)

    with open(HISTORY_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary)
