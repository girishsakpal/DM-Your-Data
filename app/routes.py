from flask import Blueprint, jsonify, render_template, request
from app.db.connection import test_connection, get_schema_summary
from app.nlp.sql_executor import generate_and_execute
from app.embeddings.semantic_search import semantic_search
from app.nlp.query_router import classify, ROUTE_SQL, ROUTE_SEMANTIC, ROUTE_HYBRID
from app.nlp.hybrid import run_hybrid_query
from app.profiling.profiler import profile_database, profile_table
from app.profiling.outliers import detect_outliers, detect_all_outliers
from app.eval.runner import run_full_eval, save_run
from app.data_upload.ingest import replace_active_dataset, UploadError
from app.data_upload import registry

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/health")
def health():
    """Confirms Flask + DB are both alive. Hit this first after setup."""
    try:
        version = test_connection()
        return jsonify({"status": "ok", "postgres_version": version})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


@bp.route("/schema")
def schema():
    """Returns the introspected DB schema — this becomes LLM context in Phase 1."""
    try:
        return jsonify(get_schema_summary())
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


@bp.route("/query", methods=["POST"])
def query():
    """
    Phase 1 core endpoint: natural language question -> generated SQL -> results.

    Request body: {"question": "average price by category"}
    Response: {"success": true, "sql": "...", "columns": [...], "rows": [...], "row_count": N}
              or {"success": false, "sql": "...", "error": "..."} on failure after retries.
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"success": False, "error": "Missing 'question' in request body."}), 400

    result = generate_and_execute(question)
    status_code = 200 if result["success"] else 422
    return jsonify(result), status_code


@bp.route("/semantic-query", methods=["POST"])
def semantic_query():
    """
    Phase 2 endpoint: conceptual/fuzzy natural language query -> semantically
    similar rows via pgvector, independent of the SQL path.

    Request body: {"query": "customers who were frustrated", "top_k": 5}
    Response: {"query": "...", "results": [...], "result_count": N}
    """
    data = request.get_json(silent=True) or {}
    query_text = (data.get("query") or "").strip()
    top_k = data.get("top_k", 10)

    if not query_text:
        return jsonify({"success": False, "error": "Missing 'query' in request body."}), 400

    try:
        result = semantic_search(query_text, top_k=top_k)
        result["success"] = True
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/ask", methods=["POST"])
def ask():
    """
    Phase 3 endpoint: the single entry point a real UI should call. Classifies
    the question, then dispatches to SQL, semantic search, or the hybrid path.

    Request body: {"question": "frustrated customers in the West region"}
    Response includes a "route" and "route_method" field for transparency —
    useful for debugging and for the Phase 5 eval harness later.
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"success": False, "error": "Missing 'question' in request body."}), 400

    routing = classify(question)
    route = routing["route"]

    try:
        if route == ROUTE_SQL:
            result = generate_and_execute(question)
        elif route == ROUTE_SEMANTIC:
            result = semantic_search(question)
            result["success"] = True
        else:  # hybrid
            result = run_hybrid_query(question)
            result["success"] = True

        result["route"] = route
        result["route_method"] = routing["method"]
        status_code = 200 if result.get("success", True) else 422
        return jsonify(result), status_code

    except Exception as e:
        return jsonify({
            "success": False,
            "route": route,
            "route_method": routing["method"],
            "error": str(e),
        }), 500


@bp.route("/profile")
def profile():
    """
    Phase 4: live data profiling. Optional ?table=<name> to profile just one
    table instead of the whole database (profiling every table can be slow
    on larger datasets — this endpoint always runs live, unlike the cached
    report schema_context.py reads for SQL generation).
    """
    table = request.args.get("table")
    try:
        if table:
            return jsonify(profile_table(table)), 200
        return jsonify(profile_database()), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/outliers")
def outliers():
    """
    Phase 4: statistical outlier detection. Query params:
      table (required), column (optional — omit to check all numeric columns),
      method ("zscore" default, or "iqr")

    Examples:
      /outliers?table=product_reviews&column=price
      /outliers?table=product_reviews&method=iqr
    """
    table = request.args.get("table")
    column = request.args.get("column")
    method = request.args.get("method", "zscore")

    if not table:
        return jsonify({"success": False, "error": "Missing required 'table' query param."}), 400

    try:
        if column:
            return jsonify(detect_outliers(table, column, method=method)), 200
        return jsonify(detect_all_outliers(table, method=method)), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/eval", methods=["POST"])
def eval_endpoint():
    """
    Phase 5: runs the full evaluation suite against tests/eval_dataset.json
    and returns the summary + per-case results. Also saves the run to
    data/eval_runs/ and appends to data/eval_history.csv, same as
    scripts/run_eval.py — this just makes it triggerable over HTTP too.

    Warning: this makes a real LLM call (and DB query) per test case, so it
    is not fast — expect it to take a while depending on your dataset size.
    """
    try:
        eval_output = run_full_eval()
        run_path = save_run(eval_output)
        eval_output["saved_to"] = run_path
        return jsonify(eval_output), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/upload", methods=["POST"])
def upload():
    """
    Accepts a CSV file (multipart/form-data, field name 'file') and makes it
    the active dataset for the whole app — SQL generation, semantic search,
    and hybrid queries all start targeting it immediately. This REPLACES any
    previously uploaded dataset (single active dataset by design); the
    original seed data is untouched and can be restored via /reset-dataset.
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided (expected form field 'file')."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"success": False, "error": "No file selected."}), 400
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"success": False, "error": "Only .csv files are supported right now."}), 400

    try:
        file_bytes = file.read()
        metadata = replace_active_dataset(file_bytes, file.filename)
        return jsonify({"success": True, **metadata}), 200
    except UploadError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Unexpected error during upload: {e}"}), 500


@bp.route("/active-dataset")
def active_dataset():
    """Returns metadata about whichever dataset is currently active — an upload, or the seed data if none."""
    ds = registry.get_active_dataset()
    if ds:
        return jsonify({"is_upload": True, **ds}), 200
    return jsonify({
        "is_upload": False,
        "table": registry.DEFAULT_TABLE,
        "text_column": registry.DEFAULT_TEXT_COLUMN,
        "id_column": registry.DEFAULT_ID_COLUMN,
        "columns": None,  # caller can fall back to /schema for the seed table's columns
    }), 200


@bp.route("/reset-dataset", methods=["POST"])
def reset_dataset():
    """Reverts to the seed product_reviews data. Does NOT drop the uploaded table — just stops using it."""
    registry.clear_active_dataset()
    return jsonify({"success": True, "message": "Reverted to seed data.", "table": registry.DEFAULT_TABLE}), 200
