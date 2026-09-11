from flask import Blueprint, jsonify, render_template, request
from app.db.connection import test_connection, get_schema_summary
from app.nlp.sql_executor import generate_and_execute
from app.embeddings.semantic_search import semantic_search
from app.nlp.query_router import classify, ROUTE_SQL, ROUTE_SEMANTIC, ROUTE_HYBRID
from app.nlp.hybrid import run_hybrid_query

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
