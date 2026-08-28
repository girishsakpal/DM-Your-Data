from flask import Blueprint, jsonify, render_template
from app.db.connection import test_connection, get_schema_summary

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
