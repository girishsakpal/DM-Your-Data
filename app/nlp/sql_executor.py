import re
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from app.db.connection import get_engine

# Statement types we refuse to run, even if somehow generated despite the prompt rules.
# This is a defense-in-depth check, not the only safeguard — see validate_sql().
FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "EXECUTE", "CALL", "COPY",
]

MAX_ROWS = 500          # hard cap on rows returned to the UI
STATEMENT_TIMEOUT_MS = 5000  # kill runaway queries (e.g. accidental cross join)


class UnsafeSQLError(Exception):
    pass


def validate_sql(sql: str) -> None:
    """Raises UnsafeSQLError if the SQL isn't a plain SELECT, or contains forbidden keywords."""
    stripped = sql.strip().rstrip(";").strip()

    if not stripped:
        raise UnsafeSQLError("Generated SQL is empty.")

    if not re.match(r"^\s*(SELECT|WITH)\b", stripped, re.IGNORECASE):
        raise UnsafeSQLError("Only SELECT statements are allowed.")

    # Reject multiple statements (e.g. "SELECT 1; DROP TABLE x;") — the split in
    # extract_sql() should already prevent this, but check again here since this
    # function may be called on SQL from other sources later (e.g. user-edited SQL).
    if ";" in stripped:
        raise UnsafeSQLError("Multiple statements are not allowed.")

    upper_sql = stripped.upper()
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper_sql):
            raise UnsafeSQLError(f"Forbidden keyword detected: {keyword}")


def enforce_limit(sql: str, max_rows: int = MAX_ROWS) -> str:
    """Appends a LIMIT if the query doesn't already have one, so a bad query can't return everything."""
    if re.search(r"\bLIMIT\s+\d+\b", sql, re.IGNORECASE):
        return sql
    return f"{sql.rstrip(';').strip()} LIMIT {max_rows};"


def execute_sql(sql: str) -> dict:
    """
    Validates and executes SQL against a read path only.
    Returns {"columns": [...], "rows": [[...], ...], "row_count": int}
    Raises UnsafeSQLError or SQLAlchemyError (caller should catch and surface to the user/retry loop).
    """
    validate_sql(sql)
    safe_sql = enforce_limit(sql)

    engine = get_engine()
    with engine.connect() as conn:
        # Per-connection statement timeout so a pathological query can't hang the request.
        conn.execute(text(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}"))
        result = conn.execute(text(safe_sql))
        columns = list(result.keys())
        rows = [list(row) for row in result.fetchall()]

    return {"columns": columns, "rows": rows, "row_count": len(rows), "sql_executed": safe_sql}


def generate_and_execute(question: str, max_retries: int = 1):
    """
    Full pipeline: generate SQL, validate, execute — with one retry that feeds
    the error back to the LLM if the first attempt fails. Returns a dict with
    either results or an error, plus the SQL for transparency/debugging.
    """
    from app.nlp.sql_generator import generate_sql  # local import avoids circular import

    attempt = 0
    previous_error, previous_sql = None, None

    while attempt <= max_retries:
        gen = generate_sql(question, previous_error=previous_error, previous_sql=previous_sql)
        sql = gen["sql"]

        try:
            result = execute_sql(sql)
            return {
                "success": True,
                "question": question,
                "sql": sql,
                "columns": result["columns"],
                "rows": result["rows"],
                "row_count": result["row_count"],
                "attempts": attempt + 1,
            }
        except (UnsafeSQLError, SQLAlchemyError) as e:
            previous_error = str(e)
            previous_sql = sql
            attempt += 1

    return {
        "success": False,
        "question": question,
        "sql": previous_sql,
        "error": previous_error,
        "attempts": attempt,
    }
