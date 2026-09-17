import os
import json
import re
from app.llm.ollama_client import generate
from app.nlp.schema_context import build_schema_context
from app.nlp.sql_executor import validate_sql, UnsafeSQLError
from app.db.connection import get_engine
from app.embeddings.semantic_search import semantic_search
from app.data_upload import registry
from sqlalchemy import text as sql_text
from sqlalchemy.exc import SQLAlchemyError

DECOMPOSE_SYSTEM_PROMPT = """You split a natural language question into two parts:

1. "filter_sql": a PostgreSQL WHERE-clause condition (just the condition, no \
"WHERE" keyword) capturing any structured filters — region, price, rating, \
date range, category. Use only columns from the schema given. If there is no \
structured filter, use null.
2. "semantic_query": the conceptual/fuzzy part of the question, rewritten as a \
short phrase suitable for semantic similarity search against review text. If \
there is no conceptual part, use null.

Respond with ONLY a JSON object like:
{"filter_sql": "region = 'West'", "semantic_query": "frustrated with shipping"}

No explanation, no markdown fences — just the raw JSON object."""


def decompose_query(question: str) -> dict:
    """Returns {"filter_sql": str|None, "semantic_query": str|None}"""
    model = os.getenv("ROUTER_MODEL", "llama3.2:3b")
    schema_context = build_schema_context(sample_rows=1)
    prompt = f"Schema:\n{schema_context}\n\nQuestion: {question}"

    raw = generate(model=model, prompt=prompt, system=DECOMPOSE_SYSTEM_PROMPT)

    # Strip markdown fences if the model adds them despite instructions
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    json_str = match.group(0) if match else raw.strip()

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # If decomposition fails, treat the whole question as semantic-only —
        # a safer fallback than crashing the request.
        return {"filter_sql": None, "semantic_query": question}

    return {
        "filter_sql": parsed.get("filter_sql") or None,
        "semantic_query": parsed.get("semantic_query") or question,
    }


def get_candidate_ids(filter_sql: str, table: str = None) -> list[int]:
    """
    Runs the LLM-generated WHERE condition to get matching row IDs.
    Validated the same way as the main SQL path — no arbitrary SQL execution.
    table defaults to whatever dataset is currently active.
    """
    table = table or registry.active_table()
    id_column = registry.active_id_column()
    full_sql = f'SELECT "{id_column}" FROM "{table}" WHERE {filter_sql}'
    validate_sql(full_sql)  # reuses Phase 1's SELECT-only / forbidden-keyword checks

    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(sql_text("SET statement_timeout = 5000"))
        rows = conn.execute(sql_text(full_sql)).fetchall()

    return [r[0] for r in rows]


def run_hybrid_query(question: str, top_k: int = 10) -> dict:
    """
    Full hybrid pipeline: decompose -> SQL pre-filter -> semantic rank within
    that subset. Falls back gracefully to pure semantic search if the SQL
    filter step fails or produces no usable condition.
    """
    decomposed = decompose_query(question)
    filter_sql = decomposed["filter_sql"]
    semantic_q = decomposed["semantic_query"]

    candidate_ids = None
    filter_error = None

    if filter_sql:
        try:
            candidate_ids = get_candidate_ids(filter_sql)
        except (UnsafeSQLError, SQLAlchemyError) as e:
            # Don't fail the whole request — fall back to unfiltered semantic search
            # and surface the issue for transparency instead of hiding it.
            filter_error = str(e)
            candidate_ids = None

    result = semantic_search(semantic_q, top_k=top_k, candidate_ids=candidate_ids)

    return {
        "question": question,
        "filter_sql": filter_sql,
        "filter_error": filter_error,
        "semantic_query": semantic_q,
        "results": result["results"],
        "result_count": result["result_count"],
    }
