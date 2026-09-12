from sqlalchemy import text
from app.db.connection import get_engine, get_schema_summary

# Postgres type name -> our bucket, so we know which stats make sense per column.
NUMERIC_TYPES = {"integer", "bigint", "smallint", "numeric", "real", "double precision"}
TEXT_TYPES = {"text", "character varying", "character", "varchar"}
DATE_TYPES = {"date", "timestamp", "timestamp without time zone", "timestamp with time zone"}

# Columns we can't meaningfully profile with simple SQL aggregates (pgvector's
# vector type doesn't support MIN/MAX/AVG the way numeric columns do).
SKIP_TYPES = {"vector", "USER-DEFINED"}


def _profile_numeric_column(conn, table: str, column: str) -> dict:
    result = conn.execute(text(f"""
        SELECT
            MIN("{column}") AS min_val,
            MAX("{column}") AS max_val,
            AVG("{column}"::numeric) AS mean_val,
            STDDEV("{column}"::numeric) AS stddev_val
        FROM "{table}"
        WHERE "{column}" IS NOT NULL
    """)).mappings().first()

    return {
        "min": float(result["min_val"]) if result["min_val"] is not None else None,
        "max": float(result["max_val"]) if result["max_val"] is not None else None,
        "mean": round(float(result["mean_val"]), 3) if result["mean_val"] is not None else None,
        "stddev": round(float(result["stddev_val"]), 3) if result["stddev_val"] is not None else None,
    }


def _profile_text_column(conn, table: str, column: str, top_n: int = 5) -> dict:
    avg_len = conn.execute(text(f"""
        SELECT AVG(LENGTH("{column}")) AS avg_len
        FROM "{table}" WHERE "{column}" IS NOT NULL
    """)).scalar()

    top_values = conn.execute(text(f"""
        SELECT "{column}" AS val, COUNT(*) AS cnt
        FROM "{table}"
        WHERE "{column}" IS NOT NULL
        GROUP BY "{column}"
        ORDER BY cnt DESC
        LIMIT :n
    """), {"n": top_n}).mappings().all()

    return {
        "avg_length": round(float(avg_len), 1) if avg_len is not None else None,
        "top_values": [{"value": r["val"], "count": r["cnt"]} for r in top_values],
    }


def profile_column(conn, table: str, column: str, data_type: str) -> dict:
    total = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()
    non_null = conn.execute(text(f'SELECT COUNT("{column}") FROM "{table}"')).scalar()
    distinct = conn.execute(text(f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"')).scalar()

    profile = {
        "data_type": data_type,
        "null_count": total - non_null,
        "null_rate": round((total - non_null) / total, 4) if total else 0.0,
        "distinct_count": distinct,
        "distinct_ratio": round(distinct / total, 4) if total else 0.0,
    }

    if data_type in NUMERIC_TYPES:
        profile["stats"] = _profile_numeric_column(conn, table, column)
    elif data_type in TEXT_TYPES:
        profile["stats"] = _profile_text_column(conn, table, column)
    # date/other types: null rate + cardinality only for now — good enough signal
    # for the SQL generator without adding a lot more branching logic.

    return profile


def profile_table(table: str) -> dict:
    """Profiles every column in a table. Returns {"row_count": int, "columns": {col: {...}}}"""
    engine = get_engine()
    schema = get_schema_summary()
    columns = schema.get(table, [])

    with engine.connect() as conn:
        row_count = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()
        column_profiles = {}
        for col_name, data_type in columns:
            if data_type in SKIP_TYPES:
                column_profiles[col_name] = {"data_type": data_type, "note": "skipped (unsupported type for profiling)"}
                continue
            column_profiles[col_name] = profile_column(conn, table, col_name, data_type)

    return {"table": table, "row_count": row_count, "columns": column_profiles}


def profile_database(exclude_tables: list[str] = None) -> dict:
    """
    Profiles every table in the public schema. Excludes embedding/junction
    tables by default since 'profiling' a foreign-key + vector table isn't
    useful the same way profiling the main data table is.
    """
    exclude_tables = exclude_tables or ["review_embeddings"]
    schema = get_schema_summary()
    return {
        table: profile_table(table)
        for table in schema
        if table not in exclude_tables
    }
