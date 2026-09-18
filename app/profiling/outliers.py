from sqlalchemy import text
from app.db.connection import get_engine, get_schema_summary
from app.profiling.profiler import NUMERIC_TYPES

Z_SCORE_THRESHOLD = 3.0   # values beyond 3 standard deviations from the mean
IQR_MULTIPLIER = 1.5      # classic Tukey's fence - values beyond Q1-1.5*IQR or Q3+1.5*IQR


def detect_outliers_zscore(table: str, column: str, threshold: float = Z_SCORE_THRESHOLD, limit: int = 100) -> dict:
    """
    Flags rows where the column's z-score exceeds the threshold. Computed
    entirely in SQL (no fetching the whole table into Python) so this scales
    reasonably even on larger datasets.
    """
    engine = get_engine()
    sql = text(f"""
        WITH stats AS (
            SELECT AVG("{column}"::numeric) AS mean_val, STDDEV("{column}"::numeric) AS std_val
            FROM "{table}" WHERE "{column}" IS NOT NULL
        )
        SELECT r.*, ((r."{column}"::numeric - s.mean_val) / NULLIF(s.std_val, 0)) AS z_score
        FROM "{table}" r, stats s
        WHERE r."{column}" IS NOT NULL
          AND ABS((r."{column}"::numeric - s.mean_val) / NULLIF(s.std_val, 0)) > :threshold
        ORDER BY ABS((r."{column}"::numeric - s.mean_val) / NULLIF(s.std_val, 0)) DESC
        LIMIT :limit
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, {"threshold": threshold, "limit": limit}).mappings().all()

    return {
        "table": table, "column": column, "method": "zscore", "threshold": threshold,
        "outliers": [dict(r) for r in rows], "outlier_count": len(rows),
    }


def detect_outliers_iqr(table: str, column: str, multiplier: float = IQR_MULTIPLIER, limit: int = 100) -> dict:
    """
    Flags rows outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR]. Generally
    more robust than z-score for skewed distributions (e.g. price data, which
    is rarely symmetric) since it doesn't assume a normal distribution.
    """
    engine = get_engine()
    sql = text(f"""
        WITH quartiles AS (
            SELECT
                percentile_cont(0.25) WITHIN GROUP (ORDER BY "{column}"::numeric) AS q1,
                percentile_cont(0.75) WITHIN GROUP (ORDER BY "{column}"::numeric) AS q3
            FROM "{table}" WHERE "{column}" IS NOT NULL
        ),
        bounds AS (
            SELECT q1, q3, (q3 - q1) AS iqr,
                   q1 - :multiplier * (q3 - q1) AS lower_bound,
                   q3 + :multiplier * (q3 - q1) AS upper_bound
            FROM quartiles
        )
        SELECT r.*, b.lower_bound, b.upper_bound
        FROM "{table}" r, bounds b
        WHERE r."{column}" IS NOT NULL
          AND (r."{column}"::numeric < b.lower_bound OR r."{column}"::numeric > b.upper_bound)
        ORDER BY r."{column}" DESC
        LIMIT :limit
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, {"multiplier": multiplier, "limit": limit}).mappings().all()

    return {
        "table": table, "column": column, "method": "iqr", "multiplier": multiplier,
        "outliers": [dict(r) for r in rows], "outlier_count": len(rows),
    }


def detect_outliers(table: str, column: str, method: str = "zscore", **kwargs) -> dict:
    if method == "iqr":
        return detect_outliers_iqr(table, column, **kwargs)
    return detect_outliers_zscore(table, column, **kwargs)


def detect_all_outliers(table: str, method: str = "zscore") -> dict:
    """Runs outlier detection on every numeric column in a table - useful for a quick full sweep."""
    schema = get_schema_summary()
    numeric_columns = [col for col, dtype in schema.get(table, []) if dtype in NUMERIC_TYPES]

    return {col: detect_outliers(table, col, method=method) for col in numeric_columns}
