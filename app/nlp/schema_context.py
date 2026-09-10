from sqlalchemy import text
from app.db.connection import get_engine, get_schema_summary


def build_schema_context(sample_rows: int = 2) -> str:
    """
    Produces a compact schema description for the LLM prompt, e.g.:

        Table: product_reviews
        Columns: id (integer), product_name (text), price (numeric), ...
        Sample rows:
          (1, 'Wireless Mouse', 799.00, ...)
          (2, 'Wireless Mouse', 799.00, ...)

    Including a couple of sample rows measurably helps the LLM guess correct
    value formats (e.g. date format, whether region is 'West' vs 'west').
    """
    schema = get_schema_summary()
    engine = get_engine()
    blocks = []

    with engine.connect() as conn:
        for table_name, columns in schema.items():
            col_desc = ", ".join(f"{name} ({dtype})" for name, dtype in columns)
            block = f"Table: {table_name}\nColumns: {col_desc}"

            if sample_rows > 0:
                try:
                    rows = conn.execute(
                        text(f'SELECT * FROM "{table_name}" LIMIT :n'),
                        {"n": sample_rows},
                    ).fetchall()
                    if rows:
                        sample_text = "\n".join(f"  {tuple(r)}" for r in rows)
                        block += f"\nSample rows:\n{sample_text}"
                except Exception:
                    # Skip sample rows for tables that error (e.g. vector columns
                    # with no straightforward repr) — schema info alone is still useful.
                    pass

            blocks.append(block)

    return "\n\n".join(blocks)
