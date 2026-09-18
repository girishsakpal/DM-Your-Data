import re
import io
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from app.db.connection import get_engine
from app.embeddings.embedder import embed_batch
from app.data_upload.registry import set_active_dataset

MAX_ROWS = 5000               # keep upload + embedding time reasonable on a laptop
MAX_FILE_BYTES = 5 * 1024 * 1024   # 5MB
MAX_EMBED_ROWS = 1000         # cap how many rows get embedded even if the CSV has more
MIN_AVG_TEXT_LENGTH = 25      # below this, a string column reads as categorical, not free text

DATASET_TABLE = "dataset"
DATASET_EMBEDDING_TABLE = "dataset_embeddings"
ID_COLUMN = "id"


class UploadError(Exception):
    """Raised for anything the UI should show as a clear, actionable message rather than a stack trace."""
    pass


def _sanitize_column_name(name: str, seen: set) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", str(name).strip().lower()).strip("_") or "column"
    if clean[0].isdigit():
        clean = f"c_{clean}"
    original = clean
    i = 2
    while clean in seen:
        clean = f"{original}_{i}"
        i += 1
    seen.add(clean)
    return clean


def _try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempts to convert object columns that look like dates into real
    datetime columns, so the SQL generator gets a proper date type instead
    of treating everything as text. Safe because of the ratio check - a
    genuinely non-date column (like 'region') will fail to parse for most
    values and get left alone.
    """
    for col in df.select_dtypes(include="object").columns:
        non_null = df[col].notna().sum()
        if non_null == 0:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        if parsed.notna().sum() / non_null >= 0.95:
            df[col] = parsed
    return df


def _detect_text_column(df: pd.DataFrame) -> str | None:
    """
    Picks the object-dtype column with the longest average string length,
    as long as it clears MIN_AVG_TEXT_LENGTH - a heuristic for 'this looks
    like free text (a review, a comment)' vs 'this looks like a short
    categorical value (a region, a status)'.
    """
    candidates = {}
    for col in df.select_dtypes(include="object").columns:
        avg_len = df[col].astype(str).map(len).mean()
        if avg_len >= MIN_AVG_TEXT_LENGTH:
            candidates[col] = avg_len

    if not candidates:
        return None
    return max(candidates, key=candidates.get)


def parse_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if len(file_bytes) > MAX_FILE_BYTES:
        raise UploadError(f"File is too large ({len(file_bytes) / 1024 / 1024:.1f}MB). "
                           f"Max size is {MAX_FILE_BYTES / 1024 / 1024:.0f}MB.")

    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise UploadError(f"Couldn't parse '{filename}' as CSV: {e}")

    if df.empty:
        raise UploadError("The CSV has no rows.")

    if len(df) > MAX_ROWS:
        raise UploadError(f"CSV has {len(df)} rows, which is over the {MAX_ROWS}-row limit for this demo. "
                           f"Trim it down and re-upload.")

    seen = set()
    df.columns = [_sanitize_column_name(c, seen) for c in df.columns]
    df = _try_parse_dates(df)

    # The upload pipeline always adds its own "id" column from the DataFrame's
    # index (see replace_active_dataset). If the CSV already has a column
    # called "id", rename it so the two don't collide - the alternative is
    # a hard-to-diagnose "duplicate column" failure from to_sql.
    if ID_COLUMN in df.columns:
        df = df.rename(columns={ID_COLUMN: f"source_{ID_COLUMN}"})

    return df


def replace_active_dataset(file_bytes: bytes, filename: str) -> dict:
    """
    Full pipeline: parse -> replace the 'dataset' table -> detect + embed a
    text column if one exists -> update the registry so every module
    (SQL generation, semantic search, hybrid) starts using this data.

    This REPLACES whatever was previously uploaded (single active dataset,
    by design) - it does not touch the original product_reviews seed table.

    Table load and embedding are deliberately decoupled: if embedding fails
    (GPU memory pressure, an unexpected model error, etc.) after the table
    has already been committed, that is NOT treated as an upload failure -
    the data is genuinely loaded and queryable via SQL. The failure is
    recorded in the returned metadata's "embedding_error" field instead of
    raising, so the UI can say "loaded, but semantic search isn't available"
    rather than reporting the whole upload as failed when it demonstrably
    was not.
    """
    df = parse_csv(file_bytes, filename)
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{DATASET_EMBEDDING_TABLE}"'))

    # pandas + SQLAlchemy infer sensible Postgres column types from the
    # DataFrame dtypes automatically (int64 -> BIGINT, float64 -> DOUBLE
    # PRECISION, datetime64 -> TIMESTAMP, object -> TEXT) - no manual
    # type mapping needed here.
    df_indexed = df.reset_index(drop=True)
    df_indexed.to_sql(DATASET_TABLE, con=engine, if_exists="replace", index=True, index_label=ID_COLUMN)
    # Everything above this line either succeeds or raises before any table
    # is touched - if we get here, the data is safely loaded and queryable,
    # regardless of what happens next.

    detected_text_column = _detect_text_column(df)
    text_column = None
    embedded_rows = 0
    embedding_error = None

    if detected_text_column:
        try:
            embed_subset = df_indexed.head(MAX_EMBED_ROWS)
            texts = embed_subset[detected_text_column].fillna("").astype(str).tolist()
            vectors = embed_batch(texts)
            # Determined from the actual embedding output rather than hardcoded -
            # if EMBEDDING_MODEL ever changes to a model with different output
            # dimensions, this table is created to match instead of silently
            # breaking on a dimension mismatch at insert time.
            vector_dim = len(vectors[0]) if vectors else 384

            with engine.begin() as conn:
                conn.execute(text(f'''
                    CREATE TABLE "{DATASET_EMBEDDING_TABLE}" (
                        row_id INTEGER PRIMARY KEY,
                        embedding VECTOR({vector_dim})
                    )
                '''))

            with engine.begin() as conn:
                # embed_subset's index IS the id (0..n-1) - that's what to_sql just
                # wrote as the "id" column via index_label. There's no in-memory
                # "id" column to read from; the index itself is the row identifier.
                for row_id, vector in zip(embed_subset.index, vectors):
                    vector_literal = "[" + ",".join(str(x) for x in vector) + "]"
                    conn.execute(
                        text(f'INSERT INTO "{DATASET_EMBEDDING_TABLE}" (row_id, embedding) '
                             f'VALUES (:row_id, CAST(:embedding AS vector))'),
                        {"row_id": int(row_id), "embedding": vector_literal},
                    )
            text_column = detected_text_column
            embedded_rows = len(embed_subset)

        except Exception as e:
            # Table load already succeeded (see comment above) - don't let an
            # embedding failure masquerade as an upload failure. Clean up any
            # partially-created embedding table, and fall back to SQL-only
            # for this dataset rather than leaving it half-configured.
            embedding_error = str(e)
            with engine.begin() as conn:
                conn.execute(text(f'DROP TABLE IF EXISTS "{DATASET_EMBEDDING_TABLE}"'))

    metadata = {
        "table": DATASET_TABLE,
        "embedding_table": DATASET_EMBEDDING_TABLE,
        "id_column": ID_COLUMN,
        "embedding_fk": "row_id",
        "text_column": text_column,
        "detected_text_column": detected_text_column,
        "embedding_error": embedding_error,
        "columns": list(df.columns),
        "row_count": len(df),
        "embedded_row_count": embedded_rows,
        "source_filename": filename,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    set_active_dataset(metadata)
    return metadata
