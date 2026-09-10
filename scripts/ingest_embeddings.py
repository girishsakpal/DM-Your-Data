"""
Embeds review_text for any product_reviews rows that don't yet have an
embedding, and stores the vectors in review_embeddings. Safe to re-run —
only processes rows that are missing an embedding.

Usage:
    python scripts/ingest_embeddings.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.db.connection import get_engine
from app.embeddings.embedder import embed_batch

BATCH_SIZE = 64


def fetch_unembedded_rows(conn):
    result = conn.execute(text("""
        SELECT r.id, r.review_text
        FROM product_reviews r
        LEFT JOIN review_embeddings e ON e.review_id = r.id
        WHERE e.review_id IS NULL
    """))
    return result.fetchall()


def upsert_embeddings(conn, ids, vectors):
    for review_id, vector in zip(ids, vectors):
        vector_literal = "[" + ",".join(str(x) for x in vector) + "]"
        conn.execute(
            text("""
                INSERT INTO review_embeddings (review_id, embedding)
                VALUES (:review_id, CAST(:embedding AS vector))
                ON CONFLICT (review_id) DO UPDATE SET embedding = EXCLUDED.embedding
            """),
            {"review_id": review_id, "embedding": vector_literal},
        )


def main():
    engine = get_engine()
    with engine.connect() as conn:
        rows = fetch_unembedded_rows(conn)

        if not rows:
            print("Nothing to do — all rows already have embeddings.")
            return

        print(f"Embedding {len(rows)} row(s)...")

        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            ids = [r[0] for r in batch]
            texts = [r[1] for r in batch]

            vectors = embed_batch(texts)
            upsert_embeddings(conn, ids, vectors)
            conn.commit()

            print(f"  Embedded rows {i + 1}-{i + len(batch)} of {len(rows)}")

        print("Done.")


if __name__ == "__main__":
    main()
