from sqlalchemy import text
from app.db.connection import get_engine
from app.embeddings.embedder import embed_text

DEFAULT_TOP_K = 10
# Cosine distance ranges 0 (identical) to 2 (opposite). Results above this are
# treated as "not actually relevant" — tune this if results feel too loose/strict.
MAX_DISTANCE = 0.8


def semantic_search(query: str, top_k: int = DEFAULT_TOP_K, table: str = "product_reviews",
                     text_column: str = "review_text", embedding_table: str = "review_embeddings") -> dict:
    """
    Embeds the query, finds the top_k nearest review_text rows by cosine
    distance, and joins back to the source table for full row context.

    Returns {"query": str, "results": [{"id", "distance", ...row fields}, ...]}
    """
    query_vector = embed_text(query)
    # pgvector expects the literal as a string like '[0.1,0.2,...]'
    vector_literal = "[" + ",".join(str(x) for x in query_vector) + "]"

    engine = get_engine()
    sql = text(f"""
        SELECT r.*, (e.embedding <=> CAST(:qvec AS vector)) AS distance
        FROM "{embedding_table}" e
        JOIN "{table}" r ON r.id = e.review_id
        ORDER BY e.embedding <=> CAST(:qvec AS vector)
        LIMIT :top_k
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, {"qvec": vector_literal, "top_k": top_k}).mappings().all()

    results = [dict(row) for row in rows]
    filtered = [r for r in results if r["distance"] <= MAX_DISTANCE]

    return {
        "query": query,
        "results": filtered,
        "result_count": len(filtered),
        "raw_result_count": len(results),  # useful for debugging if MAX_DISTANCE feels too aggressive
    }
