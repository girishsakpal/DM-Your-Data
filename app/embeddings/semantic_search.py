from sqlalchemy import text
from app.db.connection import get_engine
from app.embeddings.embedder import embed_text

DEFAULT_TOP_K = 10
# Cosine distance ranges 0 (identical) to 2 (opposite). Results above this are
# treated as "not actually relevant" — tune this if results feel too loose/strict.
MAX_DISTANCE = 0.8


def semantic_search(query: str, top_k: int = DEFAULT_TOP_K, table: str = "product_reviews",
                     text_column: str = "review_text", embedding_table: str = "review_embeddings",
                     candidate_ids: list[int] = None) -> dict:
    """
    Embeds the query, finds the top_k nearest review_text rows by cosine
    distance, and joins back to the source table for full row context.

    candidate_ids: if provided, restricts the search to only these row IDs.
    Used by the hybrid query path (Phase 3) to rank a SQL-filtered subset
    semantically, rather than searching the whole table.

    Returns {"query": str, "results": [{"id", "distance", ...row fields}, ...]}
    """
    query_vector = embed_text(query)
    # pgvector expects the literal as a string like '[0.1,0.2,...]'
    vector_literal = "[" + ",".join(str(x) for x in query_vector) + "]"

    engine = get_engine()

    id_filter_clause = ""
    params = {"qvec": vector_literal, "top_k": top_k}
    if candidate_ids is not None:
        if not candidate_ids:
            # Empty candidate set (SQL filter matched nothing) — no point querying.
            return {"query": query, "results": [], "result_count": 0, "raw_result_count": 0}
        id_filter_clause = "WHERE r.id = ANY(:candidate_ids)"
        params["candidate_ids"] = candidate_ids

    sql = text(f"""
        SELECT r.*, (e.embedding <=> CAST(:qvec AS vector)) AS distance
        FROM "{embedding_table}" e
        JOIN "{table}" r ON r.id = e.review_id
        {id_filter_clause}
        ORDER BY e.embedding <=> CAST(:qvec AS vector)
        LIMIT :top_k
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, params).mappings().all()

    results = [dict(row) for row in rows]
    filtered = [r for r in results if r["distance"] <= MAX_DISTANCE]

    return {
        "query": query,
        "results": filtered,
        "result_count": len(filtered),
        "raw_result_count": len(results),  # useful for debugging if MAX_DISTANCE feels too aggressive
    }
