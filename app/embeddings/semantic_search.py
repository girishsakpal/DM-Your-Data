from sqlalchemy import text
from app.db.connection import get_engine
from app.embeddings.embedder import embed_text
from app.data_upload import registry

DEFAULT_TOP_K = 10
# Cosine distance ranges 0 (identical) to 2 (opposite). Results above this are
# treated as "not actually relevant" - tune this if results feel too loose/strict.
MAX_DISTANCE = 0.8


def semantic_search(query: str, top_k: int = DEFAULT_TOP_K, table: str = None,
                     text_column: str = None, embedding_table: str = None,
                     candidate_ids: list[int] = None) -> dict:
    """
    Embeds the query, finds the top_k nearest text-column rows by cosine
    distance, and joins back to the source table for full row context.

    table/text_column/embedding_table default to whatever dataset is
    currently active (Phase 6.1's upload feature) - the seed product_reviews
    table if nothing's been uploaded. Pass them explicitly to override.

    candidate_ids: if provided, restricts the search to only these row IDs.
    Used by the hybrid query path (Phase 3) to rank a SQL-filtered subset
    semantically, rather than searching the whole table.

    Returns {"query": str, "results": [{"id", "distance", ...row fields}, ...]}
    """
    table = table or registry.active_table()
    text_column = text_column or registry.active_text_column()
    embedding_table = embedding_table or registry.active_embedding_table()
    id_column = registry.active_id_column()
    embedding_fk = registry.active_embedding_fk()

    if not text_column:
        # This dataset has no column that looked like free text - nothing to search semantically.
        return {"query": query, "results": [], "result_count": 0, "raw_result_count": 0,
                "note": "The active dataset has no text column suitable for semantic search."}

    query_vector = embed_text(query)
    # pgvector expects the literal as a string like '[0.1,0.2,...]'
    vector_literal = "[" + ",".join(str(x) for x in query_vector) + "]"

    engine = get_engine()

    id_filter_clause = ""
    params = {"qvec": vector_literal, "top_k": top_k}
    if candidate_ids is not None:
        if not candidate_ids:
            # Empty candidate set (SQL filter matched nothing) - no point querying.
            return {"query": query, "results": [], "result_count": 0, "raw_result_count": 0}
        id_filter_clause = f'WHERE r."{id_column}" = ANY(:candidate_ids)'
        params["candidate_ids"] = candidate_ids

    sql = text(f"""
        SELECT r.*, (e.embedding <=> CAST(:qvec AS vector)) AS distance
        FROM "{embedding_table}" e
        JOIN "{table}" r ON r."{id_column}" = e."{embedding_fk}"
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
