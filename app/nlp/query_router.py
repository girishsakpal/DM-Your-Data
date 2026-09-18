import os
import re
from app.llm.ollama_client import generate
from app.data_upload import registry

ROUTE_SQL = "sql"
ROUTE_SEMANTIC = "semantic"
ROUTE_HYBRID = "hybrid"

# Signals that strongly suggest a precise, structured question - aggregations,
# counts, comparisons, sorting. These map to the SQL path.
SQL_SIGNALS = [
    r"\bhow many\b", r"\baverage\b", r"\bavg\b", r"\bcount\b", r"\bsum\b",
    r"\btotal\b", r"\btop \d+\b", r"\bmost expensive\b", r"\bcheapest\b",
    r"\bgroup by\b", r"\bhighest\b", r"\blowest\b", r"\bmaximum\b", r"\bminimum\b",
    r"\bbetween .* and .*\b", r"\blist all\b",
]

# Signals that suggest a fuzzy/conceptual question - sentiment, similarity,
# themes. These map to the semantic search path.
SEMANTIC_SIGNALS = [
    r"\bsimilar to\b", r"\babout\b", r"\bfeel(s|ing)?\b", r"\bfrustrat", r"\bhappy\b",
    r"\bupset\b", r"\bmood\b", r"\bsentiment\b", r"\bsounds like\b", r"\btone\b",
    r"\bcomplain", r"\bpraise", r"\bexpressing\b", r"\bdescribing\b",
]

# Words that suggest BOTH a structured filter AND a conceptual component are
# present together (e.g. "frustrated customers IN the West region").
HYBRID_CONNECTORS = [r"\bin the\b", r"\bfrom\b", r"\blast\b", r"\bwithin\b", r"\bunder \$?\d+\b", r"\bover \$?\d+\b"]

ROUTER_SYSTEM_PROMPT = """You classify natural language questions about a \
product reviews database into exactly one category:

- "sql": the question needs precise aggregation, counting, filtering, or sorting \
of structured data (prices, ratings, dates, regions, categories).
- "semantic": the question is conceptual/fuzzy and about the MEANING or SENTIMENT \
of review text (e.g. finding reviews that express a feeling or theme), with no \
structured filter.
- "hybrid": the question needs BOTH - a structured filter (e.g. a region, price \
range, or date range) AND a conceptual/semantic match on review text.

Respond with exactly one word: sql, semantic, or hybrid. Nothing else."""


def _matches_any(patterns, text_lower):
    return any(re.search(p, text_lower) for p in patterns)


def heuristic_route(question: str) -> str | None:
    """
    Fast, free first pass using keyword patterns. Returns None if the
    question doesn't clearly match one category, so the caller can fall
    back to the LLM classifier instead of guessing.
    """
    q = question.lower()
    has_sql_signal = _matches_any(SQL_SIGNALS, q)
    has_semantic_signal = _matches_any(SEMANTIC_SIGNALS, q)
    has_hybrid_connector = _matches_any(HYBRID_CONNECTORS, q)

    if has_semantic_signal and (has_sql_signal or has_hybrid_connector):
        return ROUTE_HYBRID
    if has_semantic_signal:
        return ROUTE_SEMANTIC
    if has_sql_signal:
        return ROUTE_SQL

    return None  # ambiguous - let the LLM decide


def llm_route(question: str) -> str:
    """LLM-based classification fallback for questions the heuristic can't confidently place."""
    model = os.getenv("ROUTER_MODEL", "llama3.2:3b")
    response = generate(model=model, prompt=f"Question: {question}", system=ROUTER_SYSTEM_PROMPT)
    cleaned = response.strip().lower()

    for route in (ROUTE_HYBRID, ROUTE_SEMANTIC, ROUTE_SQL):
        if route in cleaned:
            return route

    return ROUTE_SQL  # safe default if the LLM's output is unparseable


def classify(question: str) -> dict:
    """
    Returns {"route": "sql"|"semantic"|"hybrid", "method": "heuristic"|"llm"|"forced"}
    The method field is kept for transparency/debugging and eventually for
    the Phase 5 eval harness to measure heuristic vs. LLM agreement.
    """
    if not registry.has_semantic_support():
        # Active dataset has no text column - semantic/hybrid have nothing to search.
        # Force SQL rather than routing somewhere that will just return empty results.
        return {"route": ROUTE_SQL, "method": "forced"}

    route = heuristic_route(question)
    if route is not None:
        return {"route": route, "method": "heuristic"}

    route = llm_route(question)
    return {"route": route, "method": "llm"}
