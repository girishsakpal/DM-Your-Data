from decimal import Decimal


def normalize_value(v):
    """
    Normalizes a single cell value so equivalent results compare equal even
    if types differ slightly (Decimal vs float, trailing zeros, etc.) - this
    is what makes this 'execution accuracy' rather than a brittle exact
    string/type match.
    """
    if isinstance(v, (Decimal, float)):
        return round(float(v), 2)
    if isinstance(v, str):
        return v.strip().lower()
    return v


def normalize_row(row) -> tuple:
    return tuple(normalize_value(v) for v in row)


def result_sets_match(actual_rows, expected_rows) -> bool:
    """
    Compares two result sets as unordered sets of normalized rows -
    intentionally ignoring row order and column order/naming, since a
    correct query can list columns in a different order than the reference
    and still be correct. This is 'execution accuracy', not exact-match.
    """
    actual_set = {normalize_row(r) for r in actual_rows}
    expected_set = {normalize_row(r) for r in expected_rows}
    return actual_set == expected_set


def precision_at_k(returned_ids: list[int], relevant_ids: list[int], k: int = None) -> dict:
    """
    Standard retrieval metrics for the semantic/hybrid paths.
    precision = fraction of returned results that are actually relevant
    recall = fraction of relevant results that were actually returned
    """
    k = k or len(returned_ids)
    top_k = returned_ids[:k]
    relevant_set = set(relevant_ids)
    hits = [rid for rid in top_k if rid in relevant_set]

    precision = len(hits) / len(top_k) if top_k else 0.0
    recall = len(hits) / len(relevant_set) if relevant_set else 0.0

    return {"precision": round(precision, 3), "recall": round(recall, 3), "hits": hits, "returned": top_k}
