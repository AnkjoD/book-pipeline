"""Book-level evaluation; labels belong to the evaluator, never to search()."""

import math
from collections import defaultdict
from statistics import mean


def _unique_ids(ids) -> list[str]:
    if any(value is None or not str(value).strip() for value in ids):
        raise ValueError("Book IDs must not be empty")
    return list(dict.fromkeys(str(value) for value in ids))


def recall_at_k(retrieved_ids, relevant_ids, k: int) -> float:
    """Fraction of labeled relevant books recovered among k unique books."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    relevant = set(_unique_ids(relevant_ids))
    if not relevant:
        raise ValueError("Recall needs at least one labeled relevant book")
    top_k = set(_unique_ids(retrieved_ids)[:k])
    return len(top_k & relevant) / len(relevant)


def reciprocal_rank(retrieved_ids, relevant_ids) -> float:
    """Inverse rank of the first relevant unique book; zero when none is found."""
    relevant = set(_unique_ids(relevant_ids))
    if not relevant:
        raise ValueError("Reciprocal rank needs relevant labels")
    for rank, book_id in enumerate(_unique_ids(retrieved_ids), start=1):
        if book_id in relevant:
            return 1.0 / rank
    return 0.0


def passes_constraints(result: dict, constraints: list[dict]) -> bool:
    """Check explicit labels, independent of the retriever's filter parser."""
    metadata = {**result.get("metadata", {}), **result}
    comparisons = {
        "eq": lambda a, b: a == b,
        "lt": lambda a, b: a < b,
        "lte": lambda a, b: a <= b,
        "gt": lambda a, b: a > b,
        "gte": lambda a, b: a >= b,
        "contains": lambda a, b: str(b).casefold() in str(a).casefold(),
        "text_eq": lambda a, b: str(a).casefold() == str(b).casefold(),
    }
    for condition in constraints:
        op = condition["op"]
        if op not in comparisons:
            raise ValueError(f"Unsupported evaluation operator: {op}")
        value = metadata.get(condition["field"])
        if value is None:
            return False
        try:
            if isinstance(value, float) and not math.isfinite(value):
                return False
            if not comparisons[op](value, condition["value"]):
                return False
        except TypeError:
            return False
    return True


def evaluate_query(query: dict, results: list[dict], k: int) -> dict:
    """Evaluate top-k unique books. Unanswerable cases have no recall/MRR."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    unique = {}
    for item in results:
        book_id = _unique_ids([item["book_id"]])[0]
        unique.setdefault(book_id, item)
    top = list(unique.values())[:k]
    ids = list(unique)[:k]
    relevant = query.get("relevant_book_ids", [])
    unanswerable = query.get("expected_empty", False)
    if unanswerable and relevant:
        raise ValueError("Unanswerable cases cannot contain positive labels")
    constraints = query.get("constraints", [])
    valid = sum(passes_constraints(item, constraints) for item in top)
    return {
        "query_id": query["id"],
        "type": query.get("type", "unspecified"),
        "language": query.get("language", "unspecified"),
        "label_scope": query.get("label_scope", "unspecified"),
        "k": k,
        "retrieved_ids": ids,
        "result_count": len(top),
        "recall_at_k": None if unanswerable else recall_at_k(ids, relevant, k),
        "reciprocal_rank": None if unanswerable else reciprocal_rank(ids, relevant),
        "hit_at_k": None if unanswerable else bool(set(ids) & set(map(str, relevant))),
        "empty_correct": not ids if unanswerable else None,
        "constraint_rate": valid / len(top) if constraints and top else None,
    }


def summarize(rows: list[dict]) -> dict:
    """Errors contribute zero to answerable means, never disappear from the score."""
    answerable = [row for row in rows if row.get("type") != "unanswerable"]
    empty = [row for row in rows if row.get("type") == "unanswerable"]
    constraints = [row["constraint_rate"] for row in rows if row.get("constraint_rate") is not None]
    return {
        "cases": len(rows),
        "errors": sum(bool(row.get("error")) for row in rows),
        "answerable_cases": len(answerable),
        "mean_recall_at_k": mean(r.get("recall_at_k") or 0 for r in answerable)
        if answerable
        else None,
        "mrr_at_k": mean(r.get("reciprocal_rank") or 0 for r in answerable) if answerable else None,
        "hit_rate_at_k": mean(bool(r.get("hit_at_k")) for r in answerable) if answerable else None,
        "unanswerable_cases": len(empty),
        "empty_accuracy": mean(bool(r.get("empty_correct")) for r in empty) if empty else None,
        "mean_constraint_rate_on_nonempty": mean(constraints) if constraints else None,
    }


def grouped_summary(rows: list[dict], field: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return {name: summarize(group) for name, group in groups.items()}
