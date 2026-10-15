"""Rank-based fusion; retrieval scores from different engines are not added."""


def reciprocal_rank_fusion(rankings: list[list[dict]], k: int = 60) -> list[dict]:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    scores, records, sources = {}, {}, {}
    for ranking in rankings:
        seen = set()
        for item in ranking:
            book_id = str(item["book_id"])
            if book_id in seen:
                continue
            seen.add(book_id)
            rank = len(seen)
            scores[book_id] = scores.get(book_id, 0) + 1 / (k + rank)
            records.setdefault(book_id, item)
            sources.setdefault(book_id, set()).add(item.get("source", "unknown"))
            # Keep graph provenance even if the dense result supplies the evidence.
            if item.get("graph_paths"):
                records[book_id] = {**records[book_id], "graph_paths": item["graph_paths"]}
    return [
        {
            **records[book_id],
            "fusion_score": scores[book_id],
            "retrieval_sources": sorted(sources[book_id]),
        }
        for book_id in sorted(scores, key=lambda key: (-scores[key], key))
    ]
