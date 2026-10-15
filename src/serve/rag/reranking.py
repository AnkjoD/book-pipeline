"""Optional CPU cross-encoder reranking of already-filtered book candidates."""

import functools
import math
import os
import time

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def rerank_enabled() -> bool:
    return os.getenv("RAG_RERANK_ENABLED", "0").strip().lower() in {"1", "true", "yes"}


@functools.lru_cache(maxsize=1)
def _load_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu", max_length=512, trust_remote_code=False)


def rerank_results(query: str, results: list[dict], limit: int = 5, *, model=None) -> list[dict]:
    """Score real passages, select each book's best passage, preserve provenance.

    The default model ranks English text. Vietnamese callers pass the validated
    English semantic_query, not the original Vietnamese sentence. Scores are not
    probabilities and must not replace retrieval scores or hard constraints.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Reranking needs a nonempty semantic query")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if not results:
        return []
    for item in results:
        if not str(item.get("book_id", "")).strip() or not item.get("evidence"):
            raise ValueError("Candidates need book_id and real evidence")
    model_name = os.getenv("RERANK_MODEL", DEFAULT_MODEL)
    scorer = model if model is not None else _load_reranker(model_name)
    pairs = [(query, f"Title: {r.get('title', '')}\n{r['evidence']}") for r in results]
    scores = scorer.predict(pairs, batch_size=16, show_progress_bar=False)
    if len(scores) != len(results):
        raise ValueError("Reranker returned the wrong number of scores")
    books = {}
    for position, (result, score) in enumerate(zip(results, scores)):
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("Reranker returned a nonfinite score")
        book_id = str(result["book_id"])
        current = books.get(book_id)
        if current is None or score > current[0]:
            books[book_id] = (score, position, result)
    ranked = sorted(books.values(), key=lambda row: (-row[0], row[1]))[:limit]
    return [
        {**result, "rerank_score": score, "rerank_status": "ok", "rerank_model": model_name}
        for score, _, result in ranked
    ]


def rerank_with_fallback(query: str, results: list[dict], limit: int) -> list[dict]:
    """Reranker failure keeps the already-filtered candidates and records failure."""
    start = time.perf_counter()
    try:
        ranked = rerank_results(query, results, limit)
    except Exception as exc:
        # Never expose connection paths/tokens from dependency exceptions to UI.
        ranked = [
            {**r, "rerank_status": "fallback", "rerank_error": type(exc).__name__}
            for r in results[:limit]
        ]
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    return [{**r, "rerank_ms": elapsed} for r in ranked]
