"""Opt-in same-corpus dense, hybrid BM25+RRF, and local GraphRAG retrieval."""

import hashlib
import json

from .fusion import reciprocal_rank_fusion
from .graph_retriever import graph_search, relation_request
from .lexical import DEFAULT_PATH, connect, lexical_search, matches_filter

MODES = {"dense", "hybrid", "graph"}


def load_corpus(path=DEFAULT_PATH):
    with connect(path) as connection:
        manifest = json.loads(connection.execute("SELECT data FROM manifest").fetchone()[0])
        books = {
            r["book_id"]: json.loads(r["metadata"])
            for r in connection.execute("SELECT book_id,metadata FROM books")
        }
    fingerprint = hashlib.sha256(
        "\n".join(sorted(key + ":" + book["document_hash"] for key, book in books.items())).encode()
    ).hexdigest()
    if fingerprint != manifest["corpus_sha256"]:
        raise ValueError("Experimental corpus fingerprint mismatch")
    return manifest, books


def prepare_query(question, path=DEFAULT_PATH, *, include_plan=False):
    from .query_translator import BookPGVectorTranslator
    from .query_understanding import plan_to_structured_query, understand_question
    from .retriever import _parse_filters, _prefer_exact_title

    def packed(text, filters, catalog_only):
        return (text, filters, catalog_only) if include_plan else (text, filters)

    if not question.strip() or _parse_filters(question)["is_impossible"]:
        return packed(None, None, True)

    relation = relation_request(question)
    if relation:
        # Explicit, quoted relation requests use the same exclusion in every arm.
        _, books = load_corpus(path)
        anchors = [
            key for key, book in books.items() if book["title"].casefold() == relation[1].casefold()
        ]
        if not anchors:
            raise ValueError("The anchor title is outside the experiment corpus")
        conditions = [{"book_id": {"$ne": key}} for key in anchors]
        return packed(
            question, conditions[0] if len(conditions) == 1 else {"$and": conditions}, True
        )
    query_plan = understand_question(question)
    # The LLM may leave an English bound only in semantic_query. Literal numeric
    # conditions remain authoritative for every retrieval arm, just as in baseline.
    literal = _parse_filters(question)
    overrides = {}
    for field in (
        "min_pages",
        "max_pages",
        "min_rating",
        "max_rating",
        "min_year",
        "max_year",
        "exact_year",
    ):
        if literal[field] is not None:
            overrides[field] = literal[field]
            if field.endswith(("pages", "rating")):
                overrides[field + "_inclusive"] = not literal[field + "_strict"]
    query_plan = query_plan.model_copy(update=overrides)
    plan = _prefer_exact_title(plan_to_structured_query(query_plan))
    text, kwargs = BookPGVectorTranslator().visit_structured_query(plan)
    return packed(text, kwargs.get("filter"), query_plan.catalog_only)


def search_prepared(
    text, filters=None, *, mode="hybrid", limit=5, path=DEFAULT_PATH, dense_results=None
):
    from .retriever import _load_vectorstore

    if mode not in MODES or limit < 1:
        raise ValueError("Unknown retrieval mode or invalid limit")
    manifest, books = load_corpus(path)
    condition = {"book_id": {"$in": list(books)}}
    if filters:
        condition = {"$and": [condition, filters]}
    if dense_results is None:
        docs = _load_vectorstore().similarity_search(text, k=120, filter=condition)
        dense_results, seen = [], set()
        for doc in docs:
            metadata = doc.metadata
            book_id = str(metadata.get("book_id", ""))
            if (
                book_id not in books
                or metadata.get("document_hash") != books[book_id]["document_hash"]
            ):
                raise ValueError("Vector result is outside or stale relative to experiment corpus")
            if book_id in seen or not matches_filter(filters, metadata):
                continue
            seen.add(book_id)
            dense_results.append(
                {
                    "book_id": book_id,
                    "title": metadata.get("title", ""),
                    "author_names": metadata.get("author_names", ""),
                    "evidence": doc.page_content,
                    "metadata": dict(metadata),
                    "source": "dense_experiment",
                    "chunk_index": metadata.get("chunk_index", 0),
                    "score": None,
                    "distance": None,
                }
            )
    # A caller-supplied pool is validated too: no filter/corpus bypass in benchmarks.
    for item in dense_results:
        key = str(item["book_id"])
        if (
            key not in books
            or item["metadata"].get("document_hash") != books[key]["document_hash"]
            or not matches_filter(filters, item["metadata"])
        ):
            raise ValueError("Invalid frozen dense pool")
    if mode == "dense":
        results = dense_results
    else:
        auxiliary = (
            lexical_search(text, 30, filters, path)
            if mode == "hybrid"
            else graph_search(text, 30, filters, path)
        )
        results = reciprocal_rank_fusion([dense_results[:30], auxiliary])
        if mode == "graph" and relation_request(text):
            # Explicit same-series/shared-credit requests require a witnessed path.
            # Dense similarity must not reintroduce an unrelated book after fusion.
            witnessed = {item["book_id"] for item in auxiliary}
            results = [item for item in results if item["book_id"] in witnessed]
    return [
        {
            **item,
            "experimental_mode": mode,
            "experiment_books": manifest["books"],
            "query_trace": {
                "semantic_query": text,
                "filter": filters,
                "corpus_sha256": manifest["corpus_sha256"],
                "graph_paths": item.get("graph_paths", []),
            },
            "source": mode + "_experiment",
        }
        for item in results[:limit]
    ]


def search_experiment(question, limit=5, *, mode="hybrid", path=DEFAULT_PATH):
    if not isinstance(question, str) or not question.strip():
        return []
    if relation_request(question) and mode != "graph":
        from .query_understanding import QueryClarification

        raise QueryClarification("Bạn chọn chế độ GraphRAG để tìm theo quan hệ giữa các sách nhé.")
    text, filters, catalog_only = prepare_query(question, path, include_plan=True)
    if text is None:
        return []
    return [
        {**item, "catalog_only": catalog_only}
        for item in search_prepared(text, filters, mode=mode, limit=limit, path=path)
    ]
