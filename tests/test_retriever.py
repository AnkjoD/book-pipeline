"""Unit and integration test checks cho retriever.py."""

import pytest
from src.serve.rag.retriever import _parse_filters, _route, search


def test_empty_query_raises_value_error():
    assert search("   ") == []
    assert search("") == []


def test_route_classification():
    assert _route("Dune by Frank Herbert") == "sql"
    assert _route("books rated at least 4.2 under 350 pages") == "sql"
    assert _route("a dark story about memory and grief") == "vector"
    assert _route("dark fantasy under 400 pages rated above 4") == "mixed"
    assert _route("books above rating 6 and under 0 pages") == "sql"


def test_exact_sql_search():
    results = search("Dune by Frank Herbert", limit=3)
    assert len(results) >= 1
    top = results[0]
    assert "dune" in top["title"].lower()
    assert "frank herbert" in top["author_names"].lower()
    assert top["source"] == "sql"
    assert top["score"] == 1.0
    assert top["distance"] == 0.0
    assert "book_id" in top
    assert "document_hash" in top["metadata"]


def test_numeric_sql_search():
    results = search("books rated at least 4.2 under 350 pages", limit=5)
    assert len(results) > 0
    assert len(results) <= 5

    seen_ids = set()
    for item in results:
        # Check no duplicates
        assert item["book_id"] not in seen_ids
        seen_ids.add(item["book_id"])

        # Check hard constraints
        rating = item["metadata"]["rating"]
        pages = item["metadata"]["pages_clean"]
        assert rating is not None and rating >= 4.2
        assert pages is not None and pages <= 350
        assert item["source"] == "sql"


def test_semantic_vector_search():
    results = search("a dark story about memory and grief", limit=5)
    assert len(results) > 0
    assert len(results) <= 5

    seen_ids = set()
    for item in results:
        # Check no duplicates
        assert item["book_id"] not in seen_ids
        seen_ids.add(item["book_id"])

        assert item["source"] == "vector"
        assert item["score"] >= 0.0
        assert item["distance"] >= 0.0
        assert len(item["evidence"]) > 0


def test_mixed_search():
    results = search("dark fantasy under 400 pages rated above 4", limit=5)
    assert len(results) > 0
    assert len(results) <= 5

    seen_ids = set()
    for item in results:
        assert item["book_id"] not in seen_ids
        seen_ids.add(item["book_id"])

        rating = item["metadata"]["rating"]
        pages = item["metadata"]["pages_clean"]
        genres = item["metadata"]["genres"].lower()

        assert rating is not None and rating >= 4.0
        assert pages is not None and pages <= 400
        assert "fantasy" in genres
        assert item["source"] == "mixed"


def test_impossible_constraints_returns_empty():
    results = search("books above rating 6 and under 0 pages", limit=5)
    assert results == []
