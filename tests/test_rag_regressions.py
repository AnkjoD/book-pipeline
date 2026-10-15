"""Offline regression checks for contracts and failure cases discovered locally."""

import json
import urllib.error

import pytest
from langchain_core.documents import Document

from src.serve.rag import generator, pipeline, retriever
from src.serve.rag.context_builder import build_context
from src.serve.rag.verifier import verify_answer


def test_vietnamese_filters_and_year_boundaries():
    filters = retriever._parse_filters("sách kinh dị dưới 300 trang sau năm 2020")
    assert filters["genre"] == "horror"
    assert filters["max_pages"] == 300
    assert filters["max_pages_strict"]
    assert filters["min_year"] == 2021
    assert retriever._parse_filters("before 2000")["max_year"] == 1999
    assert retriever._parse_filters("sách xuất bản năm 9999")["is_impossible"]
    assert retriever._parse_filters("after 2020 before 2021")["is_impossible"]


def test_context_budget_and_provenance():
    document = {
        "book_id": "1",
        "title": "A",
        "evidence": "x" * 1000,
        "metadata": {"release_year_clean": 2008, "document_hash": "hash"},
    }
    text, sources = build_context([document, document], max_chars=200)
    assert len(text) <= 200
    assert sources[0]["document_hash"] == "hash"
    assert sources[0]["release_year"] == 2008
    assert build_context([document], max_chars=1)[1] == []


def test_mixed_search_keeps_title_and_strict_filters(monkeypatch):
    metadata = {"book_id": "1", "title": "Dune", "rating": 4.0, "pages_clean": 300}
    candidates = [
        (Document(page_content="wrong title", metadata={**metadata, "title": "Other"}), 0.1),
        (Document(page_content="boundary", metadata=metadata), 0.2),
        (Document(page_content="valid", metadata={**metadata, "rating": 4.1}), 0.3),
    ]

    class Store:
        def similarity_search_with_score(self, *args, **kwargs):
            return candidates

    monkeypatch.setattr(retriever, "_load_vectorstore", lambda: Store())
    results = retriever.search('"Dune" dark story rated above 4', limit=5)
    assert len(results) == 1
    assert results[0]["evidence"] == "valid"


def test_bare_title_uses_sql_without_loading_vector(monkeypatch):
    expected = [{"title": "The King's Equal"}]
    monkeypatch.setattr(retriever, "_sql_search", lambda *args, **kwargs: expected)
    monkeypatch.setattr(retriever, "_vector_search", lambda *args, **kwargs: pytest.fail())
    assert retriever.search("The King's Equal") == expected


@pytest.mark.parametrize(
    "answer,has_answer",
    [
        ("A factual answer without citations.", True),
        ("I cannot find this information in the catalog. [Doc 1]", False),
    ],
)
def test_generator_distinguishes_answer_from_citation(monkeypatch, answer, has_answer):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"message": {"content": answer}}).encode()

    monkeypatch.setattr(generator.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    result = generator.generate_answer("question", "context")
    assert result["has_answer"] is has_answer
    verdict = verify_answer(result, [{"doc_id": 1}])
    assert verdict["status"] == ("REGENERATE" if has_answer else "ABSTAIN")


def test_ollama_error_is_not_missing_evidence(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "search",
        lambda *args, **kwargs: [{"book_id": "1", "title": "A", "evidence": "some evidence"}],
    )

    def unavailable(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(generator.urllib.request, "urlopen", unavailable)
    result = pipeline.run_rag_pipeline("question")
    assert result["verification_status"] == "ERROR"
    assert result["error"]
    assert result["cited_books"] == []


def test_retrieval_error_does_not_expose_connection_details(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("postgresql://private-user:private-password@host")

    monkeypatch.setattr(pipeline, "search", broken)
    result = pipeline.run_rag_pipeline("question")
    assert result["verification_status"] == "ERROR"
    assert "private-password" not in str(result)


def test_sql_catalog_answer_does_not_need_ollama(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "search",
        lambda *args, **kwargs: [
            {
                "book_id": "1",
                "title": "Dune",
                "author_names": "Frank Herbert",
                "source": "sql",
                "evidence": "catalog data",
                "metadata": {"release_year_clean": 1965, "pages_clean": 300, "rating": 4.25},
            }
        ],
    )
    monkeypatch.setattr(pipeline, "generate_answer", lambda *args, **kwargs: pytest.fail())
    result = pipeline.run_rag_pipeline("Dune by Frank Herbert")
    assert result["verification_status"] == "ACCEPT"
    assert "300" in result["answer"] and "4.25" in result["answer"]
    assert "[Doc 1]" in result["answer"]
    assert "số trang: 300" in result["answer"]
    assert "năm: 1965" in result["answer"]
