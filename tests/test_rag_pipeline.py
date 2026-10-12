"""Unit tests for Context Builder, Verifier, and Pipeline integration."""
import pytest
from src.serve.rag.context_builder import build_context
from src.serve.rag.verifier import verify_answer
from src.serve.rag.pipeline import run_rag_pipeline


def test_context_builder_empty():
    text, prov = build_context([])
    assert text == "No relevant documents found."
    assert prov == []


def test_context_builder_formatting():
    docs = [
        {
            "book_id": "1",
            "title": "Clean Code",
            "author_names": "Robert Martin",
            "evidence": "Good code matters.",
            "source": "sql",
            "metadata": {"release_year": 2008, "rating": 4.5},
        }
    ]
    text, prov = build_context(docs)
    assert "[Doc 1] Title: Clean Code" in text
    assert len(prov) == 1
    assert prov[0]["book_id"] == "1"


def test_verifier_accept():
    prov = [{"doc_id": 1, "book_id": "1", "title": "Clean Code"}]
    ans = {
        "answer": "Clean code is key [Doc 1].",
        "citations": ["Doc 1"],
        "has_answer": True,
    }
    verdict = verify_answer(ans, prov)
    assert verdict["status"] == "ACCEPT"
    assert len(verdict["cited_books"]) == 1


def test_verifier_ghost_citation_reject():
    prov = [{"doc_id": 1, "book_id": "1", "title": "Clean Code"}]
    ans = {
        "answer": "Clean code is key [Doc 99].",
        "citations": ["Doc 99"],
        "has_answer": True,
    }
    verdict = verify_answer(ans, prov)
    assert verdict["status"] == "REGENERATE"
    assert "Doc 99" in verdict["invalid_citations"]


def test_verifier_abstain_on_empty():
    verdict = verify_answer({"answer": "", "citations": [], "has_answer": False}, [])
    assert verdict["status"] == "ABSTAIN"


def test_pipeline_impossible_query_abstains():
    res = run_rag_pipeline("books published in year 9999", limit=2)
    assert res["verification_status"] == "ABSTAIN"
    assert res["retrieved_count"] == 0
    assert res["cited_books"] == []
