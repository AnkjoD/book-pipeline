"""Unit tests for Context Builder, Verifier, and Pipeline integration."""

import pytest

from src.serve.rag import pipeline
from src.serve.rag.context_builder import build_context
from src.serve.rag.pipeline import run_rag_pipeline
from src.serve.rag.verifier import verify_answer


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
            "metadata": {"release_year_clean": 2008, "rating": 4.5},
        }
    ]
    text, prov = build_context(docs)
    assert "[Doc 1] Title: Clean Code" in text
    assert len(prov) == 1
    assert prov[0]["book_id"] == "1"
    assert "Year: 2008" in text
    assert prov[0]["release_year"] == 2008


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


@pytest.mark.parametrize(
    "first_citation,retry_citation,expected_status,expected_calls",
    [
        ("Doc 1", "Doc 1", "ACCEPT", 1),
        ("Doc 99", "Doc 1", "ACCEPT", 2),
        ("Doc 99", "Doc 99", "REGENERATE", 2),
    ],
)
def test_pipeline_only_exposes_accepted_answer(
    monkeypatch, first_citation, retry_citation, expected_status, expected_calls
):
    documents = [
        {
            "book_id": "1",
            "title": "Clean Code",
            "evidence": "Good code matters.",
            "metadata": {"release_year_clean": 2008},
        }
    ]
    monkeypatch.setattr(pipeline, "search", lambda *args, **kwargs: documents)
    generated = []

    def fake_generate(*args, **kwargs):
        citation = first_citation if not generated else retry_citation
        result = {
            "answer": f"Good code matters [{citation}].",
            "citations": [citation],
            "has_answer": True,
        }
        generated.append(result)
        return result

    monkeypatch.setattr(pipeline, "generate_answer", fake_generate)
    result = pipeline.run_rag_pipeline("What does this book teach?")

    assert len(generated) == expected_calls
    assert result["verification_status"] == expected_status
    if expected_status == "ACCEPT":
        assert result["answer"] == generated[-1]["answer"]
    else:
        assert result["answer"] != generated[-1]["answer"]
        assert "Doc 99" not in result["answer"]
        assert result["cited_books"] == []


def test_pipeline_abstain_hides_generated_answer(monkeypatch):
    monkeypatch.setattr(pipeline, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        lambda *args, **kwargs: {
            "answer": "An unsupported answer [Doc 99].",
            "citations": ["Doc 99"],
            "has_answer": True,
        },
    )

    result = pipeline.run_rag_pipeline("Unknown book")

    assert result["verification_status"] == "ABSTAIN"
    assert "unsupported" not in result["answer"]
    assert "Doc 99" not in result["answer"]
    assert result["citations"] == []
