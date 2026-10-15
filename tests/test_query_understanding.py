"""Live routing must preserve explicit Vietnamese conditions and fail closed."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from src.serve.rag import pipeline, retriever
from src.serve.rag import query_understanding as understanding
from src.serve.rag.query_translator import BookPGVectorTranslator


def fake_model(monkeypatch, payload, *, complete=True):
    # Valid fixtures follow the new required-field contract. Malformed-plan tests
    # opt out so missing fields cannot be silently repaired by this fixture.
    if complete:
        payload = {"semantic_query": "book recommendations", "catalog_only": False, **payload}

    class Model:
        def invoke(self, messages):
            return SimpleNamespace(content=json.dumps(payload))

    monkeypatch.setattr(understanding, "_query_model", lambda: Model())


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"semantic_query": "memory"},
        {"catalog_only": False},
    ],
)
def test_missing_required_query_fields_clarify(monkeypatch, payload):
    fake_model(monkeypatch, payload, complete=False)
    with pytest.raises(understanding.QueryClarification):
        understanding.understand_question("tìm truyện về ký ức")


@pytest.mark.parametrize("semantic_query", ["", "   "])
def test_empty_semantic_request_cannot_search_generic_book(monkeypatch, semantic_query):
    fake_model(monkeypatch, {"semantic_query": semantic_query, "catalog_only": False})
    monkeypatch.setattr(retriever, "_load_vectorstore", lambda: pytest.fail())
    result = pipeline.run_rag_pipeline("tìm truyện về ký ức")
    assert result["verification_status"] == "CLARIFY"
    assert result["retrieved_count"] == 0


def test_catalog_lookup_can_have_empty_semantic_query(monkeypatch):
    fake_model(monkeypatch, {"semantic_query": "", "catalog_only": True, "title": "Dune"})
    plan = understanding.understand_question("tìm sách Dune")
    assert plan.title == "Dune" and plan.catalog_only


@pytest.mark.parametrize(
    "question,inclusive",
    [
        ("sách kinh dị dưới 300 trang", False),
        ("tim truyen kinh di duoi 300 trang", False),
        ("tim sach kinh di duoi 300 trnag", False),
        ("sách kinh dị không quá 300 trang", True),
        ("Cho mình truyện trinh thám, số trang ít hơn 300 nhé", False),
    ],
)
def test_literal_page_constraints_override_incorrect_model(monkeypatch, question, inclusive):
    fake_model(monkeypatch, {"max_pages": 500, "max_pages_inclusive": not inclusive})
    plan = understanding.understand_question(question)
    assert plan.max_pages == 300
    assert plan.max_pages_inclusive is inclusive


def test_decimal_comma_and_negation(monkeypatch):
    fake_model(monkeypatch, {"genre": "horror", "min_rating": 4.0})
    plan = understanding.understand_question("sách không thuộc thể loại kinh dị, đánh giá trên 4,2")
    assert plan.genre is None
    assert plan.excluded_genres == ["horror"]
    assert plan.min_rating == 4.2 and not plan.min_rating_inclusive
    _, kwargs = understanding.plan_to_structured_query(plan).accept(BookPGVectorTranslator())
    assert {"$not": {"genres": {"$ilike": "%horror%"}}} in kwargs["filter"]["$and"]


def test_model_cannot_invent_genre_for_named_book(monkeypatch):
    fake_model(
        monkeypatch, {"title": "The Hobbit", "genre": "adventure", "author": "J.R.R. Tolkien"}
    )
    plan = understanding.understand_question("Cuốn The Hobbit kể về điều gì?")
    assert plan.title == "The Hobbit"
    assert plan.genre is None
    assert plan.author is None


def test_literal_named_book_overrides_model_expansion(monkeypatch):
    fake_model(monkeypatch, {"title": "The Hobbit, or There and Back Again"})
    plan = understanding.understand_question("Cuốn The Hobbit kể về điều gì?")
    assert plan.title == "The Hobbit"


def test_ambiguous_request_clarifies_without_model_or_database(monkeypatch):
    monkeypatch.setattr(understanding, "_query_model", lambda: pytest.fail())
    monkeypatch.setattr(retriever, "_load_vectorstore", lambda: pytest.fail())
    result = pipeline.run_rag_pipeline("tìm sách mỏng mỏng thôi")
    assert result["verification_status"] == "CLARIFY"
    assert result["retrieved_count"] == 0


def test_invalid_model_plan_never_searches_unfiltered(monkeypatch):
    fake_model(monkeypatch, {"max_pages": "300"})
    monkeypatch.setattr(retriever, "_load_vectorstore", lambda: pytest.fail())
    result = pipeline.run_rag_pipeline("tìm sách kinh dị")
    assert result["verification_status"] == "CLARIFY"


def test_invented_numeric_threshold_is_blocked(monkeypatch):
    fake_model(monkeypatch, {"max_pages": 250})
    with pytest.raises(understanding.QueryClarification):
        understanding.understand_question("tìm truyện về tình bạn")


def test_unavailable_query_model_reports_error(monkeypatch):
    def unavailable():
        raise ConnectionError("private connection details")

    monkeypatch.setattr(understanding, "_query_model", unavailable)
    result = pipeline.run_rag_pipeline("tìm sách về ký ức")
    assert result["verification_status"] == "ERROR"
    assert "Ollama" in result["answer"]
    assert "private" not in str(result)


def test_vietnamese_uses_new_route(monkeypatch):
    expected = [{"title": "Example"}]
    monkeypatch.setattr(retriever, "_self_query_search", lambda *a, **k: expected)
    assert retriever.search("tim truyen kinh di duoi 300 trang") == expected


def test_simple_vietnamese_title_author_keeps_sql(monkeypatch):
    expected = [{"title": "Dune"}]
    monkeypatch.setattr(retriever, "_sql_search", lambda *a, **k: expected)
    monkeypatch.setattr(retriever, "_self_query_search", lambda *a, **k: pytest.fail())
    assert retriever.search("Dune của Frank Herbert") == expected


def test_live_self_query_normalizes_and_deduplicates_documents(monkeypatch):
    fake_model(monkeypatch, {"semantic_query": "dark memory"})
    metadata = {"book_id": "1", "title": "A", "document_hash": "hash"}

    class Store:
        def search(self, query, search_type, **kwargs):
            assert query == "dark memory"
            assert kwargs["filter"]["$and"][1] == {"pages_clean": {"$lt": 300}}
            return [Document(page_content="text", metadata=metadata)] * 2

    from langchain_core.vectorstores import VectorStore

    # Use an actual VectorStore subtype because SelfQueryRetriever validates it.
    class VectorStoreStub(VectorStore):
        def similarity_search(self, query, k=4, **kwargs):
            return Store().search(query, "similarity", **kwargs)

        @classmethod
        def from_texts(cls, *args, **kwargs):
            return cls()

    monkeypatch.setattr(retriever, "_load_vectorstore", VectorStoreStub)
    results = retriever.search("sách kinh dị dưới 300 trang")
    assert len(results) == 1
    assert results[0]["source"] == "self_query"
    assert results[0]["metadata"]["document_hash"] == "hash"
    assert results[0]["query_trace"]["semantic_query"] == "dark memory"


def test_metadata_self_query_answer_uses_template(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "search",
        lambda *args, **kwargs: [
            {
                "book_id": "1",
                "title": "A",
                "source": "self_query",
                "catalog_only": True,
                "evidence": "catalog",
                "metadata": {"pages_clean": 280, "rating": 4.5},
            }
        ],
    )
    monkeypatch.setattr(pipeline, "generate_answer", lambda *args, **kwargs: pytest.fail())
    result = pipeline.run_rag_pipeline("sách kinh dị dưới 300 trang")
    assert result["verification_status"] == "ACCEPT"
    assert "số trang: 280" in result["answer"]
