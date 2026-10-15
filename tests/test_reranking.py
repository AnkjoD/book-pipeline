"""Reranking must preserve candidate membership, constraints, and evidence."""

import copy

import pytest

from src.serve.rag import reranking, retriever


def book(book_id, evidence="passage", **updates):
    return {
        "book_id": book_id,
        "title": book_id,
        "evidence": evidence,
        "source": "vector",
        "score": 0.8,
        "metadata": {"document_hash": "hash-" + book_id, "pages_clean": 250},
        **updates,
    }


class Scores:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs, **kwargs):
        assert all(query == "plot query" and passage for query, passage in pairs)
        return self.scores


def test_rank_preserves_evidence_hash_and_input():
    candidates = [book("A"), book("B")]
    original = copy.deepcopy(candidates)
    result = reranking.rerank_results("plot query", candidates, 1, model=Scores([-1, 5]))
    assert result[0]["book_id"] == "B"
    assert result[0]["score"] == 0.8
    assert result[0]["metadata"] == candidates[1]["metadata"]
    assert result[0]["evidence"] == candidates[1]["evidence"]
    assert result[0]["rerank_score"] == 5
    assert candidates == original


def test_best_real_passage_per_book_and_stable_ties():
    candidates = [book("A", "a1"), book("B"), book("A", "a2")]
    result = reranking.rerank_results("plot query", candidates, 5, model=Scores([0, 1, 2]))
    assert [r["book_id"] for r in result] == ["A", "B"]
    assert result[0]["evidence"] == "a2"
    tied = reranking.rerank_results("plot query", candidates[:2], model=Scores([1, 1]))
    assert [r["book_id"] for r in tied] == ["A", "B"]


@pytest.mark.parametrize("scores", [[1], [1, float("nan")], [1, float("inf")]])
def test_invalid_model_scores_rejected(scores):
    with pytest.raises(ValueError):
        reranking.rerank_results("plot query", [book("A"), book("B")], model=Scores(scores))


def test_empty_candidates_never_load_model(monkeypatch):
    monkeypatch.setattr(reranking, "_load_reranker", lambda *a: pytest.fail())
    assert reranking.rerank_results("plot query", []) == []


def test_failure_preserves_filtered_candidates_and_hides_exception(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("private token in dependency error")

    monkeypatch.setattr(reranking, "rerank_results", broken)
    results = reranking.rerank_with_fallback("plot query", [book("A")], 5)
    assert results[0]["book_id"] == "A"
    assert results[0]["metadata"]["pages_clean"] < 300
    assert results[0]["rerank_status"] == "fallback"
    assert "private" not in str(results)


def test_sql_lookup_never_reranks(monkeypatch):
    monkeypatch.setenv("RAG_RERANK_ENABLED", "1")
    monkeypatch.setattr(retriever, "_sql_search", lambda *a, **k: [book("A")])
    monkeypatch.setattr(reranking, "_load_reranker", lambda *a: pytest.fail())
    assert retriever.search("Dune by Frank Herbert")[0]["book_id"] == "A"


def test_english_semantic_route_fetches_pool_before_reranking(monkeypatch):
    monkeypatch.setenv("RAG_RERANK_ENABLED", "1")

    def candidates(question, limit):
        assert limit == 30
        return [book("A"), book("B")]

    monkeypatch.setattr(retriever, "_vector_search", candidates)
    monkeypatch.setattr(reranking, "_load_reranker", lambda *a: Scores([0, 1]))
    # Use an unambiguous semantic phrase, and verify query survives the facade.
    monkeypatch.setattr(retriever, "_route", lambda q: "vector")
    monkeypatch.setattr(retriever, "_has_semantic_concept", lambda q: True)
    assert retriever.search("plot query", limit=1)[0]["book_id"] == "B"


def test_vietnamese_rerank_uses_english_plan_and_keeps_backend_filter(monkeypatch):
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStore

    from src.serve.rag import query_understanding as understanding

    monkeypatch.setenv("RAG_RERANK_ENABLED", "1")
    plan = understanding.QueryPlan(
        semantic_query="plot query",
        catalog_only=False,
        max_pages=300,
        max_pages_inclusive=False,
    )
    monkeypatch.setattr(understanding, "understand_question", lambda q: plan)

    class Store(VectorStore):
        def similarity_search(self, query, k=4, **kwargs):
            assert query == "plot query"
            assert k == 120
            assert kwargs["filter"] == {"pages_clean": {"$lt": 300}}
            return [
                Document(
                    page_content="passage",
                    metadata={
                        "book_id": value,
                        "title": value,
                        "pages_clean": 250,
                        "document_hash": "hash-" + value,
                    },
                )
                for value in ["A", "B"]
            ]

        @classmethod
        def from_texts(cls, *args, **kwargs):
            return cls()

    monkeypatch.setattr(retriever, "_load_vectorstore", Store)
    monkeypatch.setattr(reranking, "_load_reranker", lambda *a: Scores([0, 1]))
    result = retriever.search("tìm truyện dưới 300 trang", limit=1)
    assert result[0]["book_id"] == "B"
    assert result[0]["metadata"]["pages_clean"] == 250
    assert result[0]["metadata"]["document_hash"] == "hash-B"


@pytest.mark.parametrize("catalog_only,title", [(True, None), (False, "Dune")])
def test_vietnamese_catalog_or_named_title_bypasses_rerank(monkeypatch, catalog_only, title):
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStore

    from src.serve.rag import query_understanding as understanding

    monkeypatch.setenv("RAG_RERANK_ENABLED", "1")
    plan = understanding.QueryPlan(
        semantic_query="plot query",
        catalog_only=catalog_only,
        title=title,
    )
    monkeypatch.setattr(understanding, "understand_question", lambda q: plan)
    monkeypatch.setattr(retriever, "_prefer_exact_title", lambda plan: plan)

    class Store(VectorStore):
        def similarity_search(self, query, k=4, **kwargs):
            assert k == 20
            return [Document(page_content="passage", metadata={"book_id": "A"})]

        @classmethod
        def from_texts(cls, *args, **kwargs):
            return cls()

    monkeypatch.setattr(retriever, "_load_vectorstore", Store)
    monkeypatch.setattr(reranking, "_load_reranker", lambda *a: pytest.fail())
    assert retriever.search("tìm sách", limit=1)[0]["book_id"] == "A"


@pytest.mark.parametrize(
    "question",
    [
        "story about farm animals rebelling led by pigs Napoleon and Snowball",
        "a dark story about a child haunted by ghosts",
        "a novel about a village surrounded by mountains",
    ],
)
def test_plot_by_is_not_authorship(question):
    filters = retriever._parse_filters(question)
    assert filters["title"] is None and filters["author"] is None


@pytest.mark.parametrize(
    "question,title,author",
    [
        ("Dune by Frank Herbert", "Dune", "Frank Herbert"),
        ('"Dune" by Frank Herbert', "Dune", "Frank Herbert"),
        ("books by Bernard MacLaverty", None, "Bernard MacLaverty"),
        ("dark books by Stephen King", None, "Stephen King"),
        ("Dune của Frank Herbert", "Dune", "Frank Herbert"),
    ],
)
def test_explicit_author_lookup_survives(question, title, author):
    filters = retriever._parse_filters(question)
    assert filters["title"] == title
    assert filters["author"] == author
