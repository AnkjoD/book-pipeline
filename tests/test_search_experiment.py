"""Offline checks for lexical, graph, fusion, and experiment isolation."""

import hashlib
import json

import pytest

from scripts.build_search_experiment import create_artifact
from src.serve.rag.experimental import load_corpus, prepare_query, search_prepared
from src.serve.rag.fusion import reciprocal_rank_fusion
from src.serve.rag.graph_retriever import graph_search, grounded_entities
from src.serve.rag.lexical import lexical_search, matches_filter


@pytest.fixture
def corpus(tmp_path):
    def book(key, title, author, description, series="", pages=200):
        return {
            "book_id": key,
            "title": title,
            "author_names": author,
            "document_hash": "hash-" + key,
            "pages_clean": pages,
            "series_info": series,
            "genres": "Science Fiction",
            "document_text": f"Title: {title}\nDescription: {description}",
        }

    books = [
        book("A", "Dune", "Frank Herbert", "Paul Atreides lives on Arrakis."),
        book("B", "Dune Messiah", "Frank Herbert", "A sequel about Paul."),
        book("C", "The Hunger Games", "Suzanne Collins", "Katniss volunteers.", "Games #1.0"),
        book("D", "Catching Fire", "Suzanne Collins", "Katniss returns.", "Games #2.0", 500),
    ]
    fingerprint = hashlib.sha256(
        "\n".join(sorted(b["book_id"] + ":" + b["document_hash"] for b in books)).encode()
    ).hexdigest()
    path = tmp_path / "corpus.sqlite"
    create_artifact(
        path,
        books,
        {
            "A": [
                {
                    "name": "Paul Atreides",
                    "kind": "character",
                    "quote": "Paul Atreides lives on Arrakis.",
                }
            ]
        },
        {"books": 4, "corpus_sha256": fingerprint},
    )
    return path


def test_grounding_rejects_invented_entity_and_quote():
    entities, rejected = grounded_entities(
        {
            "entities": [
                {"kind": "character", "name": "Paul", "quote": "Paul lives on Arrakis."},
                {"kind": "character", "name": "Harry", "quote": "Paul lives on Arrakis."},
                {"kind": "place", "name": "London", "quote": "London is nearby."},
            ]
        },
        "Paul lives on Arrakis.",
    )
    assert len(entities) == 1 and rejected == 2


def test_bm25_returns_real_evidence_and_hash(corpus):
    result = lexical_search("Paul Atreides", path=corpus)
    assert result[0]["book_id"] == "A"
    assert result[0]["evidence"] == "Paul Atreides lives on Arrakis."
    assert result[0]["metadata"]["document_hash"] == "hash-A"
    assert result[0]["chunk_index"] is None
    assert lexical_search('" OR * ; DROP TABLE books;', path=corpus) == []
    assert load_corpus(corpus)[0]["books"] == 4


def test_graph_three_edge_path_and_missing_series(corpus):
    results = graph_search(
        'other books with the same credited person as "Dune"',
        filters={"book_id": {"$ne": "A"}},
        path=corpus,
    )
    assert results[0]["book_id"] == "B"
    assert len(results[0]["graph_paths"][0]) == 3
    assert all(edge["document_hash"] for edge in results[0]["graph_paths"][0])
    # No fabricated Dune series relation: this fixture has no series metadata.
    assert (
        graph_search(
            'other books in the same series as "Dune"',
            filters={"book_id": {"$ne": "A"}},
            path=corpus,
        )
        == []
    )


def test_graph_cannot_return_high_score_book_violating_filter(corpus):
    condition = {"pages_clean": {"$lt": 300}}
    assert all(
        r["metadata"]["pages_clean"] < 300
        for r in graph_search("Suzanne Collins", filters=condition, path=corpus)
    )
    assert all(
        r["metadata"]["pages_clean"] < 300
        for r in lexical_search("Katniss", filters=condition, path=corpus)
    )


def test_negation_preserves_missing_value_semantics():
    condition = {"$not": {"genres": {"$ilike": "%horror%"}}}
    assert not matches_filter(condition, {})
    assert not matches_filter(condition, {"genres": "Horror"})
    assert matches_filter(condition, {"genres": "Fantasy"})
    assert matches_filter({"title": {"$ilike": r"%100\%%"}}, {"title": "100% true"})
    assert not matches_filter({"title": {"$ilike": r"%100\%%"}}, {"title": "1000 true"})


def test_fusion_dedup_does_not_add_incompatible_scores_or_mutate():
    first = {"book_id": "A", "source": "dense", "score": 0.9, "metadata": {"hash": "a"}}
    second = {"book_id": "B", "source": "lexical", "score": 9999}
    rankings = [[first, first, second], [second]]
    original = json.dumps(rankings)
    result = reciprocal_rank_fusion(rankings)
    assert result[0]["book_id"] == "B"
    assert result[0]["fusion_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert json.dumps(rankings) == original
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], 0)


def test_experiment_rejects_outside_and_stale_frozen_candidates(corpus):
    for key, document_hash in [("outside", "h"), ("A", "old")]:
        with pytest.raises(ValueError):
            search_prepared(
                "query",
                path=corpus,
                dense_results=[
                    {
                        "book_id": key,
                        "metadata": {"document_hash": document_hash},
                    }
                ],
            )


def test_explicit_relation_plan_excludes_anchor_without_calling_model(corpus, monkeypatch):
    from src.serve.rag import query_understanding

    monkeypatch.setattr(query_understanding, "understand_question", lambda q: pytest.fail())
    query, filters = prepare_query('other books in the same series as "The Hunger Games"', corpus)
    assert "The Hunger Games" in query
    assert not matches_filter(filters, {"book_id": "C"})
    assert matches_filter(filters, {"book_id": "D"})


def test_explicit_english_numeric_filter_is_not_lost(corpus, monkeypatch):
    from src.serve.rag import query_understanding, retriever

    monkeypatch.setattr(
        query_understanding,
        "understand_question",
        lambda q: query_understanding.QueryPlan(
            semantic_query="Mars survival under 400 pages", catalog_only=False
        ),
    )
    monkeypatch.setattr(retriever, "_prefer_exact_title", lambda q: q)
    _, filters = prepare_query("story about Mars survival under 400 pages", corpus)
    assert filters == {"pages_clean": {"$lt": 400}}


def test_graph_fusion_cannot_reintroduce_unrelated_book(corpus):
    unrelated = {
        "book_id": "A",
        "title": "Dune",
        "source": "dense",
        "metadata": {"document_hash": "hash-A", "book_id": "A"},
    }
    results = search_prepared(
        'other books in the same series as "The Hunger Games"',
        {"book_id": {"$ne": "C"}},
        mode="graph",
        path=corpus,
        dense_results=[unrelated],
    )
    assert [r["book_id"] for r in results] == ["D"]


def test_public_hybrid_does_not_claim_to_solve_explicit_graph_relations(corpus):
    from src.serve.rag.experimental import search_experiment
    from src.serve.rag.query_understanding import QueryClarification

    with pytest.raises(QueryClarification):
        search_experiment('other books in the same series as "Dune"', path=corpus)


@pytest.mark.parametrize("mode", ["hybrid", "graph"])
def test_pipeline_dispatches_opt_in_mode_without_touching_baseline(monkeypatch, mode):
    from src.serve.rag import experimental, pipeline

    def search(question, limit, *, mode):
        assert mode in {"hybrid", "graph"}
        return [
            {
                "book_id": "A",
                "title": "Dune",
                "evidence": "catalog",
                "catalog_only": True,
                "metadata": {"pages_clean": 200},
            }
        ]

    monkeypatch.setattr(experimental, "search_experiment", search)
    monkeypatch.setattr(pipeline, "search", lambda *a, **k: pytest.fail())
    monkeypatch.setattr(pipeline, "generate_answer", lambda *a, **k: pytest.fail())
    assert pipeline.run_rag_pipeline("Dune", retrieval_mode=mode)["verification_status"] == "ACCEPT"
