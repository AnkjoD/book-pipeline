"""Metric edge cases and the evaluation/search boundary, without local services."""

import json
from pathlib import Path

import pytest

from scripts.evaluate_relevance import run_evaluation, validate_dataset, write_reports
from src.serve.rag.evaluation import (
    evaluate_query,
    grouped_summary,
    passes_constraints,
    recall_at_k,
    reciprocal_rank,
    summarize,
)


def case(**updates):
    return {
        "id": "q",
        "question": "find A",
        "relevant_book_ids": ["A"],
        "type": "exact",
        "language": "en",
        **updates,
    }


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        recall_at_k(["A"], ["A"], k)
    with pytest.raises(ValueError):
        evaluate_query(case(), [], k)


def test_ids_normalized_duplicates_and_missing_labels():
    assert recall_at_k([1, "1", 2], [2, "2"], 2) == 1
    with pytest.raises(ValueError):
        recall_at_k([], [], 2)
    with pytest.raises(ValueError):
        reciprocal_rank([], [])
    with pytest.raises(ValueError):
        recall_at_k([None], ["A"], 2)


def test_report_rr_is_cut_off_at_k():
    report = evaluate_query(case(), [{"book_id": "X"}, {"book_id": "A"}], k=1)
    assert report["recall_at_k"] == report["reciprocal_rank"] == 0


def test_unanswerable_has_no_relevance_scores():
    report = evaluate_query(
        case(type="unanswerable", relevant_book_ids=[], expected_empty=True), [], k=5
    )
    assert report["recall_at_k"] is None
    assert report["reciprocal_rank"] is None
    assert report["empty_correct"] is True


def test_constraint_boundaries_and_missing_value():
    constraints = [{"field": "pages_clean", "op": "lt", "value": 300}]
    assert passes_constraints({"metadata": {"pages_clean": 299}}, constraints)
    assert not passes_constraints({"metadata": {"pages_clean": 300}}, constraints)
    assert not passes_constraints({"metadata": {}}, constraints)
    assert not passes_constraints({"metadata": {"pages_clean": float("nan")}}, constraints)
    with pytest.raises(ValueError):
        passes_constraints({}, [{"field": "pages_clean", "op": "raw_sql", "value": 0}])


def test_labels_not_sent_to_search():
    def search_fn(question, *, limit):
        assert question == "find A" and limit == 2
        return [{"book_id": "A"}]

    rows = run_evaluation({"cases": [case()]}, search_fn, 2)
    assert summarize(rows)["mrr_at_k"] == 1
    assert grouped_summary(rows, "language")["en"]["cases"] == 1


def test_service_error_not_scored_as_empty_success():
    def broken(question, *, limit):
        raise RuntimeError("service unavailable")

    dataset = {
        "cases": [
            case(),
            case(id="empty", type="unanswerable", relevant_book_ids=[], expected_empty=True),
        ]
    }
    summary = summarize(run_evaluation(dataset, broken, 2))
    assert summary["errors"] == 2
    assert summary["mean_recall_at_k"] == 0
    assert summary["empty_accuracy"] == 0


def test_dataset_and_report_serialization(tmp_path):
    dataset = json.loads((Path(__file__).parent / "relevance_dataset.json").read_text("utf-8"))
    validate_dataset(dataset)
    assert len(dataset["cases"]) == 40
    assert all(c["split"] == "dev" for c in dataset["cases"])
    for c in dataset["cases"]:
        assert set(c["relevant_book_ids"]) <= dataset["book_snapshots"].keys()
    with pytest.raises(ValueError):
        validate_dataset({"cases": [case(), case()]})
    path = tmp_path / "report.json"
    write_reports({"scope": "dev", "summary": {}, "limitations": [], "rows": []}, path)
    assert json.loads(path.read_text("utf-8"))["scope"] == "dev"
    assert path.with_suffix(".md").exists()
