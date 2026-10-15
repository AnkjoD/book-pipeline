import pytest

from src.serve.rag.evaluation import evaluate_query, recall_at_k, reciprocal_rank


def test_recall_finds_half():
    result = recall_at_k(["Doc 1", "Doc 2", "Doc 3"], ["Doc 2", "Doc 4"], k=2)
    assert result == 0.5


def test_recall_finds_none():
    result = recall_at_k(["Doc 1", "Doc 2", "Doc 3"], ["Doc 4", "Doc 5"], k=2)
    assert result == 0.0


def test_recall_finds_all():
    result = recall_at_k(["Doc 1", "Doc 2", "Doc 3"], ["Doc 1", "Doc 2"], k=3)
    assert result == 1.0


def test_recall_with_duplicates():
    result = recall_at_k(
        ["B", "B", "A"],
        ["A", "D"],
        k=2,
    )

    assert result == 0.5


def test_recall_rejects_invalid_k():
    with pytest.raises(ValueError):
        recall_at_k(["A"], ["A"], k=0)


@pytest.mark.parametrize(
    "retrieved, relevant, expected",
    [
        (["A", "X"], ["A"], 1.0),
        (["X", "A"], ["A"], 0.5),
        (["X", "Y"], ["A"], 0.0),
        (["X", "X", "A"], ["A"], 0.5),
        ([], ["A"], 0.0),
    ],
)
def test_reciprocal_rank(retrieved, relevant, expected):
    result = reciprocal_rank(retrieved, relevant)

    assert result == expected


def test_evaluate_query():
    query = {
        "id": "q01",
        "relevant_book_ids": ["A", "D"],
    }
    results = [
        {"book_id": "X"},
        {"book_id": "A"},
    ]

    report = evaluate_query(query, results, k=2)

    assert report["query_id"] == "q01"
    assert report["recall_at_k"] == 0.5
    assert report["reciprocal_rank"] == 0.5
