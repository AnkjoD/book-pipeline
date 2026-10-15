"""Check that diagnostic scoring catches dropped filters and boundary errors."""

from types import SimpleNamespace

from scripts.evaluate_vietnamese import ast_matches, compare, current_matches
from src.serve.rag.retriever import _parse_filters


def test_scoring_distinguishes_strict_boundary_and_decimal():
    assert not compare(300, "lt", 300)
    assert compare(300, "lte", 300)
    assert not compare(4.1, "gt", 4.2)
    assert compare(4.3, "gt", 4.2)


def test_scoring_detects_missing_filter():
    metadata = {
        "title": "A",
        "author_names": "B",
        "genres": "Horror",
        "pages_clean": 301,
        "rating": 4.3,
        "release_year_clean": 2020,
    }
    filters = _parse_filters("sách kinh dị dưới 300 trang")
    assert not current_matches(filters, metadata)
    # A parser that drops pages must be scored differently, even if genre is right.
    filters["max_pages"] = None
    assert current_matches(filters, metadata)


def test_scoring_preserves_negation():
    comparison = SimpleNamespace(
        attribute="genres", comparator=SimpleNamespace(value="contain"), value="horror"
    )
    negation = SimpleNamespace(operator=SimpleNamespace(value="not"), arguments=[comparison])
    assert not ast_matches(negation, {"genres": "Horror, Fiction"})
    assert ast_matches(negation, {"genres": "Mystery, Fiction"})
