"""The adapter must preserve comparisons and reject unsupported inputs."""

import pytest
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)

from src.serve.rag.query_translator import BookPGVectorTranslator, translate_comparison


@pytest.mark.parametrize("operator", ["eq", "ne", "lt", "lte", "gt", "gte"])
def test_numeric_comparison(operator):
    assert translate_comparison("pages_clean", operator, 300) == {
        "pages_clean": {"$" + operator: 300}
    }


def test_contain_is_case_insensitive_literal_substring():
    assert translate_comparison("genres", "contain", "Horror") == {"genres": {"$ilike": "%Horror%"}}
    assert translate_comparison("title", "contain", "100%_book") == {
        "title": {"$ilike": "%100\\%\\_book%"}
    }


@pytest.mark.parametrize(
    "field,operator,value",
    [
        ("unknown", "eq", 3),
        ("pages_clean", "like", 3),
        ("pages_clean", "lt", True),
        ("pages_clean", "lt", "300"),
        ("pages_clean", "lte", 300.5),
        ("rating", "gte", float("nan")),
        ("rating", "lt", float("inf")),
        ("rating", "contain", 4),
        ("genres", "contain", ""),
        ("genres", "lt", "horror"),
    ],
)
def test_invalid_filter_is_not_silently_dropped(field, operator, value):
    with pytest.raises(ValueError):
        translate_comparison(field, operator, value)


def test_nested_not_and_lte():
    condition = Operation(
        operator=Operator.AND,
        arguments=[
            Operation(
                operator=Operator.NOT,
                arguments=[
                    Comparison(attribute="genres", comparator=Comparator.CONTAIN, value="horror")
                ],
            ),
            Comparison(attribute="pages_clean", comparator=Comparator.LTE, value=300),
        ],
    )
    plan = StructuredQuery(query="memory loss", filter=condition)
    assert plan.accept(BookPGVectorTranslator()) == (
        "memory loss",
        {
            "filter": {
                "$and": [
                    {"$not": {"genres": {"$ilike": "%horror%"}}},
                    {"pages_clean": {"$lte": 300}},
                ]
            }
        },
    )


def test_or_and_no_filter():
    translator = BookPGVectorTranslator()
    condition = Operation(
        operator=Operator.OR,
        arguments=[
            Comparison(attribute="rating", comparator=Comparator.GT, value=4.2),
            Comparison(attribute="pages_clean", comparator=Comparator.LT, value=300),
        ],
    )
    assert condition.accept(translator) == {
        "$or": [
            {"rating": {"$gt": 4.2}},
            {"pages_clean": {"$lt": 300}},
        ]
    }
    assert StructuredQuery(query="dragons", filter=None).accept(translator) == ("dragons", {})


def test_not_rejects_multiple_children():
    child = Comparison(attribute="rating", comparator=Comparator.GT, value=4)
    with pytest.raises(ValueError):
        Operation(operator=Operator.NOT, arguments=[child, child]).accept(BookPGVectorTranslator())
