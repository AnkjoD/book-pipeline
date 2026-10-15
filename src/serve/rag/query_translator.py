"""Translate LangChain structured queries to our PGVector JSONB filters.

Our textual metadata fields are strings, so CONTAIN means literal substring,
not array membership. This adapter does not repair an incorrectly parsed query.
"""

import math

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)

COMPARATOR_MAP = {name: "$" + name for name in ("eq", "ne", "lt", "lte", "gt", "gte")}
TEXT_FIELDS = {"book_id", "title", "author_names", "genres", "moods", "document_hash"}
INTEGER_FIELDS = {"pages_clean", "release_year_clean", "ratings_count", "chunk_index"}
ALLOWED_FIELDS = TEXT_FIELDS | INTEGER_FIELDS | {"rating"}


def translate_comparison(attribute: str, comparator: str, value) -> dict:
    """Validate one comparison, then return a backend filter without dropping it."""
    if attribute not in ALLOWED_FIELDS:
        raise ValueError(f"Không hỗ trợ trường: {attribute}")
    if comparator not in COMPARATOR_MAP and comparator != "contain":
        raise ValueError(f"Không hỗ trợ toán tử: {comparator}")

    if attribute in TEXT_FIELDS:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Bộ lọc văn bản cần chuỗi không rỗng")
        if comparator not in {"eq", "ne", "contain"}:
            raise ValueError("Trường văn bản chỉ hỗ trợ eq, ne hoặc contain")
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Bộ lọc số cần giá trị số, không nhận boolean hoặc chuỗi")
        if not math.isfinite(value):
            raise ValueError("Giá trị số phải hữu hạn")
        if attribute in INTEGER_FIELDS and not isinstance(value, int):
            raise ValueError("Trường số nguyên cần giá trị int")
        if comparator == "contain":
            raise ValueError("Không dùng contain cho trường số")

    if comparator == "contain":
        # Escape LIKE wildcards: user's % and _ are literal characters.
        escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return {attribute: {"$ilike": "%" + escaped + "%"}}
    return {attribute: {COMPARATOR_MAP[comparator]: value}}


class BookPGVectorTranslator(Visitor):
    """Visitor for the operators accepted by the installed PGVector backend."""

    allowed_comparators = [
        Comparator.EQ,
        Comparator.NE,
        Comparator.LT,
        Comparator.LTE,
        Comparator.GT,
        Comparator.GTE,
        Comparator.CONTAIN,
    ]
    allowed_operators = [Operator.AND, Operator.OR, Operator.NOT]

    def visit_comparison(self, comparison: Comparison) -> dict:
        self._validate_func(comparison.comparator)
        return translate_comparison(
            comparison.attribute, comparison.comparator.value, comparison.value
        )

    def visit_operation(self, operation: Operation) -> dict:
        self._validate_func(operation.operator)
        if not operation.arguments:
            raise ValueError("Điều kiện logic không được rỗng")
        if operation.operator == Operator.NOT:
            if len(operation.arguments) != 1:
                raise ValueError("NOT chỉ nhận đúng một điều kiện con")
            return {"$not": operation.arguments[0].accept(self)}
        return {"$" + operation.operator.value: [a.accept(self) for a in operation.arguments]}

    def visit_structured_query(self, structured_query: StructuredQuery) -> tuple[str, dict]:
        kwargs = {}
        if structured_query.filter is not None:
            kwargs["filter"] = structured_query.filter.accept(self)
        return structured_query.query, kwargs
