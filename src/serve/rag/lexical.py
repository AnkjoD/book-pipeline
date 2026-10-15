"""Small, read-only experimental corpus: SQLite FTS5/BM25 and metadata filters."""

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data/experiments/book_search.sqlite"


@contextmanager
def connect(path=DEFAULT_PATH):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("Build the experimental corpus first")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def _truth(condition: dict, metadata: dict):
    """Three-valued logic matches missing JSONB fields under NOT as well."""
    if not condition:
        return True
    if len(condition) != 1:
        raise ValueError("Expected a single filter node")
    key, value = next(iter(condition.items()))
    if key == "$not":
        child = _truth(value, metadata)
        return None if child is None else not child
    if key in {"$and", "$or"}:
        children = [_truth(node, metadata) for node in value]
        if not children:
            raise ValueError("Empty logical filter")
        if key == "$and":
            return False if False in children else None if None in children else True
        return True if True in children else None if None in children else False
    if len(value) != 1:
        raise ValueError("Expected one comparator")
    operator, expected = next(iter(value.items()))
    if operator not in {"$eq", "$ne", "$lt", "$lte", "$gt", "$gte", "$ilike"}:
        raise ValueError("Unsupported experimental filter")
    actual = metadata.get(key)
    if actual is None:
        return None
    if operator == "$ilike":
        pattern, escaped = "", False
        for char in expected:
            if escaped:
                pattern += re.escape(char)
                escaped = False
            elif char == "\\":
                escaped = True
            else:
                pattern += ".*" if char == "%" else "." if char == "_" else re.escape(char)
        return bool(re.fullmatch(pattern, str(actual), re.IGNORECASE | re.DOTALL))
    return {
        "$eq": lambda: actual == expected,
        "$ne": lambda: actual != expected,
        "$lt": lambda: actual < expected,
        "$lte": lambda: actual <= expected,
        "$gt": lambda: actual > expected,
        "$gte": lambda: actual >= expected,
    }[operator]()


def matches_filter(condition: dict | None, metadata: dict) -> bool:
    return _truth(condition or {}, metadata) is True


def row_to_result(row, source="lexical", score=None):
    metadata = json.loads(row["metadata"])
    text = row["document_text"]
    # Return an actual excerpt, no HTML highlighting or synthetic facts.
    description = text.partition("Description: ")[2] or text
    return {
        "book_id": row["book_id"],
        "title": row["title"],
        "author_names": metadata.get("author_names", ""),
        "evidence": description[:1500],
        "source": source,
        "score": score,
        "distance": None,
        "chunk_index": None,
        "metadata": metadata,
    }


def lexical_search(question, limit=20, filters=None, path=DEFAULT_PATH):
    if limit < 1:
        raise ValueError("limit must be positive")
    tokens = list(dict.fromkeys(re.findall(r"\w+", question.casefold())))[:48]
    if not tokens:
        return []
    expression = " OR ".join('"' + token + '"' for token in tokens)
    with connect(path) as connection:
        rows = connection.execute(
            "SELECT b.*, bm25(book_fts, 0.0, 3.0, 1.0) AS rank "
            "FROM book_fts JOIN books b ON book_fts.book_id=b.book_id "
            "WHERE book_fts MATCH ? ORDER BY rank, b.book_id",
            [expression],
        ).fetchall()
    return [
        row_to_result(row, score=-row["rank"])
        for row in rows
        if matches_filter(filters, json.loads(row["metadata"]))
    ][:limit]
