"""Local graph retrieval with evidence-backed edges; not Microsoft global GraphRAG."""

import json
import re

from .lexical import DEFAULT_PATH, connect, matches_filter, row_to_result
from .query_understanding import normalize_question


def relation_request(question):
    quoted = re.search(r'"([^"]+)"', question)
    normalized = normalize_question(question)
    if not quoted:
        return None
    if re.search(r"same series|cung (?:bo|series)", normalized):
        return "series", quoted.group(1)
    if re.search(r"same credited person|shared credits|cung nguoi duoc ghi cong", normalized):
        return "credit", quoted.group(1)
    return None


def grounded_entities(payload, description):
    """Literal support is necessary, but not proof that an entity's type is correct."""
    accepted, rejected = [], 0
    if not isinstance(payload, dict) or not isinstance(payload.get("entities"), list):
        raise ValueError("Expected an entities list")
    for entity in payload.get("entities", [])[:16]:
        if not isinstance(entity, dict):
            rejected += 1
            continue
        name, quote = entity.get("name", "").strip(), entity.get("quote", "").strip()
        if (
            entity.get("kind") not in {"character", "place"}
            or len(name) < 3
            or not quote
            or quote.casefold() not in description.casefold()
            or name.casefold() not in quote.casefold()
        ):
            rejected += 1
            continue
        accepted.append({"name": name, "kind": entity["kind"], "quote": quote})
    return accepted, rejected


def graph_search(question, limit=20, filters=None, path=DEFAULT_PATH):
    if limit < 1:
        raise ValueError("limit must be positive")
    normalized = normalize_question(question)
    with connect(path) as connection:
        books = {r["book_id"]: r for r in connection.execute("SELECT * FROM books")}
        edges = [
            dict(r) for r in connection.execute("SELECT * FROM edges ORDER BY book_id,node_id")
        ]
    by_node, by_book = {}, {}
    for edge in edges:
        by_node.setdefault(edge["node_id"], []).append(edge)
        by_book.setdefault(edge["book_id"], []).append(edge)
    scores, paths = {}, {}
    roots = {}
    relation = relation_request(question)
    for edge in edges:
        name = normalize_question(edge["name"])
        if relation and (edge["kind"] != "title" or name != normalize_question(relation[1])):
            continue
        if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", normalized):
            roots[edge["node_id"]] = edge
    for node_id, root in roots.items():
        # Kind labels are explicit; no statement that a credited person is main author.
        weight = 4 if root["kind"] in {"character", "place", "title"} else 2
        for direct in by_node[node_id]:
            book_id = direct["book_id"]
            scores[book_id] = scores.get(book_id, 0) + weight
            paths.setdefault(book_id, []).append([direct])
            # One bounded expansion through shared credit or series, not generic genres.
            for bridge in by_book[book_id]:
                if bridge["kind"] not in {"credit", "series"} or bridge["node_id"] == node_id:
                    continue
                if relation and bridge["kind"] != relation[0]:
                    continue
                for other in by_node[bridge["node_id"]][:25]:
                    target = other["book_id"]
                    if target == book_id:
                        continue
                    scores[target] = scores.get(target, 0) + weight / 4
                    paths.setdefault(target, []).append([direct, bridge, other])
    results = []
    for book_id in sorted(scores, key=lambda key: (-scores[key], key)):
        row = books[book_id]
        if matches_filter(filters, json.loads(row["metadata"])):
            result = row_to_result(row, source="graph", score=scores[book_id])
            result["graph_paths"] = paths[book_id][:3]
            results.append(result)
        if len(results) >= limit:
            break
    return results
