"""Build a separate 300-book lexical/graph artifact; never modify pgvector."""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENTITY_SCHEMA = {
    "type": "object",
    "required": ["entities"],
    "additionalProperties": False,
    "properties": {
        "entities": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "kind", "quote"],
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": ["character", "place"]},
                    "quote": {"type": "string"},
                },
            },
        }
    },
}


def create_artifact(path, books, enrichments, manifest):
    """Build privately, then atomically publish the complete single-file artifact."""
    from src.serve.rag.query_understanding import normalize_question

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".building-" + uuid.uuid4().hex)
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript("""
            CREATE TABLE manifest (data TEXT NOT NULL);
            CREATE TABLE books (book_id TEXT PRIMARY KEY, title TEXT NOT NULL,
                                document_text TEXT NOT NULL, metadata TEXT NOT NULL);
            CREATE VIRTUAL TABLE book_fts USING fts5(book_id UNINDEXED, title, body,
                                                     tokenize='unicode61 remove_diacritics 2');
            CREATE TABLE edges (book_id TEXT NOT NULL, node_id TEXT NOT NULL,
                                name TEXT NOT NULL, kind TEXT NOT NULL, quote TEXT NOT NULL,
                                document_hash TEXT NOT NULL, origin TEXT NOT NULL,
                                PRIMARY KEY(book_id,node_id));
        """)
        for book in books:
            book_id = str(book["book_id"])
            metadata = {k: v for k, v in book.items() if k != "document_text"}
            metadata["book_id"] = book_id
            connection.execute(
                "INSERT INTO books VALUES (?,?,?,?)",
                [
                    book_id,
                    book["title"],
                    book["document_text"],
                    json.dumps(metadata),
                ],
            )
            connection.execute(
                "INSERT INTO book_fts VALUES (?,?,?)",
                [
                    book_id,
                    book["title"],
                    book["document_text"],
                ],
            )
            nodes = [{"name": book["title"], "kind": "title", "quote": book["title"]}]
            for name in (book.get("author_names") or "").split(","):
                if name.strip():
                    nodes.append({"name": name.strip(), "kind": "credit", "quote": name.strip()})
            # A series label is a metadata assertion; missing series stays missing.
            series = re.sub(r"\s+#\d+(?:\.\d+)?", "", book.get("series_info") or "").strip()
            if series:
                nodes.append({"name": series, "kind": "series", "quote": book["series_info"]})
            for node in [*nodes, *enrichments.get(book_id, [])]:
                node_id = node["kind"] + ":" + normalize_question(node["name"])
                connection.execute(
                    "INSERT OR IGNORE INTO edges VALUES (?,?,?,?,?,?,?)",
                    [
                        book_id,
                        node_id,
                        node["name"],
                        node["kind"],
                        node["quote"],
                        book["document_hash"],
                        "metadata" if node in nodes else "llm_literal_checked",
                    ],
                )
        manifest["edges"] = connection.execute("SELECT count(*) FROM edges").fetchone()[0]
        manifest["nodes"] = connection.execute(
            "SELECT count(distinct node_id) FROM edges"
        ).fetchone()[0]
        connection.execute("INSERT INTO manifest VALUES (?)", [json.dumps(manifest)])
        connection.commit()
    finally:
        connection.close()
    os.replace(temporary, path)
    return manifest


def main():
    import duckdb
    import psycopg
    import requests

    from src.serve.rag import retriever
    from src.serve.rag.graph_retriever import grounded_entities
    from src.serve.rag.lexical import DEFAULT_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books", type=int, default=300)
    parser.add_argument("--extract-books", type=int, default=24)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    if not 1 <= args.books <= 1000 or not 0 <= args.extract_books <= args.books:
        parser.error("Use 1..1000 books and 0..books extractions")
    with psycopg.connect(
        host=retriever.PG_HOST,
        port=retriever.PG_PORT,
        dbname=retriever.PG_DB,
        user=retriever.PG_USER,
        password=retriever.PG_PASSWORD,
        connect_timeout=5,
    ) as connection:
        indexed = connection.execute(
            "SELECT DISTINCT e.cmetadata->>'book_id',e.cmetadata->>'document_hash' "
            "FROM langchain_pg_embedding e JOIN langchain_pg_collection c "
            "ON e.collection_id=c.uuid WHERE c.name=%s",
            [retriever.COLLECTION],
        ).fetchall()
    hashes = dict(indexed)
    with duckdb.connect(retriever.DB_PATH, read_only=True) as connection:
        result = connection.execute(
            "SELECT cast(book_id as varchar) AS book_id,title,author_names,genres,moods,"
            "series_info,rating,ratings_count,pages_clean,release_year_clean,document_hash,"
            "document_text FROM rag_documents "
            "WHERE cast(book_id as varchar) IN (SELECT unnest(?)) "
            "ORDER BY ratings_count DESC NULLS LAST,book_id LIMIT ?",
            [list(hashes), args.books],
        )
        columns = [c[0] for c in result.description]
        books = [dict(zip(columns, row)) for row in result.fetchall()]
    if not books or any(hashes[b["book_id"]] != b["document_hash"] for b in books):
        raise ValueError("Empty corpus or Gold/vector hash mismatch")
    cache_path = ROOT / "data/experiments/graph_entity_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text("utf-8")) if cache_path.exists() else {}
    model = os.getenv("QUERY_MODEL", "qwen2.5:7b-instruct-q4_K_M")
    enrichments, failures, rejected = {}, [], 0
    for book in books[: args.extract_books]:
        book_id = book["book_id"]
        key = f"v1:{model}:{book_id}:{book['document_hash']}"
        description = book["document_text"].partition("Description: ")[2][:4000]
        if not description:
            continue
        try:
            if key not in cache:
                response = requests.post(
                    os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/") + "/api/chat",
                    json={
                        "model": model,
                        "stream": False,
                        "format": ENTITY_SCHEMA,
                        "options": {"temperature": 0, "num_predict": 1024},
                        "messages": [
                            {
                                "role": "system",
                                "content": "Extract named fictional characters and places ONLY "
                                "from the provided description. Copy names and quotes verbatim. "
                                "Do not infer from outside knowledge. Return entities=[] if none.",
                            },
                            {"role": "user", "content": description},
                        ],
                    },
                    timeout=90,
                )
                response.raise_for_status()
                data = response.json()
                if data.get("error"):
                    raise ValueError("Ollama error")
                cache[key] = json.loads(data["message"]["content"])
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), "utf-8")
            accepted, invalid = grounded_entities(cache[key], description)
            enrichments[book_id] = accepted
            rejected += invalid
            print(
                f"{book['title']}: {len(accepted)} supported entities, {invalid} rejected",
                flush=True,
            )
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            failures.append({"book_id": book_id, "error": type(exc).__name__})
    fingerprint = hashlib.sha256(
        "\n".join(sorted(b["book_id"] + ":" + b["document_hash"] for b in books)).encode()
    ).hexdigest()
    manifest = create_artifact(
        args.output,
        books,
        enrichments,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "books": len(books),
            "corpus_sha256": fingerprint,
            "collection": retriever.COLLECTION,
            "extraction_model": model,
            "extraction_attempted_books": args.extract_books,
            "extraction_completed_books": len(enrichments),
            "enriched_books": sum(bool(entities) for entities in enrichments.values()),
            "rejected_entities": rejected,
            "extraction_errors": failures,
            "graph_scope": "local graph, no community summaries",
        },
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
