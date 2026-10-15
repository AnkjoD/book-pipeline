"""Run labeled development queries through public search(), save JSON and Markdown."""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.serve.rag.evaluation import evaluate_query, grouped_summary, summarize  # noqa: E402


def audit_sources(dataset: dict) -> dict:
    """Verify frozen label hashes and report the actual vector corpus, read-only."""
    import duckdb
    import psycopg

    from src.serve.rag import retriever

    snapshots = dataset["book_snapshots"]
    with duckdb.connect(retriever.DB_PATH, read_only=True) as connection:
        actual = dict(
            connection.execute(
                "SELECT cast(book_id AS varchar), document_hash FROM rag_documents "
                "WHERE cast(book_id AS varchar) IN (SELECT unnest(?))",
                [list(snapshots)],
            ).fetchall()
        )
        if any(
            actual.get(book_id) != snapshot["document_hash"]
            for book_id, snapshot in snapshots.items()
        ):
            raise ValueError("Gold label snapshots changed; review and rebuild dataset")
    try:
        with psycopg.connect(
            host=retriever.PG_HOST,
            port=retriever.PG_PORT,
            dbname=retriever.PG_DB,
            user=retriever.PG_USER,
            password=retriever.PG_PASSWORD,
            connect_timeout=5,
        ) as connection:
            indexed = connection.execute(
                "SELECT e.cmetadata->>'book_id', e.cmetadata->>'document_hash' "
                "FROM langchain_pg_embedding e JOIN langchain_pg_collection c "
                "ON e.collection_id=c.uuid WHERE c.name=%s",
                [retriever.COLLECTION],
            ).fetchall()
    except psycopg.Error:
        raise RuntimeError("Vector database unavailable; start postgres-vector first") from None
    book_ids = {row[0] for row in indexed}
    if not book_ids:
        raise ValueError("Vector collection is empty")
    mismatches = {
        book_id
        for book_id, document_hash in indexed
        if book_id in snapshots and document_hash != snapshots[book_id]["document_hash"]
    }
    if mismatches:
        raise ValueError("Vector label documents differ from Gold; review corpus before scoring")
    return {
        "gold_snapshot_check": "passed",
        "vector_books": len(book_ids),
        "vector_chunks": len(indexed),
        "vector_book_ids_sha256": hashlib.sha256("\n".join(sorted(book_ids)).encode()).hexdigest(),
        "indexed_positive_counts": {
            case["id"]: len(set(case["relevant_book_ids"]) & book_ids)
            for case in dataset["cases"]
            if case["relevant_book_ids"]
        },
    }


def validate_dataset(dataset: dict):
    ids = set()
    for case in dataset["cases"]:
        if case["id"] in ids:
            raise ValueError(f"Duplicate query ID: {case['id']}")
        ids.add(case["id"])
        if not isinstance(case["question"], str):
            raise ValueError("Questions must be strings")
        # Reuse metric validation without executing any search.
        evaluate_query(case, [], k=1)
    if not ids:
        raise ValueError("Dataset is empty")


def run_evaluation(dataset: dict, search_fn, k: int, progress=None) -> list[dict]:
    """Search sees only question and limit; ground truth stays with evaluator."""
    validate_dataset(dataset)
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    rows = []
    for case in dataset["cases"]:
        start = time.perf_counter()
        try:
            results = search_fn(case["question"], limit=k)
            row = evaluate_query(case, results, k)
            row["routes"] = sorted({item.get("source", "unknown") for item in results})
            row["results"] = [
                {key: item.get(key) for key in ("book_id", "title", "source")} for item in results
            ]
        except Exception as exc:
            # Never count an unavailable service as a successful empty answer.
            row = {
                "query_id": case["id"],
                "type": case["type"],
                "language": case.get("language", "unspecified"),
                "label_scope": case.get("label_scope", "unspecified"),
                "error": type(exc).__name__,
                "empty_correct": False,
                "recall_at_k": None,
                "reciprocal_rank": None,
            }
        row["question"] = case["question"]
        row["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        rows.append(row)
        if progress:
            progress(row)
    return rows


def write_reports(report: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Relevance development report",
        "",
        report["scope"],
        "",
        "Semantic labels are partial, agent-authored, and not human reviewed.",
        "Scores describe this development set, not all user queries.",
        "",
        "```json",
        json.dumps(report["summary"], indent=2),
        "```",
        "",
        "## Scores by query type and language",
        "",
        "| Group | Cases | Hit@k | MRR@k | Errors |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, group in report.get("by_type_and_language", {}).items():
        lines.append(
            f"| {name} | {group['cases']} | {group['hit_rate_at_k']} "
            f"| {group['mrr_at_k']} | {group['errors']} |"
        )
    lines.extend(
        [
            "",
            "## Individual queries",
            "",
            "| Query | Type | Recall@k | RR@k | Error |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in report["rows"]:
        lines.append(
            f"| {row['query_id']} | {row['type']} | {row.get('recall_at_k')} "
            f"| {row.get('reciprocal_rank')} | {row.get('error', '')} |"
        )
    lines.extend(["", "## Limitations", ""] + [f"- {s}" for s in report["limitations"]])
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "tests/relevance_dataset.json")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/reports/relevance_baseline.json"
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--types", nargs="+", choices=["exact", "semantic", "mixed", "unanswerable"]
    )
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    raw = args.dataset.read_bytes()
    dataset = json.loads(raw)
    if args.types:
        dataset["cases"] = [case for case in dataset["cases"] if case["type"] in args.types]
    validate_dataset(dataset)
    if args.k <= 0:
        parser.error("--k must be positive")

    from src.serve.rag.retriever import COLLECTION, EMBED_MODEL, _load_vectorstore, search

    corpus_audit = audit_sources(dataset)
    warmup = {"status": "skipped", "latency_ms": None}
    if not args.no_warmup:
        start = time.perf_counter()
        try:
            _load_vectorstore().similarity_search("book", k=1)
            warmup["status"] = "ok"
        except Exception as exc:
            warmup.update(status="error", error=type(exc).__name__)
        warmup["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)

    def progress(row):
        print(
            f"{row['query_id']}: recall={row.get('recall_at_k')} "
            f"rr={row.get('reciprocal_rank')} error={row.get('error', '-')} "
            f"({row['latency_ms']}ms)",
            flush=True,
        )

    rows = run_evaluation(dataset, search, args.k, progress)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": dataset["scope"],
        "limitations": dataset["limitations"],
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "k": args.k,
        "collection": COLLECTION,
        "embedding_model": EMBED_MODEL,
        "gold_rows_at_labeling": dataset["gold_rows"],
        "corpus_audit": corpus_audit,
        "warmup": warmup,
        "mean_query_latency_ms": mean(row["latency_ms"] for row in rows),
        "query_model_note": "Query model first-call startup may be included in query latency",
        "summary": summarize(rows),
        "by_type": grouped_summary(rows, "type"),
        "by_language": grouped_summary(rows, "language"),
        "by_type_and_language": grouped_summary(
            [{**row, "group": row["type"] + "/" + row["language"]} for row in rows], "group"
        ),
        "rows": rows,
    }
    write_reports(report, args.output)
    print(json.dumps(report["summary"], indent=2))
    print(f"Reports: {args.output} and {args.output.with_suffix('.md')}")
    if report["summary"]["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
