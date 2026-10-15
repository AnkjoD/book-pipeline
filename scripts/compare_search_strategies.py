"""Compare dense, BM25+RRF and graph+dense on exactly the same artifact and plans."""

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def relation_cases(books):
    """Labels from metadata tables, never graph search output. SQL-equivalent control."""
    cases = []
    for title, field, kind in [
        ("The Hunger Games", "series_info", "series"),
        ("Catching Fire", "series_info", "series"),
        ("Dune", "author_names", "credit"),
        ("The Martian", "author_names", "credit"),
    ]:
        anchors = {key: book for key, book in books.items() if book["title"] == title}
        if not anchors:
            continue
        if kind == "series":
            # This benchmark's anchors have one series, inspect its stored label.
            values = {b[field].split(" #")[0] for b in anchors.values() if b.get(field)}
            ids = [
                key
                for key, b in books.items()
                if key not in anchors and b.get(field, "").split(" #")[0] in values
            ]
            question = f'other books in the same series as "{title}"'
        else:
            values = {
                name.strip()
                for b in anchors.values()
                for name in b.get(field, "").split(",")
                if name.strip()
            }
            ids = [
                key
                for key, b in books.items()
                if key not in anchors
                and values.intersection(name.strip() for name in b.get(field, "").split(","))
            ]
            question = f'other books with the same credited person as "{title}"'
        if ids:
            cases.append(
                {
                    "id": "relation_" + kind + "_" + title.replace(" ", "_"),
                    "question": question,
                    "relevant_book_ids": ids,
                    "type": "relation",
                    "language": "en",
                    "split": "dev",
                    "label_scope": "complete_structured_metadata_within_corpus",
                    "metadata_control_ids": sorted(ids),
                }
            )
    return cases


def main():
    from scripts.evaluate_relevance import validate_dataset
    from src.serve.rag.evaluation import evaluate_query, grouped_summary, summarize
    from src.serve.rag.experimental import (
        load_corpus,
        prepare_query,
        search_experiment,
        search_prepared,
    )
    from src.serve.rag.graph_retriever import graph_search

    manifest, books = load_corpus()
    dataset_path = ROOT / "tests/relevance_dataset.json"
    original = json.loads(dataset_path.read_text("utf-8"))
    cases, excluded = [], []
    for source in original["cases"]:
        if source.get("expected_empty"):
            cases.append(source)
            continue
        positives = sorted(set(source["relevant_book_ids"]) & books.keys())
        if not positives:
            excluded.append(source["id"])
            continue
        cases.append({**source, "relevant_book_ids": positives})
    cases += relation_cases(books)
    validate_dataset({"cases": cases})
    scoped = ROOT / "data/reports/search_experiment_cases.json"
    scoped.parent.mkdir(parents=True, exist_ok=True)
    scoped.write_text(
        json.dumps(
            {"cases": cases, "corpus_sha256": manifest["corpus_sha256"]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    reports = {mode: [] for mode in ["dense", "hybrid", "graph"]}
    details = []
    destination = ROOT / "data/reports/search_strategy_comparison.json"
    for case in cases:
        detail = {"id": case["id"], "question": case["question"], "arms": {}}
        if case.get("expected_empty"):
            # Independently execute public experiment parsing; never synthesize empty results.
            try:
                text, filters = prepare_query(case["question"])
            except Exception as exc:
                # Clarification/error is reported as error, not successful abstention.
                detail["prepare_error"] = type(exc).__name__
                text, filters = None, None
        else:
            try:
                text, filters = prepare_query(case["question"])
            except Exception as exc:
                detail["prepare_error"] = type(exc).__name__
                text, filters = None, None
        detail.update(semantic_query=text, backend_filter=filters)
        if text is None and "prepare_error" not in detail:
            for mode in reports:
                row = evaluate_query(case, search_experiment(case["question"], mode=mode), 5)
                reports[mode].append(row)
                detail["arms"][mode] = row
            details.append(detail)
            continue
        pool = None
        if text is not None:
            try:
                pool = search_prepared(text, filters, mode="dense", limit=30)
            except Exception as exc:
                detail["prepare_error"] = type(exc).__name__
        for mode in reports:
            start = time.perf_counter()
            if pool is None:
                row = {
                    "query_id": case["id"],
                    "type": case["type"],
                    "language": case["language"],
                    "error": detail["prepare_error"],
                }
            else:
                try:
                    results = search_prepared(text, filters, mode=mode, limit=5, dense_results=pool)
                    row = evaluate_query(case, results, 5)
                    row["results"] = [
                        {
                            "book_id": r["book_id"],
                            "title": r["title"],
                            "graph_paths": r.get("graph_paths", []),
                        }
                        for r in results
                    ]
                except Exception as exc:
                    row = {
                        "query_id": case["id"],
                        "type": case["type"],
                        "language": case["language"],
                        "error": type(exc).__name__,
                    }
            row["additional_ms_after_shared_dense_pool"] = round(
                (time.perf_counter() - start) * 1000, 2
            )
            reports[mode].append(row)
            detail["arms"][mode] = row
        if text is not None:
            direct = graph_search(text, filters=filters)
            detail["graph_only_ids"] = [r["book_id"] for r in direct]
        details.append(detail)
        print(
            case["id"],
            {mode: detail["arms"][mode].get("reciprocal_rank") for mode in reports},
            flush=True,
        )
    report = {
        "scope": "Same-corpus development comparison, no held-out quality claim",
        "manifest": manifest,
        "cases": len(cases),
        "excluded_outside_corpus": excluded,
        "dataset_sha256": hashlib.sha256(scoped.read_bytes()).hexdigest(),
        "summaries": {mode: summarize(rows) for mode, rows in reports.items()},
        "by_type": {mode: grouped_summary(rows, "type") for mode, rows in reports.items()},
        "metadata_relation_control": [
            {"id": c["id"], "matching_ids": c["metadata_control_ids"]}
            for c in cases
            if c["type"] == "relation"
        ],
        "limitations": [
            "Dev labels are partial and not human-reviewed; no held-out split",
            "Entity type/relationship truth is not guaranteed by literal quote checks",
            "Only 24 descriptions enriched; graph coverage is intentionally partial",
            "Relation cases can also be solved by structured joins; graph is not required",
            "Shared dense pool time excluded from arm overhead, not end-to-end latency",
            "Local graph+dense only, no Microsoft GraphRAG community/global search",
        ],
        "rows": details,
    }
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Same-corpus search experiment",
        "",
        report["scope"],
        "",
        "| Mode | Cases | Errors | Hit@5 | MRR@5 | Recall@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, summary in report["summaries"].items():
        lines.append(
            f"| {mode} | {summary['cases']} | {summary['errors']} "
            f"| {summary['hit_rate_at_k']} | {summary['mrr_at_k']} "
            f"| {summary['mean_recall_at_k']} |"
        )
    lines.extend(["", "## Limitations", ""] + ["- " + s for s in report["limitations"]])
    destination.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report["summaries"], indent=2))


if __name__ == "__main__":
    main()
