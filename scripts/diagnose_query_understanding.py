"""Read-only diagnostics: model output, guarded plan, filters, and retrieval ablations."""

import json
import socket
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def book_results(documents, relevant):
    seen = set()
    books = []
    for document in documents:
        metadata = document.metadata
        book_id = str(metadata.get("book_id", ""))
        if not book_id or book_id in seen:
            continue
        seen.add(book_id)
        books.append({"book_id": book_id, "title": metadata.get("title", "")})
    return {
        "books": books,
        "first_positive_rank": next(
            (i for i, book in enumerate(books, 1) if book["book_id"] in relevant), None
        ),
    }


def write_summary(report: dict, path: Path):
    lines = [
        "# Query-understanding diagnosis",
        "",
        report["scope"],
        "",
        "Ranks below count unique books within the first 100 retrieved chunks.",
        "A missing positive is not proof that the entire corpus has no relevant book.",
        "",
        "| Case | Backend filter | Semantic query | Without filter rank | English probe rank |",
        "|---|---|---|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['id']} | {json.dumps(row.get('backend_filter'), ensure_ascii=False)} "
            f"| {row.get('semantic_query', '')} "
            f"| {row.get('plan_without_filters', {}).get('first_positive_rank')} "
            f"| {row.get('english_without_filters', {}).get('first_positive_rank')} |"
        )
    lines.extend(["", "## Captured plans", ""])
    for row in report["rows"]:
        lines.extend(
            [
                f"### {row['id']}",
                "",
                row["question"],
                "",
                "```json",
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "raw_model",
                            "guarded_plan",
                            "legacy_route",
                            "legacy_filters",
                            "error",
                        )
                        if key in row
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    from src.serve.rag import query_understanding as understanding
    from src.serve.rag import retriever
    from src.serve.rag.query_translator import BookPGVectorTranslator

    with socket.create_connection((retriever.PG_HOST, int(retriever.PG_PORT)), timeout=5):
        pass
    dataset = json.loads((ROOT / "tests/relevance_dataset.json").read_text("utf-8"))
    cases = [c for c in dataset["cases"] if c["type"] in {"semantic", "mixed"}]
    english = {c["id"].removesuffix("_en"): c for c in cases if c["language"] == "en"}
    cases = [c for c in cases if c["language"] == "vi" or c["id"] == "semantic_02_en"]
    vectorstore = retriever._load_vectorstore()
    model = understanding._query_model()
    report = {
        "scope": "Read-only diagnosis; English/unfiltered probes are experiments, not code changes",
        "fetch_chunks": 100,
        "rows": [],
    }
    output = ROOT / "data/reports/query_understanding_diagnosis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        row = {"id": case["id"], "question": case["question"]}
        captured = []

        class Recorder:
            def invoke(self, messages):
                message = model.invoke(messages)
                captured.append(json.loads(message.content))
                return message

        try:
            relevant = set(case["relevant_book_ids"])
            if case["language"] == "vi":
                with patch.object(understanding, "_query_model", return_value=Recorder()):
                    plan = understanding.understand_question(case["question"])
                structured = retriever._prefer_exact_title(
                    understanding.plan_to_structured_query(plan)
                )
                text, kwargs = BookPGVectorTranslator().visit_structured_query(structured)
                row.update(
                    raw_model=captured[0],
                    guarded_plan=plan.model_dump(),
                    semantic_query=text,
                    backend_filter=kwargs.get("filter"),
                )
                # Execute exactly this captured plan rather than ask the model twice.
                with patch.object(understanding, "understand_question", return_value=plan):
                    public = retriever.search(case["question"], limit=5)
                row["public_top5"] = [
                    {
                        "book_id": r["book_id"],
                        "title": r["title"],
                        "source": r["source"],
                    }
                    for r in public
                ]
                row["filtered_plan"] = book_results(
                    vectorstore.similarity_search(text, k=100, **kwargs), relevant
                )
                row["plan_without_filters"] = book_results(
                    vectorstore.similarity_search(text, k=100), relevant
                )
            else:
                row["legacy_route"] = retriever._route(case["question"])
                row["legacy_filters"] = retriever._parse_filters(case["question"])
                row["public_top5"] = retriever.search(case["question"], limit=5)
            row["original_without_filters"] = book_results(
                vectorstore.similarity_search(case["question"], k=100), relevant
            )
            counterpart = english.get(case["id"].removesuffix("_vi"), case)
            row["english_probe_question"] = counterpart["question"]
            row["english_without_filters"] = book_results(
                vectorstore.similarity_search(counterpart["question"], k=100), relevant
            )
        except Exception as exc:
            row["error"] = type(exc).__name__
            if captured:
                row["raw_model"] = captured[0]
        report["rows"].append(row)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "id": row["id"],
                    "error": row.get("error"),
                    "backend_filter": row.get("backend_filter"),
                    "semantic_query": row.get("semantic_query"),
                    "public_top5": [r["title"] for r in row.get("public_top5", [])],
                    "positive_ranks": {
                        name: row[name]["first_positive_rank"]
                        for name in (
                            "filtered_plan",
                            "plan_without_filters",
                            "original_without_filters",
                            "english_without_filters",
                        )
                        if name in row
                    },
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    print(f"Saved: {output}")
    write_summary(report, output.with_suffix(".md"))


if __name__ == "__main__":
    main()
