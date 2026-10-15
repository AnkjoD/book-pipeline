"""Run real Vietnamese/English vector probes on the existing pgvector collection.

Anchor hit means a known title appears, not general semantic relevance or recall.
The script never rebuilds the index or changes chatbot routing.
"""

import json
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.serve.rag import retriever  # noqa: E402
from src.serve.rag.pipeline import run_rag_pipeline  # noqa: E402

GROUPS = [
    {
        "anchor": "hobbit",
        "queries": {
            "en": "What is The Hobbit about?",
            "vi": "Cuốn The Hobbit kể về điều gì?",
            "vi_no_accents": "cuon The Hobbit ke ve dieu gi",
            "vi_typo": "cuon The Hobit ke ve dieu gi",
        },
    },
    {
        "anchor": "dune",
        "queries": {
            "en": "Dune by Frank Herbert",
            "vi": "Dune của Frank Herbert",
            "vi_no_accents": "sach Dune cua Frank Herbert",
            "vi_typo": "sach Duen cua Frank Herbert",
        },
    },
    {
        "anchor": "harry potter",
        "queries": {
            "en": "Harry Potter and the school of magic",
            "vi": "Harry Potter và ngôi trường phép thuật",
            "vi_no_accents": "Harry Potter va ngoi truong phep thuat",
            "vi_typo": "Hary Poter va ngoi truong phep thuat",
        },
    },
]


def save(report):
    path = ROOT / "data/reports/vietnamese_vector.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    report = {"scope": "live vector anchor-hit probes, not semantic Recall@k", "rows": []}
    try:
        with socket.create_connection((retriever.PG_HOST, int(retriever.PG_PORT)), timeout=3):
            pass
    except OSError:
        report["blocked"] = "pgvector endpoint unavailable"
        save(report)
        print(report["blocked"], flush=True)
        return 2

    start = time.perf_counter()
    print("Loading cached embedding model...", flush=True)
    retriever._load_vectorstore()
    report["warmup_seconds"] = round(time.perf_counter() - start, 2)
    for group in GROUPS:
        for language, question in group["queries"].items():
            start = time.perf_counter()
            try:
                # Force vector to isolate it from the existing SQL/rule router.
                results = retriever._vector_search(question, limit=3)
                row = {
                    "language": language,
                    "question": question,
                    "anchor": group["anchor"],
                    "anchor_hit": any(group["anchor"] in r["title"].casefold() for r in results),
                    "results": [
                        {
                            "book_id": r["book_id"],
                            "title": r["title"],
                            "distance": r["distance"],
                            "source": r["source"],
                        }
                        for r in results
                    ],
                }
            except Exception as exc:
                row = {
                    "language": language,
                    "question": question,
                    "error": type(exc).__name__,
                    "anchor_hit": False,
                }
            row["seconds"] = round(time.perf_counter() - start, 2)
            report["rows"].append(row)
            save(report)
            print(json.dumps(row, ensure_ascii=True), flush=True)
    report["pipeline_examples"] = []
    for question in [
        "Dune của Frank Herbert",
        "Cuốn The Hobbit kể về điều gì?",
        "sách xuất bản năm 9999",
    ]:
        result = run_rag_pipeline(question, limit=3)
        report["pipeline_examples"].append({"question": question, **result})
        save(report)
        print(
            json.dumps(
                {
                    "question": question,
                    "status": result["verification_status"],
                    "answer": result["answer"],
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
    report["summary"] = {
        language: {
            "hit": sum(r["anchor_hit"] for r in report["rows"] if r["language"] == language),
            "total": len(GROUPS),
        }
        for language in ["en", "vi", "vi_no_accents", "vi_typo"]
    }
    report["self_query_examples"] = self_query_examples()
    save(report)
    print(json.dumps(report["summary"]), flush=True)
    return 1 if any(r.get("error") for r in report["rows"]) else 0


def self_query_examples():
    """Exercise the actual installed SelfQueryRetriever, not just its constructor."""
    from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
    from langchain_core.runnables import RunnableLambda

    from scripts.evaluate_vietnamese import constructor
    from src.serve.rag.query_translator import BookPGVectorTranslator

    translator = BookPGVectorTranslator()
    plans = []

    def capture_plan(plan):
        plans.append(plan)
        return plan

    searcher = SelfQueryRetriever(
        query_constructor=constructor(few_shot=True) | RunnableLambda(capture_plan),
        vectorstore=retriever._load_vectorstore(),
        structured_query_translator=translator,
        search_kwargs={"k": 3},
    )
    rows = []
    for question, expected in [
        (
            "sách có đánh giá trên 4,2 và dưới 350 trang",
            lambda m: m["rating"] > 4.2 and m["pages_clean"] < 350,
        ),
        (
            "sách kinh dị dưới 300 trang",
            lambda m: "horror" in m["genres"].casefold() and m["pages_clean"] < 300,
        ),
        (
            "Tìm sách kinh dị không quá 300 trang",
            lambda m: "horror" in m["genres"].casefold() and m["pages_clean"] <= 300,
        ),
        (
            "sách không thuộc thể loại kinh dị, dưới 300 trang",
            lambda m: "horror" not in m["genres"].casefold() and m["pages_clean"] < 300,
        ),
    ]:
        try:
            documents = searcher.invoke(question)
            row = {
                "question": question,
                "translated_query": translator.visit_structured_query(plans[-1]),
                "count": len(documents),
                "constraints_pass": bool(documents)
                and all(expected(d.metadata) for d in documents),
                "results": [
                    {
                        "title": d.metadata.get("title"),
                        "metadata": {
                            key: d.metadata.get(key) for key in ["rating", "pages_clean", "genres"]
                        },
                    }
                    for d in documents
                ],
            }
        except Exception as exc:
            row = {
                "question": question,
                "error": type(exc).__name__,
                "message": str(exc)[:500]
                if isinstance(exc, (ValueError, NotImplementedError))
                else "See service status; no connection details are logged.",
            }
        rows.append(row)
        print(json.dumps({"self_query": row}, ensure_ascii=True), flush=True)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
