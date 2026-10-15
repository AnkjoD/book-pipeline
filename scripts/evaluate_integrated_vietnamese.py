"""Exercise the chatbot's actual search facade, not the experimental constructor."""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_vietnamese import CASES, compare, probes  # noqa: E402
from src.serve.rag import retriever  # noqa: E402
from src.serve.rag.pipeline import run_rag_pipeline  # noqa: E402


def matches_backend(condition, metadata):
    if "$and" in condition:
        return all(matches_backend(c, metadata) for c in condition["$and"])
    if "$or" in condition:
        return any(matches_backend(c, metadata) for c in condition["$or"])
    if "$not" in condition:
        return not matches_backend(condition["$not"], metadata)
    for field, comparisons in condition.items():
        for operator, value in comparisons.items():
            if operator == "$ilike":
                # Our translator escapes wildcard characters in the user's literal.
                pattern = ""
                escaped = False
                for char in value:
                    if escaped:
                        pattern += re.escape(char)
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    else:
                        pattern += ".*" if char == "%" else "." if char == "_" else re.escape(char)
                if not re.fullmatch(pattern, str(metadata[field]), re.IGNORECASE):
                    return False
            elif not compare(metadata[field], operator.removeprefix("$"), value):
                return False
    return True


def main():
    report = {"scope": "live search and explicit filter checks; not semantic relevance", "rows": []}
    path = ROOT / "data/reports/integrated_vietnamese.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = probes()
    for case_id, question, expected in CASES:
        start = time.perf_counter()
        try:
            results = retriever.search(question, limit=3)
            row = {
                "id": case_id,
                "question": question,
                "count": len(results),
                "route": results[0]["source"] if results else None,
                "results": results,
                "expected": expected,
            }
            row["returned_constraints_pass"] = bool(results) and all(
                all(
                    compare(
                        ({**r["metadata"], "title": r["title"], "author_names": r["author_names"]})[
                            field
                        ],
                        op,
                        value,
                    )
                    for field, op, value in expected
                )
                for r in results
            )
            if results and results[0].get("query_trace"):
                actual_filter = results[0]["query_trace"].get("filter", {})
                row["plan_constraints_pass"] = all(
                    matches_backend(actual_filter, m)
                    == all(compare(m[field], op, value) for field, op, value in expected)
                    for m in samples
                )
            else:
                row["plan_constraints_pass"] = row["returned_constraints_pass"]
        except Exception as exc:
            row = {
                "id": case_id,
                "question": question,
                "error": type(exc).__name__,
                "returned_constraints_pass": False,
                "plan_constraints_pass": False,
            }
        row["seconds"] = round(time.perf_counter() - start, 2)
        report["rows"].append(row)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps({k: v for k, v in row.items() if k != "results"}, ensure_ascii=True),
            flush=True,
        )
    report["pipeline_examples"] = []
    for question in [
        "Cuốn The Hobbit kể về điều gì?",
        "tìm sách mỏng mỏng thôi",
        "sach xuat ban nam 9999",
    ]:
        result = run_rag_pipeline(question, limit=3)
        report["pipeline_examples"].append({"question": question, **result})
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
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Plans:", sum(r["plan_constraints_pass"] for r in report["rows"]), "/", len(CASES))
    print("Results:", sum(r["returned_constraints_pass"] for r in report["rows"]), "/", len(CASES))
    return (
        0
        if all(
            r["plan_constraints_pass"] and r["returned_constraints_pass"] for r in report["rows"]
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
