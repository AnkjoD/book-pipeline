"""Run real SQL/vector/Ollama requests and save an inspectable local report."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.serve.rag.pipeline import run_rag_pipeline  # noqa: E402


def main():
    cases = [
        ("exact", "Dune by Frank Herbert", {"ACCEPT"}),
        ("semantic", "a dark story about memory and grief", {"ACCEPT", "ABSTAIN"}),
        ("mixed", "dark fantasy under 400 pages rated above 4", {"ACCEPT", "ABSTAIN"}),
        ("impossible", "sách xuất bản năm 9999", {"ABSTAIN"}),
    ]
    rows = []
    for kind, question, expected in cases:
        start = time.perf_counter()
        try:
            result = run_rag_pipeline(question, limit=3)
            passed = result["verification_status"] in expected and not result.get("error")
            if kind in {"semantic", "mixed"}:
                passed = passed and result["retrieved_count"] > 0
            row = {"kind": kind, "question": question, "passed": passed, **result}
        except Exception as exc:
            row = {
                "kind": kind,
                "question": question,
                "passed": False,
                "error": type(exc).__name__ + ": " + str(exc),
            }
        row["seconds"] = round(time.perf_counter() - start, 2)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
    destination = ROOT / "data" / "reports" / "rag_smoke.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {destination}")
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
