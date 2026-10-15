"""Same-candidate ablation: rank-only vs cross-encoder; labels never enter search."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def public_smoke():
    from src.serve.rag import retriever
    from src.serve.rag.pipeline import run_rag_pipeline

    question = (
        "tìm truyện về phi hành gia Mark Watney bị bỏ lại một mình trên sao Hỏa, dưới 400 trang"
    )
    with patch.dict(os.environ, {"RAG_RERANK_ENABLED": "1"}):
        exact = retriever.search("Dune by Frank Herbert", limit=3)
        vietnamese = retriever.search(question, limit=3)
        pipeline_result = run_rag_pipeline(question, limit=3)
    return {
        "sql_bypasses_rerank": bool(exact) and all("rerank_status" not in r for r in exact),
        "vietnamese_reranked": bool(vietnamese)
        and all(r.get("rerank_status") == "ok" for r in vietnamese),
        "page_filter_preserved": bool(vietnamese)
        and all(
            r["metadata"].get("pages_clean") is not None and r["metadata"]["pages_clean"] < 400
            for r in vietnamese
        ),
        "vietnamese_titles": [r["title"] for r in vietnamese],
        "pipeline_status": pipeline_result["verification_status"],
        "pipeline_safe": not pipeline_result.get("error")
        and pipeline_result["verification_status"] in {"ACCEPT", "ABSTAIN"},
        "pipeline_result": pipeline_result,
    }


def main():
    from scripts.evaluate_relevance import audit_sources
    from src.serve.rag import retriever
    from src.serve.rag.evaluation import evaluate_query, grouped_summary, summarize
    from src.serve.rag.reranking import DEFAULT_MODEL, _load_reranker, rerank_results

    raw = (ROOT / "tests/relevance_dataset.json").read_bytes()
    dataset = json.loads(raw)
    corpus_audit = audit_sources(dataset)
    start = time.perf_counter()
    scorer = _load_reranker(os.getenv("RERANK_MODEL", DEFAULT_MODEL))
    scorer.predict([("book", "A story about a book.")], show_progress_bar=False)
    warmup_ms = round((time.perf_counter() - start) * 1000, 2)
    baseline_rows, rerank_rows = [], []
    output = ROOT / "data/reports/reranking_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    details = []
    for case in dataset["cases"]:
        with patch.dict(os.environ, {"RAG_RERANK_ENABLED": "0"}):
            candidates = retriever.search(
                case["question"], limit=(30 if case["type"] in {"semantic", "mixed"} else 5)
            )
        before = evaluate_query(case, candidates[:5], 5)
        eligible = (
            bool(candidates)
            and case["type"] in {"semantic", "mixed"}
            and candidates[0].get("source") != "sql"
            and not candidates[0].get("catalog_only")
        )
        start = time.perf_counter()
        query = (
            candidates[0].get("query_trace", {}).get("semantic_query", case["question"])
            if candidates
            else case["question"]
        )
        ranked = rerank_results(query, candidates, 5, model=scorer) if eligible else candidates[:5]
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        after = evaluate_query(case, ranked, 5)
        baseline_rows.append(before)
        rerank_rows.append(after)
        details.append(
            {
                "id": case["id"],
                "eligible": eligible,
                "candidate_ids": [r["book_id"] for r in candidates],
                "semantic_query": query,
                "rerank_ms": elapsed,
                "before": before,
                "after": after,
                "top_titles": [r["title"] for r in ranked],
            }
        )
        print(
            f"{case['id']}: RR {before['reciprocal_rank']} -> "
            f"{after['reciprocal_rank']}; rerank {elapsed}ms",
            flush=True,
        )
    report = {
        "scope": "Development ablation on the same frozen candidate pool per query; not held-out",
        "limitations": dataset["limitations"],
        "model": DEFAULT_MODEL,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "corpus_audit": corpus_audit,
        "warmup_ms": warmup_ms,
        "baseline": summarize(baseline_rows),
        "reranked": summarize(rerank_rows),
        "by_type_before": grouped_summary(baseline_rows, "type"),
        "by_type_after": grouped_summary(rerank_rows, "type"),
        "rows": details,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Exercise the public integration too; ablation above measures rank changes only.
    report["public_smoke"] = public_smoke()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": report["baseline"], "reranked": report["reranked"]}, indent=2))


if __name__ == "__main__":
    main()
