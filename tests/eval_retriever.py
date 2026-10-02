"""
Baseline retrieval evaluation against tests/eval_dataset.json.

Metrics per query type:
  - Recall@k   : >=1 result satisfying hard constraints in top-k
  - CSR        : Constraint Satisfaction Rate (fraction of results passing constraints)
  - Latency (ms)
  - Empty Rate : fraction of queries returning 0 results

Run:
    python tests/eval_retriever.py
    python tests/eval_retriever.py --limit 3
    python tests/eval_retriever.py --types sql exact impossible
    python tests/eval_retriever.py --no-warmup
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.serve.rag.retriever import search, _load_vectorstore

DATASET = Path(__file__).parent / "eval_dataset.json"
QUERY_TIMEOUT_S = 30


# ── constraint checkers ───────────────────────────────────────────────────────

def _passes(doc: dict, constraints: dict) -> bool:
    m = doc.get("metadata", {})
    if "min_rating" in constraints:
        r = m.get("rating")
        if r is None or r < constraints["min_rating"]:
            return False
    if "max_pages" in constraints:
        p = m.get("pages_clean")
        if p is None or p > constraints["max_pages"]:
            return False
    if "min_pages" in constraints:
        p = m.get("pages_clean")
        if p is None or p < constraints["min_pages"]:
            return False
    if "min_year" in constraints:
        y = m.get("release_year_clean")
        if y is None or y < constraints["min_year"]:
            return False
    if "max_year" in constraints:
        y = m.get("release_year_clean")
        if y is None or y > constraints["max_year"]:
            return False
    if "genre" in constraints:
        g = (m.get("genres") or "").lower()
        if constraints["genre"].lower() not in g:
            return False
    return True


def _title_hit(results: list, expected: str) -> bool:
    for r in results:
        title = (r.get("title") or r.get("metadata", {}).get("title") or "").lower()
        if expected.lower() in title:
            return True
    return False


def _author_hit(results: list, expected_author: str) -> bool:
    for r in results:
        # author_names at top level (sql) OR inside metadata (vector/mixed)
        authors = (
            r.get("author_names") or
            r.get("metadata", {}).get("author_names") or
            ""
        ).lower()
        if expected_author.lower() in authors:
            return True
    return False


# ── timed search wrapper ──────────────────────────────────────────────────────

def _timed_search(query: str, limit: int):
    import threading
    result_box = []
    error_box = []

    def _run():
        try:
            result_box.append(search(query, limit=limit))
        except Exception as e:
            error_box.append(str(e))

    t = threading.Thread(target=_run, daemon=True)
    t0 = time.perf_counter()
    t.start()
    t.join(timeout=QUERY_TIMEOUT_S)
    latency_ms = (time.perf_counter() - t0) * 1000

    if t.is_alive():
        return [], latency_ms, "TIMEOUT >{}s".format(QUERY_TIMEOUT_S)
    if error_box:
        return [], latency_ms, error_box[0]
    return result_box[0] if result_box else [], latency_ms, None


# ── runner ────────────────────────────────────────────────────────────────────

def evaluate(limit=5, types=None, warmup=True):
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    if types:
        dataset = [q for q in dataset if q["type"] in types]

    if warmup and any(q["type"] in ("semantic", "mixed", "exact") for q in dataset):
        print("Warming up embedding model ...", flush=True)
        t0 = time.perf_counter()
        try:
            _load_vectorstore()
            print("  warm-up done in {:.1f}s\n".format(time.perf_counter() - t0))
        except Exception as e:
            print("  warm-up failed: {} (vector queries may be slow)\n".format(e))

    per_type = defaultdict(list)
    rows = []

    for q in dataset:
        qid    = q["id"]
        qtype  = q["type"]
        query  = q["query"]
        constr = q.get("constraints", {})

        results, latency_ms, err = _timed_search(query, limit)
        n     = len(results)
        empty = (n == 0)

        if err:
            recall, csr, flag = 0.0, 0.0, "ER"
        elif qtype == "impossible":
            recall = 1.0 if empty else 0.0
            csr    = 1.0
            flag   = "OK" if recall == 1.0 else "XX"
        elif qtype == "exact":
            if "expected_match" in q:
                recall = 1.0 if (not empty and _title_hit(results, q["expected_match"])) else 0.0
            elif "expected_author" in q:
                recall = 1.0 if (not empty and _author_hit(results, q["expected_author"])) else 0.0
            else:
                recall = 1.0 if not empty else 0.0
            csr  = 1.0
            flag = "OK" if recall == 1.0 else "XX"
        elif qtype == "semantic":
            recall = 1.0 if not empty else 0.0
            csr    = 1.0
            flag   = "OK" if recall == 1.0 else "XX"
        else:  # numeric / mixed
            if empty:
                recall, csr = 0.0, 0.0
            else:
                passing = [r for r in results if _passes(r, constr)]
                csr     = len(passing) / n
                recall  = 1.0 if passing else 0.0
            flag = "OK" if recall == 1.0 else "XX"

        err_str = " [{}]".format(err) if err else ""
        print("  {} [{:4s}] {:10s} | recall={:.1f} csr={:.2f} n={:2d} lat={:6.0f}ms | {}{}".format(
            flag, qid, qtype, recall, csr, n, latency_ms, query[:50], err_str
        ))

        entry = {
            "qid": qid, "type": qtype, "query": query,
            "recall": recall, "csr": csr,
            "latency_ms": round(latency_ms, 1),
            "n_results": n, "empty": empty, "error": err,
        }
        per_type[qtype].append(entry)
        rows.append(entry)

    # ── aggregate ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("{:<12} {:>4} {:>10} {:>8} {:>13} {:>10}".format(
        "TYPE", "N", "Recall@k", "CSR", "Latency(ms)", "EmptyRate"
    ))
    print("-" * 72)

    total_recall = total_csr = total_lat = 0.0
    total_n = 0

    for qtype, items in sorted(per_type.items()):
        n_q = len(items)
        r   = sum(i["recall"] for i in items) / n_q
        csr = sum(i["csr"]    for i in items) / n_q
        lat = sum(i["latency_ms"] for i in items) / n_q
        emp = sum(1 for i in items if i["empty"]) / n_q
        print("  {:<10} {:>4}   {:>8.2%}   {:>6.2%}   {:>10.0f}   {:>8.2%}".format(
            qtype, n_q, r, csr, lat, emp
        ))
        total_recall += r * n_q
        total_csr    += csr * n_q
        total_lat    += lat * n_q
        total_n      += n_q

    print("-" * 72)
    print("  {:<10} {:>4}   {:>8.2%}   {:>6.2%}   {:>10.0f}".format(
        "OVERALL", total_n, total_recall / total_n, total_csr / total_n, total_lat / total_n
    ))
    print("=" * 72)

    report_path = Path(__file__).parent / "eval_report.json"
    report_path.write_text(
        json.dumps({"limit": limit, "results": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nReport -> {}".format(report_path))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",     type=int,  default=5)
    ap.add_argument("--types",     nargs="+", help="e.g. --types sql exact impossible numeric")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()
    evaluate(limit=args.limit, types=args.types, warmup=not args.no_warmup)
