"""Runnable verification script cho retriever baseline."""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.serve.rag.retriever import _parse_filters, _route, search


def run_all_checks():
    print("=== BAT DAU KIEM TRA RETRIEVER BASELINE ===")

    # 1. Empty query check
    print("\n[1/6] Kiem tra Empty Query...")
    try:
        search("   ")
        assert False, "Phai raise ValueError khi query rong"
    except ValueError as e:
        print(f"  PASS: Raise ValueError voi message: '{e}'")

    # 2. Route classification check
    print("\n[2/6] Kiem tra Routing Classification...")
    r1 = _route("Dune by Frank Herbert")
    assert r1 == "sql", f"Expected sql, got {r1}"
    r2 = _route("books rated at least 4.2 under 350 pages")
    assert r2 == "sql", f"Expected sql, got {r2}"
    r3 = _route("a dark story about memory and grief")
    assert r3 == "vector", f"Expected vector, got {r3}"
    r4 = _route("dark fantasy under 400 pages rated above 4")
    assert r4 == "mixed", f"Expected mixed, got {r4}"
    r5 = _route("books above rating 6 and under 0 pages")
    assert r5 == "sql", f"Expected sql, got {r5}"
    print("  PASS: Tat ca 5 route deu phan luong chinh xac.")

    # 3. Exact SQL Search check
    print("\n[3/6] Kiem tra Exact SQL Search ('Dune by Frank Herbert')...")
    res_exact = search("Dune by Frank Herbert", limit=3)
    assert len(res_exact) >= 1, "Khong tim thay Dune"
    top_exact = res_exact[0]
    assert "dune" in top_exact["title"].lower(), f"Title khong khop: {top_exact['title']}"
    assert "frank herbert" in top_exact["author_names"].lower(), f"Author khong khop: {top_exact['author_names']}"
    assert top_exact["source"] == "sql"
    assert top_exact["score"] == 1.0
    print(f"  PASS: Tim thay '{top_exact['title']}' boi '{top_exact['author_names']}' (source={top_exact['source']})")

    # 4. Numeric SQL Search check
    print("\n[4/6] Kiem tra Numeric SQL Search ('books rated at least 4.2 under 350 pages')...")
    res_num = search("books rated at least 4.2 under 350 pages", limit=5)
    assert 1 <= len(res_num) <= 5, f"So luong ket qua khong hop le: {len(res_num)}"
    seen_ids = set()
    for item in res_num:
        assert item["book_id"] not in seen_ids, "Trung lap book_id"
        seen_ids.add(item["book_id"])
        rating = item["metadata"]["rating"]
        pages = item["metadata"]["pages_clean"]
        assert rating is not None and rating >= 4.2, f"Rating vi pham: {rating}"
        assert pages is not None and pages <= 350, f"Pages vi pham: {pages}"
        assert item["source"] == "sql"
        print(f"  - [{item['book_id']}] {item['title'][:30]} | Rating: {rating:.2f} | Pages: {pages}")
    print("  PASS: 100% ket qua thoa man rating >= 4.2 va pages <= 350.")

    # 5. Semantic Vector Search check
    print("\n[5/6] Kiem tra Semantic Vector Search ('a dark story about memory and grief')...")
    res_sem = search("a dark story about memory and grief", limit=5)
    assert 1 <= len(res_sem) <= 5
    seen_ids = set()
    for item in res_sem:
        assert item["book_id"] not in seen_ids, "Trung lap book_id"
        seen_ids.add(item["book_id"])
        assert item["source"] == "vector"
        assert item["distance"] >= 0.0
        assert len(item["evidence"]) > 0
        print(f"  - [{item['book_id']}] {item['title'][:30]} | Dist: {item['distance']} | Score: {item['score']}")
    print("  PASS: Vector search tra ve ket qua ngu nghia co evidence va provenance.")

    # 6. Mixed Search & Impossible Constraint checks
    print("\n[6/6] Kiem tra Mixed Search & Impossible Constraints...")
    res_mixed = search("dark fantasy under 400 pages rated above 4", limit=5)
    assert len(res_mixed) > 0, "Khong tim thay ket qua mixed"
    for item in res_mixed:
        rating = item["metadata"]["rating"]
        pages = item["metadata"]["pages_clean"]
        genres = item["metadata"]["genres"].lower()
        assert rating is not None and rating >= 4.0, f"Rating vi pham: {rating}"
        assert pages is not None and pages <= 400, f"Pages vi pham: {pages}"
        assert "fantasy" in genres, f"Genre vi pham: {genres}"
        assert item["source"] == "mixed"
        print(f"  - [MIXED: {item['book_id']}] {item['title'][:30]} | Rating: {rating:.2f} | Pages: {pages} | Dist: {item['distance']}")
    print("  PASS: Mixed search thoa man ca hard constraints va semantic relevancy.")

    # Impossible check
    res_imp = search("books above rating 6 and under 0 pages", limit=5)
    assert res_imp == [], f"Expected empty list for impossible constraint, got {res_imp}"
    print("  PASS: Impossible query tra ve danh sach rong [] ngay lap tuc.")

    print("\n=== TAT CA 6 KIEM TRA RETRIEVAL DA PASS 100% ===")


if __name__ == "__main__":
    run_all_checks()
