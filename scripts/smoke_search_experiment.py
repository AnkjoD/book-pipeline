"""Exercise actual Streamlit mode selection and catalog answers, no patched services."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from streamlit.testing.v1 import AppTest

    from scripts.compare_search_strategies import relation_cases
    from src.serve.rag.experimental import load_corpus

    _, books = load_corpus()
    relation = next(c for c in relation_cases(books) if c["id"] == "relation_series_Catching_Fire")
    rows = []
    cases = [
        (label, 'tìm sách "Dune" của Frank Herbert', None)
        for label in [
            "Hiện tại",
            "Từ khóa + ngữ nghĩa (thử nghiệm)",
            "Theo quan hệ GraphRAG (thử nghiệm)",
        ]
    ]
    cases.append(
        (
            "Theo quan hệ GraphRAG (thử nghiệm)",
            'tìm sách cùng bộ với "Catching Fire"',
            set(relation["relevant_book_ids"]),
        )
    )
    for label, question, expected_ids in cases:
        app = AppTest.from_file(str(ROOT / "src/serve/chatbot.py"), default_timeout=180).run()
        selector = next(box for box in app.selectbox if box.label == "Cách tìm sách")
        selector.select(label).run()
        app.chat_input[0].set_value(question).run()
        if app.exception:
            raise RuntimeError("Streamlit app failed")
        result = app.session_state["messages"][-1]["meta"]
        row = {
            "mode": label,
            "question": question,
            "status": result["verification_status"],
            "titles": [r["title"] for r in result["raw_provenance"]],
            "passed": result["verification_status"] == "ACCEPT"
            and bool(result["raw_provenance"])
            and (
                all(str(r["book_id"]) in expected_ids for r in result["raw_provenance"])
                if expected_ids is not None
                else any(r["title"] == "Dune" for r in result["raw_provenance"])
            ),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    path = ROOT / "data/reports/search_experiment_ui.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if not all(row["passed"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
