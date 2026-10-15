"""Create development labels from Gold, without calling the retriever being tested."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
BOOKS = [
    ("Dune", "Frank Herbert"),
    ("Animal Farm", "George Orwell"),
    ("1984", "George Orwell"),
    ("Fahrenheit 451", "Ray Bradbury"),
    ("Pride and Prejudice", "Jane Austen"),
    ("The Great Gatsby", "F. Scott Fitzgerald"),
    ("The Hunger Games", "Suzanne Collins"),
    ("The Martian", "Andy Weir"),
]
# Explicit evidence checked against Gold; never use search() output as labels.
TOPICS = [
    (
        "Dune",
        "Frank Herbert",
        "desert planet Arrakis",
        "story about Paul Atreides on the desert planet Arrakis",
        "tìm truyện về Paul Atreides trên hành tinh sa mạc Arrakis",
    ),
    (
        "Animal Farm",
        "George Orwell",
        "pigs Napoleon and Snowball",
        "story about farm animals rebelling led by pigs Napoleon and Snowball",
        "tìm truyện về động vật nổi dậy trong trang trại do Napoleon và Snowball dẫn dắt",
    ),
    (
        "1984",
        "George Orwell",
        "Ministry of Truth",
        "story about Winston Smith rewriting history at the Ministry of Truth",
        "tìm truyện về Winston Smith sửa lại lịch sử tại Bộ Sự thật",
    ),
    (
        "Fahrenheit 451",
        "Ray Bradbury",
        "His job is to burn books",
        "story about fireman Guy Montag burning forbidden books",
        "tìm truyện về lính cứu hỏa Guy Montag có nhiệm vụ đốt sách bị cấm",
    ),
    (
        "The Martian",
        "Andy Weir",
        "stranded on Mars",
        "story about astronaut Mark Watney surviving alone stranded on Mars",
        "tìm truyện về phi hành gia Mark Watney bị bỏ lại một mình trên sao Hỏa",
    ),
]


def build_dataset(db_path: str) -> dict:
    cases = []
    snapshots = {}
    with duckdb.connect(db_path, read_only=True) as connection:

        def find(title, author):
            rows = connection.execute(
                "SELECT book_id, title, author_names, document_hash, document_text, pages_clean "
                "FROM rag_documents WHERE lower(title)=lower(?) "
                "AND contains(lower(author_names),lower(?)) "
                "ORDER BY ratings_count DESC NULLS LAST, book_id",
                [title, author],
            ).fetchall()
            if not rows:
                raise ValueError(f"No Gold labels for {title} / {author}")
            for row in rows:
                snapshots[str(row[0])] = {
                    "title": row[1],
                    "author_names": row[2],
                    "document_hash": row[3],
                }
            return rows

        for index, (title, author) in enumerate(BOOKS, start=1):
            matches = find(title, author)
            for variant, (question, language) in enumerate(
                [
                    (f"{title} by {author}", "en"),
                    (f'"{title}" by {author}', "en"),
                    (f'tìm sách "{title}" của {author}', "vi"),
                ],
                start=1,
            ):
                cases.append(
                    {
                        "id": f"exact_{index:02d}_{variant}",
                        "question": question,
                        "type": "exact",
                        "language": language,
                        "split": "dev",
                        "relevant_book_ids": [str(row[0]) for row in matches],
                        "label_scope": "all_gold_exact_title_author_matches",
                        "label_method": "Independent SQL; all matching book IDs",
                        "constraints": [
                            {"field": "title", "op": "text_eq", "value": title},
                            {"field": "author_names", "op": "contains", "value": author},
                        ],
                    }
                )
        for index, (title, author, evidence, english, vietnamese) in enumerate(TOPICS, start=1):
            matches = [r for r in find(title, author) if evidence.casefold() in r[4].casefold()]
            if not matches:
                raise ValueError(f"Label evidence missing for {title}")
            for language, question in [("en", english), ("vi", vietnamese)]:
                cases.append(
                    {
                        "id": f"semantic_{index:02d}_{language}",
                        "question": question,
                        "type": "semantic",
                        "language": language,
                        "split": "dev",
                        "relevant_book_ids": [str(r[0]) for r in matches],
                        "label_scope": "partial_evidence_supported_positives",
                        "label_method": "Agent-authored from Gold descriptions; not human reviewed",
                        "label_evidence": evidence,
                    }
                )
            if title == "The Martian":
                filtered = [r for r in matches if r[5] is not None and r[5] < 400]
                if not filtered:
                    raise ValueError("Mixed case has no eligible labels")
                for language, question in [
                    ("en", english + " under 400 pages"),
                    ("vi", vietnamese + ", dưới 400 trang"),
                ]:
                    cases.append(
                        {
                            "id": f"mixed_mars_{language}",
                            "question": question,
                            "type": "mixed",
                            "language": language,
                            "split": "dev",
                            "relevant_book_ids": [str(r[0]) for r in filtered],
                            "label_scope": "partial_evidence_supported_positives",
                            "label_evidence": evidence,
                            "constraints": [{"field": "pages_clean", "op": "lt", "value": 400}],
                        }
                    )
        for index, (question, language) in enumerate(
            [
                ("books with rating above 6.0", "en"),
                ("books under 0 pages", "en"),
                ("sách xuất bản năm 9999", "vi"),
                ("", "und"),
            ],
            start=1,
        ):
            cases.append(
                {
                    "id": f"empty_{index:02d}",
                    "question": question,
                    "type": "unanswerable",
                    "language": language,
                    "split": "dev",
                    "relevant_book_ids": [],
                    "expected_empty": True,
                    "label_scope": "impossible_or_empty_input",
                }
            )
        gold_count = connection.execute("SELECT count(*) FROM rag_documents").fetchone()[0]
    return {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Development diagnostics; no held-out or human-reviewed semantic test set",
        "gold_rows": gold_count,
        "limitations": [
            "Semantic labels are partial; recall means labeled-positive recall, not corpus recall",
            "Exact labels may include multiple editions; also inspect hit rate and MRR",
            "SQL uses all Gold; vector uses only the current indexed subset",
            "Do not compare algorithms on different corpora or claim held-out gains",
        ],
        "book_snapshots": snapshots,
        "cases": cases,
    }


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tests/relevance_dataset.json")
    args = parser.parse_args()
    dataset = build_dataset(os.getenv("DUCKDB_PATH", str(ROOT / "data/warehouse/gold.duckdb")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(dataset['cases'])} development cases: {args.output}")


if __name__ == "__main__":
    main()
