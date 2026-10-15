"""Compare current Vietnamese parsing with a separate LangChain query-constructor trial.

No changes to the live retriever or vector index. Reports go under data/reports/.
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.serve.rag import retriever  # noqa: E402

CASES = [
    (
        "title",
        "Dune của Frank Herbert",
        [("title", "eq", "Dune"), ("author_names", "contain", "Frank Herbert")],
    ),
    (
        "natural_title",
        "Tìm giúp mình cuốn Dune của Frank Herbert",
        [("title", "eq", "Dune"), ("author_names", "contain", "Frank Herbert")],
    ),
    (
        "no_accents_title",
        "tim sach Dune cua Frank Herbert",
        [("title", "eq", "Dune"), ("author_names", "contain", "Frank Herbert")],
    ),
    (
        "genre_pages",
        "sách kinh dị dưới 300 trang",
        [("genres", "contain", "horror"), ("pages_clean", "lt", 300)],
    ),
    (
        "inclusive",
        "Tìm sách kinh dị không quá 300 trang",
        [("genres", "contain", "horror"), ("pages_clean", "lte", 300)],
    ),
    (
        "no_accents",
        "tim truyen kinh di duoi 300 trang",
        [("genres", "contain", "horror"), ("pages_clean", "lt", 300)],
    ),
    (
        "paraphrase",
        "Cho mình truyện trinh thám, số trang ít hơn 300 nhé",
        [("genres", "contain", "mystery"), ("pages_clean", "lt", 300)],
    ),
    (
        "decimal_comma",
        "sách có đánh giá trên 4,2 và dưới 350 trang",
        [("rating", "gt", 4.2), ("pages_clean", "lt", 350)],
    ),
    (
        "from_year",
        "sách trinh thám từ năm 2020",
        [("genres", "contain", "mystery"), ("release_year_clean", "gte", 2020)],
    ),
    (
        "negation",
        "sách không thuộc thể loại kinh dị, dưới 300 trang",
        [("genres", "not_contain", "horror"), ("pages_clean", "lt", 300)],
    ),
    (
        "mixed_vi",
        "truyện giả tưởng u tối về ký ức, dưới 400 trang, đánh giá trên 4",
        [("genres", "contain", "fantasy"), ("pages_clean", "lt", 400), ("rating", "gt", 4)],
    ),
    (
        "typo",
        "tim sach kinh di duoi 300 trnag",
        [("genres", "contain", "horror"), ("pages_clean", "lt", 300)],
    ),
]


def compare(actual, operator, value):
    if operator == "contain":
        return str(value).casefold() in str(actual).casefold()
    if operator == "not_contain":
        return str(value).casefold() not in str(actual).casefold()
    if operator == "eq":
        return str(actual).casefold() == str(value).casefold()
    if operator == "ne":
        return not compare(actual, "eq", value)
    if operator == "lt":
        return actual < value
    if operator == "lte":
        return actual <= value
    if operator == "gt":
        return actual > value
    if operator == "gte":
        return actual >= value
    raise ValueError(f"Unsupported operator: {operator}")


def ast_matches(node, metadata):
    if node is None:
        return True
    if hasattr(node, "comparator"):
        return compare(metadata[node.attribute], node.comparator.value, node.value)
    children = [ast_matches(child, metadata) for child in node.arguments]
    if node.operator.value == "and":
        return all(children)
    if node.operator.value == "or":
        return any(children)
    if node.operator.value == "not" and len(children) == 1:
        return not children[0]
    raise ValueError("Unsupported logical operator")


def current_matches(filters, metadata):
    if filters["is_impossible"]:
        return False
    for key, field in [("title", "title"), ("author", "author_names"), ("genre", "genres")]:
        if filters[key] is not None:
            op = "eq" if key == "title" else "contain"
            if not compare(metadata[field], op, filters[key]):
                return False
    for name, field in [
        ("rating", "rating"),
        ("pages", "pages_clean"),
        ("year", "release_year_clean"),
    ]:
        for prefix, inclusive, strict in [("min", "gte", "gt"), ("max", "lte", "lt")]:
            key = f"{prefix}_{name}"
            value = filters[key]
            if value is not None:
                op = strict if filters.get(key + "_strict") else inclusive
                if not compare(metadata[field], op, value):
                    return False
    return filters["exact_year"] is None or metadata["release_year_clean"] == filters["exact_year"]


def probes():
    fields = {
        "title": ["Dune", "Other"],
        "author_names": ["Frank Herbert", "Brian Herbert, Frank Herbert", "Other"],
        "genres": ["Horror, Fiction", "Mystery, Fiction", "Fantasy, Fiction"],
        "pages_clean": [299, 300, 301, 349, 350, 399, 400, 401],
        "rating": [4.0, 4.1, 4.2, 4.3],
        "release_year_clean": [2019, 2020, 2021],
    }
    return [dict(zip(fields, values)) for values in itertools.product(*fields.values())]


def constructor(few_shot=False):
    from langchain_classic.chains.query_constructor.base import load_query_constructor_runnable
    from langchain_classic.chains.query_constructor.schema import AttributeInfo
    from langchain_ollama import ChatOllama

    descriptions = [
        ("title", "Tên sách, bỏ các từ yêu cầu như tìm giúp mình cuốn/sách", "string"),
        ("author_names", "Danh sách credit/tác giả. Dùng contain cho tên tác giả", "string"),
        (
            "genres",
            "Chuỗi nhiều thể loại tiếng Anh, dùng contain. kinh dị=horror, "
            "trinh thám=mystery, giả tưởng=fantasy. Không thuộc = not(contain(...)).",
            "string",
        ),
        ("pages_clean", "Số trang. dưới/ít hơn/duoi=lt; không quá/tối đa=lte", "integer"),
        ("rating", "Điểm 0-5. trên=gt, ít nhất=gte. 4,2 nghĩa là 4.2", "float"),
        ("release_year_clean", "Năm xuất bản. từ năm=gte, sau năm=gt, trước năm=lt", "integer"),
    ]
    llm = ChatOllama(
        model="qwen2.5:7b-instruct-q4_K_M",
        temperature=0,
        format="json",
        num_predict=512,
        client_kwargs={"timeout": 60},
    )
    examples = (
        [
            (
                "Tìm The Hobbit của J.R.R. Tolkien",
                {
                    "query": "",
                    "filter": 'and(eq("title", "The Hobbit"), '
                    'contain("author_names", "J.R.R. Tolkien"))',
                },
            ),
            (
                "tim truyen lang man khong qua 250 trang, danh gia it nhat 3,5",
                {
                    "query": "",
                    "filter": 'and(contain("genres", "romance"), '
                    'lte("pages_clean", 250), gte("rating", 3.5))',
                },
            ),
            (
                "truyện giả tưởng về rồng, dưới 180 trang, điểm trên 3",
                {
                    "query": "dragons",
                    "filter": 'and(contain("genres", "fantasy"), '
                    'lt("pages_clean", 180), gt("rating", 3))',
                },
            ),
            (
                "sách không phải khoa học viễn tưởng, xuất bản sau 2015",
                {
                    "query": "",
                    "filter": 'and(not(contain("genres", "science fiction")), '
                    'gt("release_year_clean", 2015))',
                },
            ),
        ]
        if few_shot
        else None
    )
    return load_query_constructor_runnable(
        llm,
        "Mô tả sách tiếng Anh. User dùng tiếng Việt có dấu, không dấu hoặc gõ sai. "
        "Chỉ lấy ràng buộc rõ ràng; không tự thêm. Giữ theme/mood trong query, "
        "có thể dịch query sang tiếng Anh. Lọc tên/tác giả/thể loại bằng đúng metadata.",
        [AttributeInfo(name=n, description=d, type=t) for n, d, t in descriptions],
        examples=examples,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--few-shot", action="store_true")
    args = parser.parse_args()
    samples = probes()
    chain = constructor(args.few_shot) if args.llm else None
    rows = []
    for case_id, question, expected in CASES:
        start = time.perf_counter()
        wanted = [
            all(compare(m[field], op, value) for field, op, value in expected) for m in samples
        ]
        filters = retriever._parse_filters(question)
        current_pass = wanted == [current_matches(filters, m) for m in samples]
        row = {
            "id": case_id,
            "question": question,
            "expected": expected,
            "current_filters": filters,
            "current_pass": current_pass,
        }
        if chain:
            try:
                plan = chain.invoke({"query": question})
                row["langchain_plan"] = plan.model_dump(mode="json", serialize_as_any=True)
                row["langchain_pass"] = wanted == [ast_matches(plan.filter, m) for m in samples]
            except Exception as exc:
                row["langchain_error"] = type(exc).__name__ + ": " + str(exc)
                row["langchain_pass"] = False
        row["seconds"] = round(time.perf_counter() - start, 2)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
        filename = (
            "vietnamese_query_understanding_few_shot.json"
            if args.few_shot
            else "vietnamese_query_understanding.json"
        )
        destination = ROOT / "data/reports" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Current:", sum(r["current_pass"] for r in rows), "/", len(rows))
    if chain:
        print("LangChain trial:", sum(r["langchain_pass"] for r in rows), "/", len(rows))
    print("Checks filter behavior on synthetic boundary probes, not semantic relevance.")


if __name__ == "__main__":
    main()
