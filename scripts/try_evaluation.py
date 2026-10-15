from pprint import pprint

from src.serve.rag.evaluation import evaluate_query
from src.serve.rag.retriever import search

test_case = {
    "id": "exact_dune",
    "question": "Dune by Frank Herbert",
    "relevant_book_ids": ["312460"],
}

results = search(test_case["question"], limit=5)

print("Sách tìm được:")
pprint([{"book_id": item["book_id"], "title": item["title"]} for item in results])

report = evaluate_query(test_case, results, k=5)

print("Điểm đánh giá:")
pprint(report)
