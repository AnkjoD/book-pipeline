# Data Science → AI Engineer Roadmap

Project này có thể trở thành portfolio lai giữa Data Science, Data Engineering
và AI Engineering. Không nên triển khai cả ba nhánh cùng lúc; mỗi giai đoạn
phải có deliverable và metric rõ ràng.

## 1. Data correctness và vertical slice

Phạm vi đầu tiên: chạy vertical slice 20.000–50.000 sách cùng các dimension liên quan.

Deliverable:

- Bronze → Silver → DuckDB chạy được từ đầu đến cuối.
- Kiểm tra unique key, null rate, duplicate, orphan foreign key và row-count reconciliation.
- Ghi lại lý do mỗi record bị loại từ Bronze sang Silver.

Gate để đi tiếp: dbt tests pass và có báo cáo data quality.

Thứ tự thực hành ngay trên V3:

1. Tạo một sample Bronze nhỏ từ dữ liệu đã fetch cho `books`, `authors`,
   `tags`, `reading_formats` (không full-fetch lại chỉ để học transform).
2. Chạy Bronze → Silver và kiểm tra schema/row count của từng output.
3. Chạy `dbt run` rồi `dbt test`; tự sửa ít nhất một test cố tình làm fail.
4. Viết một báo cáo reconciliation: input rows, valid rows, rejected rows và lý do.
5. Chỉ sau gate này mới làm EDA hoặc embedding.

## 2. Data Science foundation

Tạo `notebooks/01_eda_books.ipynb` và trả lời tối thiểu năm câu hỏi:

1. Rating và popularity phân bố ra sao?
2. Rating thay đổi thế nào theo số lượt đánh giá?
3. Genre nào phát triển theo từng giai đoạn?
4. Page count khác nhau ra sao giữa literary type?
5. Missingness và selection bias ảnh hưởng kết luận thế nào?

Dùng confidence interval/effect size khi phù hợp, không chỉ báo p-value.

## 3. Machine Learning baseline

Bài toán đề xuất: multi-label genre classification từ `title`, `subtitle`,
`description`.

Baseline:

1. TF-IDF.
2. One-vs-Rest Logistic Regression.
3. Micro F1, Macro F1, Precision@k, Recall@k.
4. Error analysis theo genre và theo độ dài description.
5. Group split theo author hoặc temporal split để kiểm tra generalization.

Chỉ thử transformer sau khi baseline, split và metrics đã ổn định.

## 4. AI Engineering / RAG

RAG đúng là một hướng AI Engineer, nhưng một RAG demo chưa đủ. Hệ thống cần có:

1. Document design: chuẩn hóa metadata và `chunk_text`.
2. Embedding pipeline có version model và idempotent upsert.
3. Hybrid retrieval: vector search + lexical/BM25.
4. Metadata filters: genre, author, release year, language.
5. Reranker và context builder.
6. Prompt bắt buộc trích nguồn và từ chối khi thiếu evidence.
7. API serving, caching, logging, latency/cost tracking.
8. Evaluation và regression tests mỗi khi đổi embedding/chunking/prompt.
9. Quan sát production: trace query → retrieved chunks → answer, theo dõi drift
   và version của dataset/embedding/prompt.

Evaluation set nên có bốn nhóm câu hỏi:

- Exact lookup: ISBN, tác giả, series.
- Semantic discovery: tìm sách theo nội dung/tâm trạng.
- Multi-filter: nội dung + genre + năm.
- Unanswerable/adversarial: thông tin không có trong kho.

Metric tối thiểu:

- Retrieval: Recall@k, MRR hoặc nDCG@k.
- Generation: groundedness/faithfulness, answer relevance, citation correctness.
- System: p50/p95 latency, error rate, token/cost per query.

## 5. Serving và portfolio

Streamlit hoặc API demo gồm:

- Analytics dashboard.
- Genre prediction.
- Book search/RAG có citation.
- Trang evaluation hiển thị metric thật.

Thứ tự học khuyến nghị:

`correctness → EDA/statistics → ML baseline → retrieval → RAG evaluation → serving/monitoring`.
