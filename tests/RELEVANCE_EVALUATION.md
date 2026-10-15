# Chạy bộ đánh giá tìm sách

Phần này là công cụ test/đánh giá, không thay đổi luồng chatbot hay thêm retriever.
Các hàm tính điểm nằm trong `src/serve/rag/evaluation.py`; tên module cũ
`evalutation.py` vẫn dùng được để giữ bài tập đã viết.

Từ thư mục gốc, dùng môi trường `aiEnv`:

```powershell
conda activate aiEnv
python -m pytest -q tests/test_evaluation.py tests/test_relevance_runner.py
python -m scripts.evaluate_relevance
```

Unit test không cần dữ liệu/model. Đánh giá thật cần Gold DuckDB, collection
pgvector `books`, cache embedding hoặc mạng để tải model, và các dịch vụ mà
retriever đang dùng. Runner kiểm tra hash nguồn rồi gọi public `search()` hiện có.
Điểm thay đổi theo phiên bản retriever/dữ liệu, không cố định theo test assertions.

Output: `data/reports/relevance_baseline.json` và `.md` (dữ liệu runtime bị Git ignore).
Có thể đổi `--k`, `--dataset`, `--output`, hoặc chọn `--types exact semantic mixed`.

`tests/relevance_dataset.json` chứa 40 câu dev:

- 24 lookup tên/tác giả, nhãn lấy độc lập từ Gold.
- 10 semantic và 2 mixed, nhãn positive một phần dựa trên mô tả có bằng chứng.
- 4 câu không thể thỏa mãn hoặc đầu vào rỗng, chấm empty accuracy riêng.

Các nhãn semantic do AI hỗ trợ chuẩn bị, chưa được người dùng duyệt. Đây không
phải tập test giữ riêng và recall chỉ đo trên positive đã gán nhãn. SQL dùng toàn
bộ Gold còn vector dùng subset, nên chưa dùng điểm này để so thuật toán khác corpus.

Người dùng thực tế chỉ nhập câu hỏi. `relevant_book_ids` là đáp án của bộ đánh giá,
không gửi vào `search()` hay model. Lỗi dịch vụ không được tính là từ chối đúng.

Giữ dataset cố định khi so trước/sau. Chỉ dựng lại khi cần review thay đổi nguồn:

```powershell
python -m scripts.build_relevance_dataset
```

Để kiểm tra model hiểu câu sai ở đâu, chạy chẩn đoán chỉ đọc:

```powershell
python -m scripts.diagnose_query_understanding
```

Script lưu raw model, plan sau guard, bộ lọc thực thi, kết quả public và các phép
thử bỏ filter/dùng câu Anh vào `data/reports/query_understanding_diagnosis.json`
và `.md`. Plan được giữ cố định khi kiểm tra để không gọi model lần nữa rồi so
hai plan khác nhau. Các probe là thí nghiệm, không thay đổi retriever phục vụ.

Reranker tùy chọn đã có trong public search. So sánh trên cùng candidate pool:

```powershell
python -m scripts.benchmark_reranking
```

Báo cáo lưu tại `data/reports/reranking_comparison.json`, bao gồm smoke qua public
search/pipeline với reranker bật. Bộ 40 câu dev sau sửa parser có hit@5 36/36 câu
có positive, 4/4 unanswerable đúng; reranker giữ nguyên MRR/hit và thêm khoảng
0.53 giây trên mỗi câu được rerank. Vì vậy mặc định `RAG_RERANK_ENABLED=0`.
Không coi đây là đánh giá giữ riêng hoặc chứng minh mọi câu hỏi đều đúng.
