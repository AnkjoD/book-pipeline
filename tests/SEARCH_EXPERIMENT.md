# Hybrid và local GraphRAG thử nghiệm

Giữ public baseline và index pgvector hiện tại; dựng artifact SQLite FTS5/BM25
và graph riêng trên 300 sách đã có trong index. Graph có title/credit/series từ
metadata và nhân vật/địa điểm từ 24 mô tả được kiểm tra literal quote.
Đây là local graph retrieval + dense, chưa phải bộ Microsoft GraphRAG global search.

```powershell
conda activate aiEnv
python -m scripts.build_search_experiment --books 300 --extract-books 24
python -m scripts.compare_search_strategies
python -m scripts.smoke_search_experiment
python -m pytest -q tests/test_search_experiment.py
```

Build cần Gold, pgvector và Ollama. Unit test không cần dịch vụ. Artifact/cache/report
lưu dưới data/ bị Git ignore. Benchmark cố định cùng corpus và query plan, so dense,
BM25+dense qua RRF, và graph+dense. Các câu quan hệ có metadata control độc lập.

Trong Streamlit, chọn **Cách tìm sách** ở sidebar. Mặc định vẫn là **Hiện tại**;
hai chế độ thử nghiệm chỉ tìm trong artifact nhỏ, không phải toàn bộ Gold/index.

Ví dụ chế độ graph:

```text
tìm sách cùng bộ với "Catching Fire"
other books with the same credited person as "The Martian"
```

Chế độ graph chỉ giữ sách có path hỗ trợ khi yêu cầu quan hệ rõ. Literal quote
check chưa chứng minh entity type/relationship đúng. Cùng tên có thể bị gộp nhầm;
đồ thị chưa đủ coverage. Bộ dev/quan hệ nhỏ không chứng minh chất lượng tổng quát.
Credit/series đã có cấu trúc cũng có thể giải bằng SQL JOIN.

Kết quả dev trên cùng 300 sách/44 câu: 4 câu quan hệ có hit@5 dense 2/4,
hybrid 3/4, graph 4/4; 36 câu answerable cũ đều hit ở cả ba cách. Toàn bộ
test local 146 passed. Không coi mức 100% của graph trên tập nhỏ này là chất lượng
tổng quát; baseline app vẫn giữ mặc định. Walkthrough chi tiết lưu local tại
`docs/WALKTHROUGH_HYBRID_GRAPHRAG.md` (docs bị Git ignore).
