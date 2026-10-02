# Bộ paper tham khảo cho Book Recommendation RAG

Thư mục này lưu bản PDF của các preprint được chọn từ:

- `D:\lap_trinh\my_project\ViRISK\tools\cs_CV_recent.json` — 756 bản ghi.
- `D:\lap_trinh\my_project\ViRISK\tools\cs_CL_recent.json` — 631 bản ghi.

Các PDF phục vụ nghiên cứu kiến trúc. Không phải dependency khi chạy ứng dụng.

## 📚 Danh mục

| File | Paper | Ý tưởng dùng trong dự án | Trạng thái áp dụng |
|---|---|---|---|
| `2609.15830v1_citeguard-rag.pdf` | [CiteGuard-RAG](https://arxiv.org/abs/2609.15830v1) | Hybrid retrieval, citation, sentence-level validation, regenerate/refuse | Sau retrieval baseline |
| `2609.14565v1_tatk.pdf` | [TATK](https://arxiv.org/abs/2609.14565v1) | Metadata-grounded candidate verification và top-M reranking | Mượn ý tưởng; chưa fine-tune/RL |
| `2609.16912v1_lit3r.pdf` | [Lit3R](https://arxiv.org/abs/2609.16912v1) | BM25+dense+RRF, reranking, evidence verification | Hybrid retrieval sau baseline |
| `2609.14245v1_attribution-compression.pdf` | [Attribution-Compression Frontier](https://arxiv.org/abs/2609.14245v1) | Không làm mất liên kết giữa claim, citation và source span khi nén context | Sau generation baseline |
| `2609.13856v1_shopease.pdf` | [ShopEase](https://arxiv.org/abs/2609.13856v1) | Ablation dense/lexical/RRF/cross-encoder; phức tạp hơn chưa chắc tốt hơn | Dùng cho thiết kế evaluation |
| `2609.16730v1_lsrep.pdf` | [LSREP](https://arxiv.org/abs/2609.16730v1) | Replay-based evaluation cho conversational memory | Giai đoạn memory |
| `2609.16073v1_immutable-past.pdf` | [The Immutable Past](https://arxiv.org/abs/2609.16073v1) | Xử lý preference cũ mâu thuẫn với cập nhật mới | Giai đoạn memory |
| `2609.12268v1_ear.pdf` | [EAR](https://arxiv.org/abs/2609.12268v1) | Retrieval unit nhỏ, parent context lớn; cần benchmark trước khi đổi chunking | Thử nghiệm sau baseline |

## ⚠️ Cách đọc kết quả

Đây là preprint mới. Kết quả trên legal QA, scientific QA, customer support hoặc Amazon Reviews không tự động chuyển sang dữ liệu sách Hardcover. Mọi kỹ thuật phải được kiểm tra trên evaluation set của dự án bằng Recall@k, MRR, nDCG@k, groundedness, citation correctness và latency.

Xem phân tích và quyết định áp dụng tại [`../RAG_RESEARCH_REPORT.md`](../RAG_RESEARCH_REPORT.md).
