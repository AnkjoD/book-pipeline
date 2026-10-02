# Nâng từ Silver/Gold cũ lên V3.1

Không fetch lại Hardcover API. Bronze là dữ liệu gốc và được giữ nguyên.

## 1. Sao lưu cấu hình

Giữ lại `.env`, `checkpoints/` và thông tin MinIO hiện tại. Không copy đè các
thư mục này khi thay source code.

## 2. Cài source V3.1

```powershell
pip install -e ".[ingestion,transform,dev]"
```

## 3. Rebuild Silver

V3.1 đổi logic authors, editions và output tags; vì writer dùng `overwrite`,
chạy lại các job sau từ Bronze hiện có:

```powershell
python -m transform.bronze_to_silver.run_books
python -m transform.bronze_to_silver.run_authors
python -m transform.bronze_to_silver.run_dimensions
```

Sau bước này, MinIO phải có `silver/book_taggings`,
`silver/book_tag_features`, `silver/countries`, `silver/languages` và
`silver/reading_formats`.

## 4. Rebuild và test Gold

Đặt `DUCKDB_PATH` đến `data/warehouse/gold.duckdb`, rồi chạy:

```powershell
dbt run --project-dir .\src\transform\silver_to_gold --profiles-dir .\src\transform\silver_to_gold
dbt test --project-dir .\src\transform\silver_to_gold --profiles-dir .\src\transform\silver_to_gold
```

Kết quả mong đợi: 6 models, 21 tests, 14 sources; mọi test pass.

## 5. Không xóa Bronze

Chỉ Silver và Gold là dữ liệu dẫn xuất có thể rebuild. Không xóa bucket
`bronze/` hoặc checkpoint nếu chưa có bản sao lưu.
