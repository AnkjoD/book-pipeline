"""
DAG: hardcover_ingestion_full
==============================
Fetch TOÀN BỘ lịch sử từ Hardcover API cho tất cả entity.

- Trigger: THỦ CÔNG (schedule=None) — chỉ chạy 1 lần đầu để seed data lake.
- Sau khi DAG này hoàn thành, chuyển sang dùng:
    • hardcover_ingestion_daily  — sync books/authors/editions mỗi ngày
    • hardcover_ingestion_weekly — sync series/tags/publishers mỗi Chủ nhật

Thứ tự chạy (sequential để không quá tải API):
    books → authors → editions → series → tags → publishers
    → countries → languages → reading_formats  (reference tables nhỏ)

Resume: nếu task bị fail/timeout, re-trigger DAG — checkpoint tự động
resume từ offset cuối cùng đã fetch thành công, không mất dữ liệu.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def _fetch(entity_key: str, mode: str, batch_size: int, checkpoint_dir: str):
    """
    Wrapper gọi run_fetch() với lazy import — tránh Airflow parse-time error
    khi scheduler đọc DAG file mà thư viện ingestion chưa load xong.
    """
    from ingestion.cli import run_fetch
    return run_fetch(
        entity_key=entity_key,
        mode=mode,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
    )


# Danh sách entity theo thứ tự ưu tiên:
#   - Fact tables lớn (books, editions) trước để nhanh có data
#   - Dimension tables nhỏ (series, tags...) sau
#   - Reference tables (countries...) cuối cùng
ENTITIES = [
    "books",
    "authors",
    "editions",
    "series",
    "tags",
    "publishers",
    "countries",
    "languages",
    "reading_formats",
]

CHECKPOINT_DIR = "/opt/airflow/checkpoints"
BATCH_SIZE = 100

with DAG(
    dag_id="hardcover_ingestion_full",
    description="Full fetch toàn bộ lịch sử Hardcover — trigger thủ công 1 lần",
    schedule=None,          # Không tự chạy, phải trigger tay
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,      # Không cho chạy song song 2 lần cùng lúc
    tags=["hardcover", "ingestion", "full"],
    doc_md=__doc__,
) as dag:

    tasks = []
    for entity in ENTITIES:
        task = PythonOperator(
            task_id=f"fetch_{entity}",
            python_callable=_fetch,
            op_kwargs={
                "entity_key": entity,
                "mode": "full",
                "batch_size": BATCH_SIZE,
                "checkpoint_dir": CHECKPOINT_DIR,
            },
            # Retry 2 lần nếu lỗi mạng thoáng qua (tenacity xử lý retry trong code,
            # đây là retry ở tầng Airflow task để handle crash/OOM)
            retries=2,
            retry_delay=timedelta(minutes=5),
            # Books ~2.7M records và giới hạn an toàn 100/request tương đương
            # khoảng 27.000 calls. Đặt timeout rộng; thời gian thật phải đo.
            execution_timeout=timedelta(hours=36),
        )
        tasks.append(task)

    # Chạy tuần tự: books → authors → editions → ...
    for i in range(1, len(tasks)):
        tasks[i - 1] >> tasks[i]
