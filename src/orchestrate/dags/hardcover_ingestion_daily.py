"""
DAG: hardcover_ingestion_daily
================================
Sync incremental hàng ngày lúc 3 giờ sáng cho các entity thay đổi thường xuyên.

Entity và chế độ:
    books     → incremental  (có updated_at, chỉ lấy sách mới/đổi từ hôm qua)
    editions  → incremental  (có updated_at)

Authors không có updated_at và full-fetch checkpoint sẽ tự bỏ qua sau lần
đầu, vì vậy không đặt authors trong DAG hằng ngày. Refresh authors là
snapshot job riêng khi thực sự cần.

Prerequisite: hardcover_ingestion_full phải đã chạy xong ít nhất 1 lần.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def _fetch(entity_key: str, mode: str, batch_size: int, checkpoint_dir: str):
    """Lazy import wrapper — xem giải thích trong hardcover_ingestion_full.py."""
    from ingestion.cli import run_fetch
    return run_fetch(
        entity_key=entity_key,
        mode=mode,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
    )


CHECKPOINT_DIR = "/opt/airflow/checkpoints"
BATCH_SIZE = 100

with DAG(
    dag_id="hardcover_ingestion_daily",
    description="Sync incremental hàng ngày: books, editions",
    schedule="0 3 * * *",   # Mỗi ngày lúc 3:00 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
        "execution_timeout": timedelta(hours=3),
    },
    tags=["hardcover", "ingestion", "daily"],
    doc_md=__doc__,
) as dag:

    # Books — incremental: chỉ lấy sách có updated_at > lần sync trước
    fetch_books = PythonOperator(
        task_id="sync_books",
        python_callable=_fetch,
        op_kwargs={
            "entity_key": "books",
            "mode": "incremental",
            "batch_size": BATCH_SIZE,
            "checkpoint_dir": CHECKPOINT_DIR,
        },
    )

    # Editions — incremental
    fetch_editions = PythonOperator(
        task_id="sync_editions",
        python_callable=_fetch,
        op_kwargs={
            "entity_key": "editions",
            "mode": "incremental",
            "batch_size": BATCH_SIZE,
            "checkpoint_dir": CHECKPOINT_DIR,
        },
    )

    # Books và editions cùng có updated_at; chạy tuần tự để giảm tải API.
    fetch_books >> fetch_editions
