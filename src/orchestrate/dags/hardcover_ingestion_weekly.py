"""
DAG: hardcover_ingestion_weekly
=================================
Sync incremental mỗi Chủ nhật lúc 4 giờ sáng cho các dimension table.

Entity và lý do chạy tuần:
    publishers — dimension table nhỏ, ít thay đổi

Series và tags không có updated_at trong schema nên không được gọi
sync_incremental(). Hai entity này được full-fetch lần đầu và refresh
bằng snapshot job riêng khi cần.

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
    dag_id="hardcover_ingestion_weekly",
    description="Sync incremental mỗi Chủ nhật: publishers",
    schedule="0 4 * * 0",   # Mỗi Chủ nhật lúc 4:00 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
        "execution_timeout": timedelta(hours=2),
    },
    tags=["hardcover", "ingestion", "weekly"],
    doc_md=__doc__,
) as dag:

    fetch_publishers = PythonOperator(
        task_id="sync_publishers",
        python_callable=_fetch,
        op_kwargs={
            "entity_key": "publishers",
            "mode": "incremental",
            "batch_size": BATCH_SIZE,
            "checkpoint_dir": CHECKPOINT_DIR,
        },
    )

    fetch_publishers
