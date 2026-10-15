"""
Unit tests cho BooksFetcher — pagination + checkpoint/resume.
Không gọi Hardcover API thật (toàn bộ mock).
"""

import json
from unittest.mock import MagicMock

from ingestion.fetchers.books_fetcher import BooksFetcher

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_record(i: int) -> dict:
    return {"id": i, "title": f"Book {i}", "updated_at": "2024-01-01T00:00:00Z"}


def _fake_client(batches: list[list[dict]]) -> MagicMock:
    """GraphQL client trả về từng batch theo thứ tự."""
    client = MagicMock()
    side_effects = [{"data": {"books": batch}} for batch in batches]
    client.execute_query.side_effect = side_effects
    client.count.return_value = sum(len(b) for b in batches)
    return client


def _fake_minio() -> MagicMock:
    return MagicMock()


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_fetch_all_single_batch(tmp_path):
    """Fetch xong trong 1 batch (n < batch_size → dừng khi rỗng)."""
    records = [_make_record(i) for i in range(3)]
    client = _fake_client([records, []])
    minio = _fake_minio()

    fetcher = BooksFetcher(client=client, minio=minio, checkpoint_dir=str(tmp_path), batch_size=10)
    total = fetcher.fetch_all(total_count=3)

    assert total == 3
    minio.upload_bytes.assert_called_once()
    # checkpoint ghi done
    cp = json.loads((tmp_path / "books_full_checkpoint.json").read_text())
    assert cp.get("completed") is True


def test_fetch_all_multi_batch(tmp_path):
    """Fetch qua 2 batch: batch đầu đầy (= batch_size), batch 2 thiếu, batch 3 rỗng → dừng."""
    batch1 = [_make_record(i) for i in range(5)]
    batch2 = [_make_record(i) for i in range(5, 8)]
    client = _fake_client([batch1, batch2, []])
    minio = _fake_minio()

    fetcher = BooksFetcher(client=client, minio=minio, checkpoint_dir=str(tmp_path), batch_size=5)
    total = fetcher.fetch_all(total_count=8)

    assert total == 8
    assert minio.upload_bytes.call_count == 2


def test_fetch_all_resume(tmp_path):
    """Nếu checkpoint tồn tại (offset=5), fetch tiếp từ offset đó."""
    # Viết checkpoint giả lập lần trước đã fetch 5 record
    cp_file = tmp_path / "books_full_checkpoint.json"
    cp_file.write_text(json.dumps({"last_offset": 5, "completed": False}), encoding="utf-8")

    batch2 = [_make_record(i) for i in range(5, 8)]
    client = _fake_client([batch2, []])
    minio = _fake_minio()

    fetcher = BooksFetcher(client=client, minio=minio, checkpoint_dir=str(tmp_path), batch_size=5)
    total = fetcher.fetch_all(total_count=8)

    assert total == 8  # fetched_total = start_offset(5) + 3 mới = 8
    # API gọi với offset=5
    call_vars = client.execute_query.call_args_list[0][0][1]
    assert call_vars["offset"] == 5


def test_fetch_all_skip_if_done(tmp_path):
    """Nếu done=True trong checkpoint, fetch_all trả về 0 ngay, không gọi API."""
    cp_file = tmp_path / "books_full_checkpoint.json"
    cp_file.write_text(json.dumps({"completed": True, "total_fetched": 100}), encoding="utf-8")

    client = _fake_client([])
    minio = _fake_minio()

    fetcher = BooksFetcher(client=client, minio=minio, checkpoint_dir=str(tmp_path), batch_size=10)
    total = fetcher.fetch_all()

    assert total == 0
    client.execute_query.assert_not_called()


def test_sync_incremental_uses_updated_at(tmp_path):
    """sync_incremental phải truyền updated_at filter vào GraphQL query."""
    # Ghi sync checkpoint
    sync_cp = tmp_path / "books_sync_checkpoint.json"
    sync_cp.write_text(json.dumps({"last_sync_at": "2024-06-01T00:00:00Z"}), encoding="utf-8")

    records = [_make_record(1)]
    client = _fake_client([records, []])  # batch 2 rỗng → dừng
    minio = _fake_minio()

    fetcher = BooksFetcher(client=client, minio=minio, checkpoint_dir=str(tmp_path), batch_size=10)
    total = fetcher.sync_incremental()

    assert total == 1
    call_vars = client.execute_query.call_args_list[0][0][1]
    assert call_vars["where"]["updated_at"]["_gt"] == "2024-06-01T00:00:00Z"
