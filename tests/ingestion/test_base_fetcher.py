import json

import pytest

from ingestion.fetchers.base_fetcher import BaseFetcher


class FakeClient:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def execute_query(self, query, variables):
        self.calls.append(variables)
        start = variables["offset"]
        end = start + variables["limit"]
        return {"data": {"books": self.records[start:end]}}


class FakeMinio:
    def __init__(self):
        self.uploads = []

    def upload_bytes(self, entity, filename, data):
        self.uploads.append((entity, filename, data))


class BooksFetcherForTest(BaseFetcher):
    entity_name = "books"
    query = "query FetchBooksForTest"


class StaticFetcherForTest(BooksFetcherForTest):
    incremental_field = None


def make_fetcher(tmp_path, records, batch_size=2, cls=BooksFetcherForTest):
    client = FakeClient(records)
    minio = FakeMinio()
    fetcher = cls(
        client=client,
        minio=minio,
        checkpoint_dir=str(tmp_path),
        batch_size=batch_size,
    )
    return fetcher, client, minio


def test_full_fetch_uploads_batches_and_marks_completed(tmp_path):
    records = [{"id": 1}, {"id": 2}, {"id": 3}]
    fetcher, client, minio = make_fetcher(tmp_path, records)

    assert fetcher.fetch_all(total_count=3) == 3
    assert [call["offset"] for call in client.calls] == [0, 2, 3]
    assert len(minio.uploads) == 2

    checkpoint = json.loads(fetcher.full_checkpoint_path.read_text())
    assert checkpoint["completed"] is True
    assert checkpoint["total_fetched"] == 3
    assert checkpoint["full_fetch_started_at"]

    sync_checkpoint = json.loads(fetcher.sync_checkpoint_path.read_text())
    assert sync_checkpoint["last_sync_at"] == checkpoint["full_fetch_started_at"]


def test_resume_preserves_original_full_fetch_start_time(tmp_path):
    records = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
    fetcher, client, _ = make_fetcher(tmp_path, records)
    original_start = "2026-01-01T01:00:00+00:00"
    fetcher._save_full_checkpoint(
        offset=2,
        total_count=4,
        batch_size=2,
        batch_number=2,
        full_fetch_started_at=original_start,
    )

    assert fetcher.fetch_all(total_count=4) == 4
    assert client.calls[0]["offset"] == 2

    checkpoint = json.loads(fetcher.full_checkpoint_path.read_text())
    assert checkpoint["full_fetch_started_at"] == original_start
    sync_checkpoint = json.loads(fetcher.sync_checkpoint_path.read_text())
    assert sync_checkpoint["last_sync_at"] == original_start


def test_incremental_uses_closed_time_window(tmp_path):
    fetcher, client, _ = make_fetcher(tmp_path, [{"id": 10}])
    previous_sync = "2026-01-01T00:00:00+00:00"
    fetcher._save_last_sync_time(previous_sync)

    assert fetcher.sync_incremental() == 1
    time_filter = client.calls[0]["where"]["updated_at"]
    assert time_filter["_gt"] == previous_sync
    assert "_lte" in time_filter


def test_incremental_rejected_without_updated_at(tmp_path):
    fetcher, _, _ = make_fetcher(tmp_path, [], cls=StaticFetcherForTest)
    with pytest.raises(ValueError, match="không có updated_at"):
        fetcher.sync_incremental()


def test_batch_size_is_clamped_to_safe_api_limit(tmp_path):
    fetcher, _, _ = make_fetcher(tmp_path, [], batch_size=500)
    assert fetcher.batch_size == 100
