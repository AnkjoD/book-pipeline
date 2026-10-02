"""
Base class cho mọi fetcher trong pipeline.

Chứa toàn bộ logic pagination + checkpoint + upload MinIO dùng chung.
Mỗi entity mới chỉ cần kế thừa và khai báo entity_name + query — xong.
Xem books_fetcher.py hoặc series_fetcher.py làm ví dụ (~10 dòng mỗi file).
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from clients import HardcoverClient, MinioClient

logger = logging.getLogger(__name__)


class BaseFetcher:
    """
    Fetcher tổng quát cho 1 entity Hardcover bất kỳ.

    Subclass cần định nghĩa:
        entity_name: str   — tên entity GraphQL, VD "books", "authors"
        query: str         — GraphQL query nhận $limit, $offset, $where
    """

    entity_name: str = None
    query: str = None
    # None cho entity không có updated_at (authors, series, tags, reference tables).
    incremental_field: str | None = "updated_at"
    api_max_batch_size: int = 100
    supports_count: bool = True
    def __init__(
        self,
        client: HardcoverClient,
        minio: MinioClient,
        checkpoint_dir: str = "checkpoints",
        batch_size: int = 100,
    ):
        if not self.entity_name or not self.query:
            raise NotImplementedError("Subclass phải định nghĩa entity_name và query")

        self.client = client
        self.minio = minio
        if batch_size < 1:
            raise ValueError("batch_size phải >= 1")
        if batch_size > self.api_max_batch_size:
            logger.warning(
                "[%s] batch_size=%s vượt giới hạn an toàn %s; tự giảm xuống %s",
                self.entity_name,
                batch_size,
                self.api_max_batch_size,
                self.api_max_batch_size,
            )
        self.batch_size = min(batch_size, self.api_max_batch_size)

        # Tách 2 file checkpoint để full-fetch và incremental-sync không đụng nhau:
        #   full_checkpoint  → lưu offset để resume khi fetch toàn bộ lịch sử
        #   sync_checkpoint  → lưu thời điểm để lọc updated_at khi chạy định kỳ
        self.full_checkpoint_path = Path(checkpoint_dir) / f"{self.entity_name}_full_checkpoint.json"
        self.sync_checkpoint_path = Path(checkpoint_dir) / f"{self.entity_name}_sync_checkpoint.json"
        self.full_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # ── FULL FETCH CHECKPOINT ────────────────────────────────────────────────

    def _load_full_checkpoint(self) -> tuple[int, str | None]:
        """Đọc offset và mốc bắt đầu gốc để resume không làm mất update."""
        if not self.full_checkpoint_path.exists():
            return 0, None

        data = json.loads(self.full_checkpoint_path.read_text(encoding="utf-8"))
        offset = data.get("last_offset", 0)
        total = data.get("total_count", "?")
        pct = data.get("progress_pct", "?")
        logger.info(
            f"[{self.entity_name}] Resume từ offset {offset:,}/{total} ({pct}%) "
            f"— checkpoint: {self.full_checkpoint_path}"
        )
        return offset, data.get("full_fetch_started_at")

    def _save_full_checkpoint(
        self,
        offset: int,
        total_count: int | None = None,
        batch_size: int | None = None,
        batch_number: int | None = None,
        full_fetch_started_at: str | None = None,
    ):
        """
        Lưu checkpoint sau mỗi batch thành công.
        Ghi atomic (tmp → rename) để tránh file bị corrupt nếu process bị kill giữa chừng.
        """
        data: dict = {
            "entity": self.entity_name,
            "last_offset": offset,
            "checkpoint_at": datetime.now(timezone.utc).isoformat(),
        }
        if batch_size is not None:
            data["batch_size"] = batch_size
        if batch_number is not None:
            data["batch_number"] = batch_number
        if full_fetch_started_at is not None:
            data["full_fetch_started_at"] = full_fetch_started_at
        if total_count:
            data["total_count"] = total_count
            data["fetched_count"] = offset
            data["progress_pct"] = round(offset / total_count * 100, 2)

        tmp = self.full_checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        # Windows: atomic rename fail nếu target đang bị lock → xóa trước
        try:
            tmp.replace(self.full_checkpoint_path)
        except PermissionError:
            self.full_checkpoint_path.unlink(missing_ok=True)
            tmp.rename(self.full_checkpoint_path)

    def _mark_full_fetch_done(self, total_fetched: int, full_fetch_started_at: str):
        """Đánh dấu full fetch hoàn tất để các lần chạy sau tự bỏ qua."""
        data = {
            "entity": self.entity_name,
            "last_offset": -1,
            "completed": True,
            "total_fetched": total_fetched,
            "full_fetch_started_at": full_fetch_started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self.full_checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            tmp.replace(self.full_checkpoint_path)
        except PermissionError:
            self.full_checkpoint_path.unlink(missing_ok=True)
            tmp.rename(self.full_checkpoint_path)

    def _is_full_fetch_done(self) -> bool:
        if not self.full_checkpoint_path.exists():
            return False
        data = json.loads(self.full_checkpoint_path.read_text(encoding="utf-8"))
        return data.get("completed", False)

    # ── INCREMENTAL SYNC CHECKPOINT ──────────────────────────────────────────

    def _load_last_sync_time(self) -> str | None:
        if not self.sync_checkpoint_path.exists():
            return None
        data = json.loads(self.sync_checkpoint_path.read_text(encoding="utf-8"))
        ts = data.get("last_sync_at")
        logger.info(f"[{self.entity_name}] Lần sync gần nhất: {ts}")
        return ts

    def _save_last_sync_time(self, ts: str):
        tmp = self.sync_checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"last_sync_at": ts}, indent=2), encoding="utf-8")
        try:
            tmp.replace(self.sync_checkpoint_path)
        except PermissionError:
            self.sync_checkpoint_path.unlink(missing_ok=True)
            tmp.rename(self.sync_checkpoint_path)

    # ── GHI BATCH LÊN MINIO ─────────────────────────────────────────────────

    def _write_batch(self, records: list[dict], run_tag: str, offset: int, batch_number: int | None = None):
        """
        Serialize records → JSONL bytes → upload thẳng lên MinIO, không ghi local.
        Tên file gồm run_tag + offset đảm bảo không bao giờ trùng nhau giữa các lần chạy.
        """
        batch_suffix = f"_batch{batch_number:03d}" if batch_number is not None else ""
        filename = f"{self.entity_name}_{run_tag}_offset{offset:07d}{batch_suffix}.jsonl"
        data = "\n".join(json.dumps(r, ensure_ascii=False) for r in records).encode("utf-8")
        self.minio.upload_bytes(self.entity_name, filename, data)

        batch_label = f"Batch {batch_number} | " if batch_number else ""
        logger.info(
            f"[{self.entity_name}] {batch_label}{len(records)} records "
            f"→ bronze/{self.entity_name}/{filename}"
        )

    # ── PUBLIC API ───────────────────────────────────────────────────────────

    def fetch_all(self, where: dict | None = None, total_count: int | None = None) -> int:
        """
        Full fetch toàn bộ historical data — chạy 1 lần, tự resume nếu bị ngắt.

        Sau khi xong, các lần gọi lại tự bỏ qua (completed=True trong checkpoint).
        Xóa file checkpoint nếu muốn fetch lại từ đầu.

        Args:
            where:       GraphQL where clause (None = lấy tất cả).
            total_count: Tổng số record — nên truyền từ client.count() trước,
                         dùng để tính % và ETA.
        Returns:
            Tổng số record đã fetch.
        """
        if self._is_full_fetch_done():
            # Với checkpoint V2, có thể khôi phục luôn mốc incremental nếu file
            # sync bị thất lạc. Checkpoint cũ không có mốc này thì giữ nguyên:
            # lần incremental đầu sẽ đọc lại an toàn thay vì có nguy cơ bỏ sót.
            if self.incremental_field and not self.sync_checkpoint_path.exists():
                checkpoint = json.loads(
                    self.full_checkpoint_path.read_text(encoding="utf-8")
                )
                original_start = checkpoint.get("full_fetch_started_at")
                if original_start:
                    self._save_last_sync_time(original_start)
                else:
                    logger.warning(
                        "[%s] Checkpoint cũ thiếu full_fetch_started_at; "
                        "incremental đầu tiên sẽ đọc lại toàn bộ time window.",
                        self.entity_name,
                    )
            logger.info(
                f"[{self.entity_name}] Full fetch đã hoàn tất trước đó — bỏ qua. "
                f"Xóa {self.full_checkpoint_path} để chạy lại từ đầu."
            )
            return 0

        start_offset, original_started_at = self._load_full_checkpoint()
        fetch_started_at = original_started_at or datetime.now(timezone.utc).isoformat()
        offset = start_offset
        fetched_total = start_offset
        batch_number = (start_offset // self.batch_size) + 1
        run_tag = "full"

        total_str = f"{total_count:,}" if total_count else "?"
        logger.info(
            f"[{self.entity_name}] ▶ Full fetch bắt đầu | "
            f"total: {total_str} | batch_size: {self.batch_size} | start_offset: {start_offset:,}"
        )

        start_time = time.monotonic()

        with tqdm(
            total=total_count,
            initial=start_offset,
            unit=" rec",
            desc=f"{self.entity_name}",
            dynamic_ncols=True,
        ) as pbar:
            while True:
                variables = {"limit": self.batch_size, "offset": offset, "where": where or {}}
                result = self.client.execute_query(self.query, variables)
                records = result["data"][self.entity_name]

                if not records:  # batch rỗng → hết data
                    logger.info(f"[{self.entity_name}] Không còn data tại offset {offset:,} — dừng.")
                    break

                self._write_batch(records, run_tag, offset, batch_number=batch_number)

                n = len(records)
                offset += n
                fetched_total = offset
                batch_number += 1

                self._save_full_checkpoint(
                    offset=offset,
                    total_count=total_count,
                    batch_size=self.batch_size,
                    batch_number=batch_number,
                    full_fetch_started_at=fetch_started_at,
                )
                pbar.update(n)

                # Log tiến độ đầy đủ — hữu ích khi đọc log trên Airflow/server
                if total_count:
                    elapsed = time.monotonic() - start_time
                    rate = (fetched_total - start_offset) / elapsed if elapsed > 0 else 0
                    remaining = (total_count - fetched_total) / rate if rate > 0 else float("inf")
                    eta_str = f"{remaining/60:.0f}m" if remaining < 3600 else f"{remaining/3600:.1f}h"
                    pct = fetched_total / total_count * 100
                    logger.info(
                        f"[{self.entity_name}] Batch {batch_number - 1} | "
                        f"+{n} records | offset {offset - n:,}→{offset:,} | "
                        f"{fetched_total:,}/{total_count:,} ({pct:.1f}%) | ETA ~{eta_str}"
                    )
                else:
                    logger.info(
                        f"[{self.entity_name}] Batch {batch_number - 1} | "
                        f"+{n} records | offset {offset:,}"
                    )

                # API Hardcover giới hạn tối đa 100 rec/call — không dùng n < batch_size
                # để detect hết data vì sẽ dừng sớm. Chỉ dừng khi records rỗng (ở đầu vòng).

        self._mark_full_fetch_done(
            total_fetched=fetched_total,
            full_fetch_started_at=fetch_started_at,
        )
        # Lấy mốc BẮT ĐẦU full fetch, không phải mốc kết thúc. Lần
        # incremental sau sẽ đọc lại các record thay đổi trong lúc full fetch
        # (Silver dedupe theo id + updated_at), nhờ đó không bỏ sót update.
        if self.incremental_field and not self.sync_checkpoint_path.exists():
            self._save_last_sync_time(fetch_started_at)
        elapsed_total = time.monotonic() - start_time
        logger.info(
            f"[{self.entity_name}] ✓ FULL FETCH HOÀN TẤT — "
            f"{fetched_total:,} records | {elapsed_total / 60:.1f} phút"
        )
        return fetched_total

    def sync_incremental(self, extra_where: dict | None = None) -> int:
        """
        Incremental sync cho chạy định kỳ (Airflow DAG hằng ngày, cron...).
        Chỉ fetch record có updated_at > lần sync trước — nhanh hơn full fetch nhiều.

        Lưu ý: entity phải có field 'updated_at'. Nên chạy fetch_all() ít nhất
        1 lần trước khi bắt đầu dùng sync_incremental().
        """
        if not self.incremental_field:
            raise ValueError(
                f"{self.entity_name} không có updated_at; không thể chạy incremental. "
                "Dùng full-fetch ban đầu hoặc snapshot refresh riêng."
            )

        sync_start_time = datetime.now(timezone.utc).isoformat()
        last_sync = self._load_last_sync_time()

        time_filter = {self.incremental_field: {"_lte": sync_start_time}}
        if last_sync:
            time_filter[self.incremental_field]["_gt"] = last_sync
        else:
            logger.warning(
                f"[{self.entity_name}] Chưa có sync trước — lần này chỉ chặn upper bound "
                f"(giống full fetch). Nên chạy fetch_all() trước."
            )

        # Dùng _and để không ghi đè filter cùng field do caller truyền vào.
        where = time_filter
        if extra_where:
            where = {"_and": [dict(extra_where), time_filter]}

        run_tag = f"sync_{sync_start_time.replace(':', '').replace('.', '')}"
        offset = 0
        fetched_total = 0
        batch_number = 1

        logger.info(
            f"[{self.entity_name}] ▶ Incremental sync bắt đầu | since: {last_sync or 'N/A'}"
        )

        while True:
            variables = {"limit": self.batch_size, "offset": offset, "where": where}
            result = self.client.execute_query(self.query, variables)
            records = result["data"][self.entity_name]

            if not records:
                break

            self._write_batch(records, run_tag, offset, batch_number=batch_number)
            offset += len(records)
            fetched_total += len(records)
            batch_number += 1

            # Không dừng khi n < batch_size — API giới hạn 100 rec/call

        # Lưu mốc thời gian sau khi toàn bộ batch upload xong — không lưu trước
        # để tránh bỏ sót record nếu job lỗi giữa chừng.
        self._save_last_sync_time(sync_start_time)
        logger.info(
            f"[{self.entity_name}] ✓ SYNC HOÀN TẤT — {fetched_total:,} records mới/cập nhật"
        )
        return fetched_total
