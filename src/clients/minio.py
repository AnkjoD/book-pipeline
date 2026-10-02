"""
MinioClient — kết nối MinIO, dùng chung cho ingestion, transform và bất kỳ
module nào trong pipeline cần đọc/ghi data lake.

Cấu trúc bucket:
    <bucket>/
        bronze/<entity>/<file>.jsonl     ← ingestion ghi vào
        silver/<entity>/part-*.parquet   ← Spark transform ghi vào

Dùng:
    client = MinioClient.from_env()
    client.upload_bytes("books", "books_full_offset0000000_batch001.jsonl", data_bytes)
"""

import io
import logging
import os
from pathlib import Path

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)


class MinioClient:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ):
        self.bucket = bucket
        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self._ensure_bucket()

    @classmethod
    def from_env(cls) -> "MinioClient":
        """Tạo MinioClient từ biến môi trường (.env). Dùng chung cho mọi entry-point."""
        return cls(
            endpoint=os.environ["MINIO_ENDPOINT"],
            access_key=os.environ["MINIO_ACCESS_KEY"],
            secret_key=os.environ["MINIO_SECRET_KEY"],
            bucket=os.environ.get("MINIO_BUCKET", "hardcover-lake"),
        )

    def _ensure_bucket(self):
        if not self._client.bucket_exists(self.bucket):
            self._client.make_bucket(self.bucket)
            logger.info(f"Đã tạo bucket: {self.bucket}")

    def upload_bytes(self, entity: str, filename: str, data: bytes):
        """
        Upload bytes thẳng từ RAM lên bronze/<entity>/<filename> — không ghi file tạm.
        Dùng sau mỗi batch fetch để stream thẳng lên MinIO, tránh tốn ổ đĩa.
        """
        object_name = f"bronze/{entity}/{filename}"
        self._client.put_object(
            self.bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type="application/x-ndjson",
        )
        logger.info(f"[{entity}] ✓ Upload: {object_name} ({len(data):,} bytes)")

    def upload_local_files(
        self,
        local_dir: str,
        entity: str,
        delete_after: bool = False,
    ) -> int:
        """
        Upload tất cả file .jsonl của entity từ thư mục local lên MinIO.
        Bỏ qua file đã tồn tại trên MinIO (idempotent).
        """
        files = sorted(Path(local_dir).glob(f"{entity}_*.jsonl"))
        if not files:
            logger.info(f"[{entity}] Không có file .jsonl trong {local_dir}")
            return 0

        existing = {
            obj.object_name.split("/")[-1]
            for obj in self._client.list_objects(
                self.bucket, prefix=f"bronze/{entity}/", recursive=True
            )
        }

        uploaded = 0
        for f in files:
            if f.name in existing:
                continue
            object_name = f"bronze/{entity}/{f.name}"
            try:
                self._client.fput_object(self.bucket, object_name, str(f))
                logger.info(f"[{entity}] ✓ Upload: {object_name}")
                uploaded += 1
                if delete_after:
                    f.unlink()
            except S3Error as e:
                logger.error(f"[{entity}] Lỗi upload {f.name}: {e}")

        logger.info(f"[{entity}] Upload xong: {uploaded}/{len(files)} file mới")
        return uploaded
