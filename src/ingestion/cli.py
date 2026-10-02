"""
Ingestion entrypoint — dùng được cả 2 cách:

    1. CLI (test thủ công, chạy 1 lần):
        python -m ingestion fetch --entity books
        python -m ingestion fetch --entity all --mode incremental

    2. Airflow PythonOperator (chạy theo lịch):
        from ingestion.main import run_fetch
        PythonOperator(task_id="fetch_books", python_callable=run_fetch,
                       op_kwargs={"entity_key": "books", "mode": "incremental", ...})

MỞ RỘNG THÊM ENTITY MỚI:
    1. Tạo file ingestion/fetchers/prompts_fetcher.py, viết PromptsFetcher(BaseFetcher) (~10 dòng)
    2. Thêm 1 dòng vào FETCHER_REGISTRY bên dưới
    Xong — pagination/checkpoint/CLI tự hoạt động, không cần sửa gì thêm.
"""

import argparse
import os
import logging

from dotenv import load_dotenv

from clients import HardcoverClient
from ingestion.fetchers.books_fetcher import BooksFetcher
from ingestion.fetchers.authors_fetcher import AuthorsFetcher
from ingestion.fetchers.series_fetcher import SeriesFetcher
from ingestion.fetchers.tags_fetcher import TagsFetcher
from ingestion.fetchers.editions_fetcher import EditionsFetcher
from ingestion.fetchers.publishers_fetcher import PublishersFetcher
from ingestion.fetchers.reference_fetchers import (
    CountriesFetcher,
    LanguagesFetcher,
    ReadingFormatsFetcher,
    PlatformsFetcher,
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# Nơi duy nhất cần sửa khi thêm entity mới — key là tên dùng trên CLI.
FETCHER_REGISTRY = {
    "books": BooksFetcher,
    "authors": AuthorsFetcher,
    "series": SeriesFetcher,
    "tags": TagsFetcher,
    "editions": EditionsFetcher,
    "publishers": PublishersFetcher,
    "countries": CountriesFetcher,
    "languages": LanguagesFetcher,
    "reading_formats": ReadingFormatsFetcher,
    # Optional/manual: chỉ hữu ích khi sau này ingest thêm book_mappings.
    "platforms": PlatformsFetcher,
}

# Domain catalog mặc định cho analytics/RAG. ``platforms`` vẫn có thể gọi
# trực tiếp nhưng không bị fetch vô ích khi người dùng yêu cầu pipeline core.
CORE_ENTITY_KEYS = (
    "books",
    "authors",
    "editions",
    "series",
    "tags",
    "publishers",
    "countries",
    "languages",
    "reading_formats",
)

# Filter mặc định cho từng entity khi fetch (tùy chọn, có thể để trống {})
DEFAULT_FILTERS = {
    # Ví dụ: chỉ fetch sách chưa bị xóa/deduped ngay từ Bronze (tùy bạn chọn
    # bật hay để trống — để trống nghĩa là fetch RAW toàn bộ, lọc ở dbt sau)
    "books": {},
    "authors": {},
}


def run_fetch(entity_key: str, mode: str = "full", batch_size: int = 100, checkpoint_dir: str = "checkpoints") -> int:
    """
    Hàm fetch chính — dùng được từ cả CLI lẫn Airflow PythonOperator.

    Args:
        entity_key:     Tên entity (phải có trong FETCHER_REGISTRY).
        mode:           "full" = fetch toàn bộ lịch sử (1 lần, có resume).
                        "incremental" = chỉ lấy record mới kể từ lần sync trước.
        batch_size:     Số record mỗi API call.
        checkpoint_dir: Thư mục lưu checkpoint file.
    Returns:
        Số record đã fetch.
    """
    api_key = os.environ.get("HARDCOVER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Chưa có HARDCOVER_API_KEY. Tạo file .env với: HARDCOVER_API_KEY=your_key_here"
        )

    # Khởi tạo minio TRƯỚC để fail sớm nếu cấu hình sai,
    # thay vì fetch xong hàng triệu record rồi mới phát hiện không upload được.
    from clients import MinioClient
    minio = MinioClient.from_env()

    client = HardcoverClient(api_key=api_key)
    fetcher_cls = FETCHER_REGISTRY[entity_key]
    fetcher = fetcher_cls(
        client=client,
        minio=minio,
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size,
    )

    where = DEFAULT_FILTERS.get(entity_key) or None

    if mode == "full":
        if fetcher.supports_count:
            try:
                total = client.count(entity_key, where=where)
                logger.info(f"[{entity_key}] Tổng số record: {total:,}")
            except Exception as e:
                logger.warning(
                    f"[{entity_key}] Không đếm được tổng số record ({e}) "
                    "— tiến độ sẽ không hiện %"
                )
                total = None
        else:
            total = None
            logger.info(
                f"[{entity_key}] API không hỗ trợ aggregate count "
                "— fetch không hiển thị phần trăm"
            )

        return fetcher.fetch_all(
            where=where,
            total_count=total,
        )

    elif mode == "incremental":
        return fetcher.sync_incremental(extra_where=where)

    else:
        raise ValueError(
            f"mode không hợp lệ: {mode!r}. "
            "Dùng 'full' hoặc 'incremental'."
        )

def main():
    parser = argparse.ArgumentParser(description="Hardcover data ingestion CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch data cho 1 hoặc tất cả entity")
    fetch_parser.add_argument(
        "--entity",
        required=True,
        choices=list(FETCHER_REGISTRY.keys()) + ["core", "all"],
        help="Entity muốn fetch; 'core' cho catalog mặc định; 'all' gồm cả optional",
    )
    fetch_parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="full",
        help=(
            "full: fetch toàn bộ historical data, có resume nếu bị ngắt, "
            "TỰ BỎ QUA nếu đã chạy xong trước đó. "
            "incremental: chỉ fetch record mới/updated kể từ lần sync trước "
            "-- dùng cho chạy ĐỊNH KỲ (cron/ofelia), phải chạy 'full' ít nhất 1 lần trước."
        ),
    )
    fetch_parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Số record/request; mặc định và giới hạn an toàn là 100.",
    )
    fetch_parser.add_argument("--checkpoint-dir", default="checkpoints")

    subparsers.add_parser("list-entities", help="Liệt kê các entity đang hỗ trợ fetch")

    args = parser.parse_args()

    if args.command == "list-entities":
        print("Các entity đang hỗ trợ fetch:")
        for name in FETCHER_REGISTRY:
            print(f"  - {name}")
        return

    if args.command == "fetch":
        selected = CORE_ENTITY_KEYS if args.entity == "core" else tuple(FETCHER_REGISTRY)
        if args.entity in {"core", "all"} and args.mode == "incremental":
            targets = [
                name for name in selected
                if (fetcher_cls := FETCHER_REGISTRY[name]).incremental_field
            ]
            skipped = [name for name in selected if name not in targets]
            logger.warning(
                "Bỏ qua entity không có updated_at trong mode incremental: %s",
                ", ".join(skipped),
            )
        else:
            targets = list(selected) if args.entity in {"core", "all"} else [args.entity]
        for entity_key in targets:
            logger.info(f"===== [{args.mode.upper()}] {entity_key} =====")
            count = run_fetch(
                entity_key=entity_key,
                mode=args.mode,
                batch_size=args.batch_size,
                checkpoint_dir=args.checkpoint_dir,
            )
            logger.info(f"===== XONG: {entity_key} — {count:,} records =====\n")
    return 0
