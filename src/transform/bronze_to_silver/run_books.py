import logging

from dotenv import load_dotenv

from transform.bronze_to_silver import books
from transform.bronze_to_silver.shared import (
    get_spark,
    list_bronze_keys,
    write_silver,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    keys = list_bronze_keys("books")

    if not keys:
        raise RuntimeError("Không tìm thấy file Bronze cho entity books.")

    logger.info(
        "books: reading toàn bộ %d Bronze files",
        len(keys),
    )

    spark = get_spark(
        "bronze_books",
        shuffle_partitions=2,
    )

    try:
        raw_books = spark.read.option("mergeSchema", "true").json(keys)

        silver_tables = books.transform(raw_books)

        for table_name, dataframe in silver_tables.items():
            logger.info("Đang ghi Silver table: %s", table_name)

            write_silver(
                dataframe,
                table_name,
                mode="overwrite",
            )

        logger.info("ALL DONE books")

    finally:
        spark.stop()
