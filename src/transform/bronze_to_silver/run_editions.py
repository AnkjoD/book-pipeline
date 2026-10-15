"""Transform toàn bộ Bronze editions sang Silver editions."""

import logging

from dotenv import load_dotenv

from transform.bronze_to_silver import editions
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
    keys = list_bronze_keys("editions")

    if not keys:
        raise RuntimeError("Không tìm thấy file Bronze cho entity editions.")

    logger.info(
        "editions: reading toàn bộ %d Bronze files",
        len(keys),
    )

    spark = get_spark(
        "bronze_editions",
        shuffle_partitions=2,
    )

    try:
        raw_editions = spark.read.option("mergeSchema", "true").json(keys)

        silver_editions = editions.transform(raw_editions)

        write_silver(
            silver_editions,
            "editions",
            mode="overwrite",
        )

        logger.info("ALL DONE editions")

    finally:
        spark.stop()
