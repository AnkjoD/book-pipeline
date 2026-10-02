import logging
import sys
from transform.bronze_to_silver.shared import get_spark, read_bronze, write_silver
from transform.bronze_to_silver import (
    countries,
    editions,
    languages,
    platforms,
    publishers,
    reading_formats,
    series,
    tags,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_JOBS = {
    "series":     series.transform,
    "editions":   editions.transform,
    "publishers": publishers.transform,
    "tags":       tags.transform,
    "countries":  countries.transform,
    "languages":  languages.transform,
    "platforms":  platforms.transform,
    "reading_formats": reading_formats.transform,
}

if __name__ == "__main__":
    entity = sys.argv[1] if len(sys.argv) > 1 else None
    targets = {entity: _JOBS[entity]} if entity else _JOBS
    spark = get_spark(f"bronze_{'_'.join(targets)}")
    for name, fn in targets.items():
        write_silver(fn(read_bronze(spark, name)), name)
    spark.stop()
