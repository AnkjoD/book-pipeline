import logging
from transform.bronze_to_silver.shared import get_spark, read_bronze, write_silver
from transform.bronze_to_silver import authors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    spark = get_spark("bronze_authors")
    write_silver(authors.transform(read_bronze(spark, "authors")), "authors")
    spark.stop()
