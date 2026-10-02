"""bronze_to_silver/platforms.py — static lookup, no updated_at."""
from pyspark.sql import DataFrame


def transform(df: DataFrame) -> DataFrame:
    return df.dropDuplicates(["id"])
