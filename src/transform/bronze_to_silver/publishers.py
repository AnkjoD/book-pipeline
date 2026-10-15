from pyspark.sql import DataFrame
from pyspark.sql.functions import col, to_timestamp

from transform.bronze_to_silver.shared import deduplicate_latest


def transform(df: DataFrame) -> DataFrame:
    return (
        deduplicate_latest(df)
        .filter(col("state").isNull() | (col("state") != "duplicate"))
        .filter(col("canonical_id").isNull())
        .withColumn("created_at", to_timestamp("created_at"))
        .withColumn("updated_at", to_timestamp("updated_at"))
        .drop("state", "canonical_id")
    )
