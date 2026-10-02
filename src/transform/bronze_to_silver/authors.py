from pyspark.sql import DataFrame
from pyspark.sql.functions import col

def transform(df: DataFrame) -> DataFrame:
    return (
        df
        .dropDuplicates(["id"])
        .filter(col("id").isNotNull())
        .filter(col("canonical_id").isNull())
        .filter(
            col("state").isNull()
            | (col("state") != "duplicate")
        )
        .drop("state", "canonical_id")
    )
