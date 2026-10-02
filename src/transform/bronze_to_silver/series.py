from pyspark.sql import DataFrame
from pyspark.sql.functions import col

def transform(df: DataFrame) -> DataFrame:
    return (
        df
        .filter(col("state") == "active")
        .filter(col("canonical_id").isNull())
        .drop("state", "canonical_id")
    )
