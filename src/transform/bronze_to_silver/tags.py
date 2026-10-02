from pyspark.sql import DataFrame
from pyspark.sql.functions import col


def transform(df: DataFrame) -> DataFrame:
    return (
        df.select(
            col("id"),
            col("tag"),
            col("slug"),
            col("count"),
            col("tag_category_id"),
            col("tag_category.category").alias("tag_category"),
            col("tag_category.slug").alias("tag_category_slug"),
        )
        .filter(col("id").isNotNull())
        .dropDuplicates(["id"])
    )