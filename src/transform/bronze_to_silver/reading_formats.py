from pyspark.sql import DataFrame


def transform(df: DataFrame) -> DataFrame:
    """Giữ lookup format chính thức thay vì hard-code mapping trong editions."""
    return df.dropDuplicates(["id"])
