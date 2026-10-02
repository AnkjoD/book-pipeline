from pyspark.sql import DataFrame


def transform(df: DataFrame) -> DataFrame:
    """Chuẩn hóa lookup ngôn ngữ, một dòng cho mỗi id."""
    return df.dropDuplicates(["id"])
