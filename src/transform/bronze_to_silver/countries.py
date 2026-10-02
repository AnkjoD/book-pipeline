from pyspark.sql import DataFrame


def transform(df: DataFrame) -> DataFrame:
    """Chuẩn hóa lookup quốc gia, một dòng cho mỗi id."""
    return df.dropDuplicates(["id"])
