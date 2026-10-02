"""
bronze/books.py — transform bronze_books → silver tables.
"""
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, explode_outer, lit, when, to_date, to_timestamp
from transform.bronze_to_silver.shared import deduplicate_latest

# Mapping lấy trực tiếp từ hardcover_full_schema.{csv,json} đi kèm dự án.
# Không tự suy đoán nhãn vì category sai sẽ làm sai toàn bộ EDA/model phía sau.
_BOOK_CATEGORY = {
    1: "Book",
    2: "Novella",
    3: "Short Story",
    4: "Graphic Novel",
    5: "Fan Fiction",
    6: "Research Paper",
    7: "Poetry",
    8: "Collection",
    9: "Web Novel",
    10: "Light Novel",
}
_LITERARY_TYPE  = {1: "Fiction", 2: "Nonfiction"}


def _map_id(col_ref, mapping: dict, default="Unknown"):
    expr = None
    for k, v in mapping.items():
        expr = when(col_ref == k, v) if expr is None else expr.when(col_ref == k, v)
    return expr.otherwise(default) if expr is not None else lit(default)


def _main(df: DataFrame) -> DataFrame:
    return (
        df
        .withColumn("book_category", _map_id(col("book_category_id"), _BOOK_CATEGORY))
        .withColumn("literary_type", _map_id(col("literary_type_id"),  _LITERARY_TYPE))
        .withColumn("release_date",  to_date("release_date"))
        .withColumn("created_at",    to_timestamp("created_at"))
        .withColumn("updated_at",    to_timestamp("updated_at"))
        .drop("book_category_id", "literary_type_id", "book_status_id",
              "contributions", "book_series", "book_characters",
              "cached_tags", "taggings", "state", "canonical_id")
    )


def _authors(df: DataFrame) -> DataFrame:
    return (
        df.select(col("id").alias("book_id"), explode_outer("contributions").alias("c"))
        .select("book_id", col("c.author_id").alias("author_id"), col("c.contribution").alias("role"))
        .filter(col("author_id").isNotNull())
    )


def _series(df: DataFrame) -> DataFrame:
    return (
        df.select(col("id").alias("book_id"), explode_outer("book_series").alias("s"))
        .select("book_id", col("s.series_id").alias("series_id"), col("s.position").alias("position"))
        .filter(col("series_id").isNotNull())
    )


def _tag_features(df: DataFrame) -> DataFrame:
    def _cat(field, label):
        return (
            df.select(col("id").alias("book_id"), explode_outer(f"cached_tags.{field}").alias("t"))
            .select("book_id", col("t.tag").alias("tag_name"), col("t.count").alias("vote_count"),
                    lit(label).alias("category"))
            .filter(col("tag_name").isNotNull())
        )

    cw = (
        df.select(col("id").alias("book_id"),
                  explode_outer("`cached_tags`.`Content Warning`").alias("t"))
        .select("book_id", col("t.tag").alias("tag_name"), col("t.count").alias("vote_count"),
                lit("Content Warning").alias("category"))
        .filter(col("tag_name").isNotNull())
    )
    return _cat("Genre", "Genre").unionByName(_cat("Mood", "Mood")).unionByName(cw)


def _taggings(df: DataFrame) -> DataFrame:
    """Quan hệ chuẩn hóa book ↔ tag; không dùng cached aggregate thay cho truth."""
    return (
        df.select(col("id").alias("book_id"), explode_outer("taggings").alias("t"))
        .select(
            "book_id",
            col("t.id").alias("tagging_id"),
            col("t.tag_id").alias("tag_id"),
            col("t.spoiler").alias("spoiler"),
        )
        .filter(col("tag_id").isNotNull())
        .dropDuplicates(["book_id", "tag_id"])
    )


def _chars(df: DataFrame) -> DataFrame:
    return (
        df.select(col("id").alias("book_id"), explode_outer("book_characters").alias("bc"))
        .select("book_id", col("bc.character_id").alias("character_id"),
                col("bc.character.name").alias("character_name"), col("bc.position").alias("position"))
        .filter(col("character_id").isNotNull())
    )


def transform(df: DataFrame) -> dict:
    # DataFrame là lazy. Không cache rồi unpersist ngay trước khi runner gọi write(),
    # vì cache khi đó chưa hề được materialize và không mang lại lợi ích.
    # Một tập hợp hợp lệ dùng chung cho main và mọi bridge table. Nếu chỉ lọc
    # canonical/duplicate trong _main(), các bridge sẽ chứa orphan book_id.
    valid_books = (
        deduplicate_latest(df)
        .filter(col("book_status_id") == 1)
        .filter(col("canonical_id").isNull())
        .filter(col("state").isNull() | (col("state") != "duplicate"))
    )
    return {
        "books":        _main(valid_books),
        "book_authors": _authors(valid_books),
        "book_series":  _series(valid_books),
        "book_taggings":     _taggings(valid_books),
        "book_tag_features": _tag_features(valid_books),
        "book_characters":   _chars(valid_books),
    }
