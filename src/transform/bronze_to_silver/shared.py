"""Shared utils cho PySpark bronze→silver."""

import logging
import os

import boto3
from dotenv import load_dotenv
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.functions import col, row_number, to_timestamp

load_dotenv()

logger = logging.getLogger(__name__)

BUCKET = os.environ.get("MINIO_BUCKET", "hardcover-lake")


def get_spark(app_name: str, shuffle_partitions: int = 8) -> SparkSession:
    endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")

    if os.name == "nt":
        hadoop_home = os.environ.get("HADOOP_HOME", r"C:\hadoop")
        os.environ.setdefault("HADOOP_HOME", hadoop_home)
        os.environ["PATH"] = hadoop_home + r"\bin;" + os.environ.get("PATH", "")

    return (
        SparkSession.builder.appName(app_name)
        # PySpark 4.x bundles Hadoop 3.5.0 — phải dùng hadoop-aws:3.5.0
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262",
        )
        .config("spark.hadoop.fs.s3a.endpoint", f"http://{endpoint}")
        .config("spark.hadoop.fs.s3a.access.key", access_key)
        .config("spark.hadoop.fs.s3a.secret.key", secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
        .config("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")
        .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED")
        .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED")
        # tránh throttle khi đọc nhiều file song song
        .config("spark.hadoop.fs.s3a.connection.maximum", "100")
        # gom file nhỏ thành partition lớn, giảm số task
        .config("spark.sql.files.maxPartitionBytes", str(128 * 1024 * 1024))
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        .getOrCreate()
    )


def list_bronze_keys(entity: str) -> list[str]:
    """List tất cả s3a:// keys trong bronze/{entity}/."""
    endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
    s3 = boto3.client(
        "s3",
        endpoint_url=f"http://{endpoint}",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=f"bronze/{entity}/"
    ):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".jsonl"):
                keys.append(f"s3a://{BUCKET}/{obj['Key']}")
    return sorted(keys)


def read_bronze(spark: SparkSession, entity: str, max_files: int = 0) -> DataFrame:
    """Đọc bronze JSONL. max_files>0 → chỉ đọc N files đầu."""
    keys = list_bronze_keys(entity)
    if max_files > 0:
        keys = keys[:max_files]
    logger.info("read_bronze %s: %d files", entity, len(keys))
    return spark.read.option("mergeSchema", "true").json(keys)


def deduplicate_latest(
    df: DataFrame,
    key: str = "id",
    updated_at: str = "updated_at",
) -> DataFrame:
    """Giữ bản mới nhất, dedup theo key + updated_at."""
    window = Window.partitionBy(key).orderBy(to_timestamp(col(updated_at)).desc_nulls_last())
    return (
        df.withColumn("_dedupe_rank", row_number().over(window))
        .filter(col("_dedupe_rank") == 1)
        .drop("_dedupe_rank")
    )


def write_silver(df: DataFrame, table: str, mode: str = "overwrite") -> None:
    path = f"s3a://{BUCKET}/silver/{table}/"
    logger.info("write %s  mode=%s", path, mode)
    df.write.format("parquet").mode(mode).save(path)
    logger.info("✓ %s", table)
