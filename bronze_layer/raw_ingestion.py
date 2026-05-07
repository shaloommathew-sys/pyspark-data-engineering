"""
bronze_layer/raw_ingestion.py
-----------------------------
Bronze layer: raw data ingestion pattern.

Design principles:
- NO business transformations here. Bronze = exact copy of source.
- Schema is enforced at read time to catch upstream changes early.
- Every record gets ingestion metadata columns added (source, timestamp, filename).
- Output is partitioned by ingestion date so downstream layers can do incremental reads.

In production (GCP): source_path = "gs://your-bucket/raw/orders/"
                      output_path = "gs://your-bucket/bronze/orders/"
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType,
    DoubleType, IntegerType, TimestampType
)
from utils.spark_session import get_spark_session


# ----------------------------------------------------------------------------
# Schema definition
# Always define schema explicitly for Bronze — never infer from CSV/JSON.
# Reason: schema inference reads the entire file on every run (slow + expensive).
# If source adds a new column, explicit schema safely ignores it until you update here.
# ----------------------------------------------------------------------------
ORDERS_SCHEMA = StructType([
    StructField("order_id",      StringType(),    nullable=False),
    StructField("customer_id",   StringType(),    nullable=True),
    StructField("product_id",    StringType(),    nullable=True),
    StructField("order_amount",  DoubleType(),    nullable=True),
    StructField("order_status",  StringType(),    nullable=True),
    StructField("order_date",    TimestampType(), nullable=True),
    StructField("region",        StringType(),    nullable=True),
])


def read_raw_data(spark: SparkSession, source_path: str) -> DataFrame:
    """
    Reads raw CSV files from the source path with explicit schema.
    In production this would be GCS: gs://bucket/raw/orders/
    """
    df = (
        spark.read
        .option("header", "true")
        .option("mode", "PERMISSIVE")      # don't fail on bad rows — flag them instead
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .schema(ORDERS_SCHEMA)
        .csv(source_path)
    )
    return df


def add_bronze_metadata(df: DataFrame, source_system: str) -> DataFrame:
    """
    Adds standard Bronze metadata columns to every record.
    These columns are used by the Silver layer to track lineage and do incremental reads.

    Columns added:
    - _source_system: where the data came from (e.g. 'orders_api', 'db_export')
    - _ingestion_timestamp: exact time this record was loaded
    - _ingestion_date: date partition column (used for output partitioning)
    - _source_file: which file this record came from (useful for debugging bad data)
    """
    return (
        df
        .withColumn("_source_system",        F.lit(source_system))
        .withColumn("_ingestion_timestamp",   F.current_timestamp())
        .withColumn("_ingestion_date",        F.current_date())
        .withColumn("_source_file",           F.input_file_name())
    )


def write_bronze(df: DataFrame, output_path: str) -> None:
    """
    Writes Bronze output partitioned by ingestion date.

    Why Parquet?
    - Columnar format = much faster reads for analytical queries
    - 70-80% smaller than CSV
    - Preserves data types (no more "2024-01-01" being read as string)

    Why partition by _ingestion_date?
    - Silver layer reads only today's partition for incremental processing
    - Without partitioning, Silver would scan ALL historical data on every run
    """
    (
        df
        .write
        .mode("append")                        # append — never overwrite Bronze
        .partitionBy("_ingestion_date")        # partition for incremental downstream reads
        .parquet(output_path)
    )
    print(f"Bronze layer written to: {output_path}")


def run_bronze_pipeline(source_path: str, output_path: str, source_system: str) -> None:
    """
    Main entry point for the Bronze ingestion pipeline.
    Orchestrated by Airflow DAG in production.
    """
    spark = get_spark_session(app_name="BronzeIngestion", env="local")

    print(f"Reading raw data from: {source_path}")
    raw_df = read_raw_data(spark, source_path)

    print(f"Raw record count: {raw_df.count()}")

    enriched_df = add_bronze_metadata(raw_df, source_system)

    write_bronze(enriched_df, output_path)

    spark.stop()


if __name__ == "__main__":
    run_bronze_pipeline(
        source_path="data/sample/orders/",
        output_path="data/output/bronze/orders/",
        source_system="orders_api"
    )
