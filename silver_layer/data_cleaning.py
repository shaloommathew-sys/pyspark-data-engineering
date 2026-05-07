"""
silver_layer/data_cleaning.py
------------------------------
Silver layer: cleaning and standardisation.

Reads from Bronze (partitioned Parquet), applies business rules:
- Drops corrupt/unparseable records
- Standardises string casing and trims whitespace
- Casts and validates data types
- Deduplicates within the same batch (keeps latest by order_date)
- Filters out records that fail mandatory field checks

Output goes to Silver as clean, deduplicated Parquet — still 1 row per order_id,
no aggregation yet (that's Gold's job).
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from utils.spark_session import get_spark_session


def read_bronze_incremental(spark: SparkSession, bronze_path: str, ingestion_date: str) -> DataFrame:
    """
    Reads only today's Bronze partition — incremental pattern.
    Without this, we'd reprocess all historical data on every run.

    Args:
        ingestion_date: format 'yyyy-MM-dd', passed in by Airflow as execution_date
    """
    return (
        spark.read
        .parquet(bronze_path)
        .filter(F.col("_ingestion_date") == ingestion_date)
    )


def drop_corrupt_records(df: DataFrame) -> DataFrame:
    """
    Removes records that Spark flagged as unparseable during Bronze ingestion.
    The _corrupt_record column is non-null only for bad rows.
    """
    corrupt_count = df.filter(F.col("_corrupt_record").isNotNull()).count()
    if corrupt_count > 0:
        print(f"WARNING: Dropping {corrupt_count} corrupt records")

    return df.filter(F.col("_corrupt_record").isNull()).drop("_corrupt_record")


def enforce_mandatory_fields(df: DataFrame) -> DataFrame:
    """
    Drops records where mandatory business keys are null.
    order_id is the primary key — a null order_id is unprocessable.
    customer_id is required for downstream joins.
    """
    before_count = df.count()

    df_clean = df.filter(
        F.col("order_id").isNotNull() &
        F.col("customer_id").isNotNull()
    )

    dropped = before_count - df_clean.count()
    if dropped > 0:
        print(f"WARNING: Dropped {dropped} records with null mandatory fields")

    return df_clean


def standardise_strings(df: DataFrame) -> DataFrame:
    """
    Cleans string columns:
    - trim() removes leading/trailing whitespace (common in CSV exports)
    - upper() standardises status codes so 'completed', 'COMPLETED', 'Completed' all match

    This prevents silent join failures — 'completed' != 'COMPLETED' in Spark.
    """
    return (
        df
        .withColumn("order_status", F.upper(F.trim(F.col("order_status"))))
        .withColumn("region",        F.upper(F.trim(F.col("region"))))
        .withColumn("customer_id",   F.trim(F.col("customer_id")))
        .withColumn("product_id",    F.trim(F.col("product_id")))
    )


def validate_amounts(df: DataFrame) -> DataFrame:
    """
    Business rule: order_amount must be > 0.
    Negative or zero amounts indicate data errors (not refunds — those have their own flow).
    Invalid rows are flagged with a _dq_flag column rather than silently dropped,
    so the data quality layer can report on them.
    """
    return (
        df
        .withColumn(
            "_dq_amount_valid",
            F.when(F.col("order_amount") > 0, True).otherwise(False)
        )
    )


def deduplicate(df: DataFrame) -> DataFrame:
    """
    Handles duplicate order_ids within the same batch.
    Strategy: keep the record with the latest order_date.

    Why use Window instead of dropDuplicates()?
    - dropDuplicates() picks an arbitrary row — not deterministic
    - Window + rank() guarantees we always keep the latest record
    """
    window = Window.partitionBy("order_id").orderBy(F.col("order_date").desc())

    return (
        df
        .withColumn("_rank", F.rank().over(window))
        .filter(F.col("_rank") == 1)           # keep only rank 1 = latest record
        .drop("_rank")
    )


def add_silver_metadata(df: DataFrame) -> DataFrame:
    """Adds Silver processing timestamp for lineage tracking."""
    return df.withColumn("_silver_processed_at", F.current_timestamp())


def write_silver(df: DataFrame, output_path: str, ingestion_date: str) -> None:
    """
    Writes Silver output. Overwrites today's partition only (safe incremental write).
    'dynamic' partitionOverwriteMode means Spark only replaces the partitions it writes —
    other date partitions are untouched. This makes reruns safe (idempotent).
    """
    (
        df
        .write
        .option("partitionOverwriteMode", "dynamic")  # KEY: only overwrite today's partition
        .mode("overwrite")
        .partitionBy("_ingestion_date")
        .parquet(output_path)
    )
    print(f"Silver layer written to: {output_path}")


def run_silver_pipeline(bronze_path: str, silver_path: str, ingestion_date: str) -> None:
    """Main entry point. Called by Airflow after Bronze pipeline succeeds."""
    spark = get_spark_session(app_name="SilverCleaning", env="local")

    df = read_bronze_incremental(spark, bronze_path, ingestion_date)
    print(f"Bronze records read: {df.count()}")

    df = drop_corrupt_records(df)
    df = enforce_mandatory_fields(df)
    df = standardise_strings(df)
    df = validate_amounts(df)
    df = deduplicate(df)
    df = add_silver_metadata(df)

    write_silver(df, silver_path, ingestion_date)

    spark.stop()


if __name__ == "__main__":
    from datetime import date
    run_silver_pipeline(
        bronze_path="data/output/bronze/orders/",
        silver_path="data/output/silver/orders/",
        ingestion_date=str(date.today())
    )
