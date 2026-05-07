"""
silver_layer/scd_type2.py
--------------------------
SCD Type 2 (Slowly Changing Dimension) implementation in PySpark.

What is SCD Type 2?
A pattern for tracking historical changes to dimension records.
Instead of overwriting old values, we close the old record and insert a new one.

Example — customer changes their city:
  BEFORE update:
    customer_id | city      | is_current | valid_from | valid_to
    C001        | Mumbai    | True       | 2023-01-01 | 9999-12-31

  AFTER update (city changed to Bangalore):
    customer_id | city      | is_current | valid_from | valid_to
    C001        | Mumbai    | False      | 2023-01-01 | 2024-05-07   ← closed
    C001        | Bangalore | True       | 2024-05-07 | 9999-12-31   ← new current

This lets you answer: "What city was customer C001 in on 2023-06-15?" 
by filtering: valid_from <= '2023-06-15' AND valid_to > '2023-06-15'
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from utils.spark_session import get_spark_session

# Sentinel date for "still active" records — far-future date
SCD_OPEN_DATE = "9999-12-31"


def identify_changes(existing_df: DataFrame, incoming_df: DataFrame,
                     primary_key: str, tracked_columns: list) -> DataFrame:
    """
    Compares incoming records against the current dimension table.
    Classifies each incoming record as:
    - NEW: primary key doesn't exist in current table
    - CHANGED: primary key exists but one of the tracked_columns has a different value
    - UNCHANGED: everything matches — no action needed

    Args:
        existing_df: current dimension table (is_current = True records only)
        incoming_df: today's updated dimension records
        primary_key: the business key column name (e.g. 'customer_id')
        tracked_columns: list of columns we care about for change detection
    """
    # Join incoming against existing on the primary key
    joined = incoming_df.alias("new").join(
        existing_df.alias("old"),
        on=primary_key,
        how="left"                          # left join: keeps all incoming, nulls for NEW records
    )

    # Build a change detection condition: any tracked column differs?
    change_condition = F.lit(False)
    for col in tracked_columns:
        change_condition = change_condition | (
            F.col(f"new.{col}") != F.col(f"old.{col}")
        )

    return (
        joined
        .withColumn(
            "_change_type",
            F.when(F.col(f"old.{primary_key}").isNull(), "NEW")
             .when(change_condition, "CHANGED")
             .otherwise("UNCHANGED")
        )
    )


def apply_scd2(existing_df: DataFrame, incoming_df: DataFrame,
               primary_key: str, tracked_columns: list,
               effective_date: str) -> DataFrame:
    """
    Applies SCD Type 2 logic:
    1. Closes changed records in existing table (sets valid_to = today, is_current = False)
    2. Inserts new records for NEW and CHANGED rows (valid_from = today, is_current = True)
    3. Keeps UNCHANGED records as-is

    Returns the full updated dimension table.
    """
    # Work only with currently active records
    current_df = existing_df.filter(F.col("is_current") == True)

    changes_df = identify_changes(current_df, incoming_df, primary_key, tracked_columns)

    # ── Step 1: Close changed records ────────────────────────────────────────
    # Find the primary keys of records that changed
    changed_keys = (
        changes_df
        .filter(F.col("_change_type") == "CHANGED")
        .select(primary_key)
    )

    # Update existing records where key is in changed_keys: set valid_to and is_current
    closed_records = (
        existing_df
        .join(changed_keys, on=primary_key, how="inner")
        .withColumn("valid_to",   F.lit(effective_date).cast("date"))
        .withColumn("is_current", F.lit(False))
    )

    # All existing records NOT in changed_keys remain untouched
    unchanged_existing = (
        existing_df
        .join(changed_keys, on=primary_key, how="left_anti")  # left_anti = "not in"
    )

    # ── Step 2: Insert new active records for NEW and CHANGED ─────────────────
    new_active_records = (
        changes_df
        .filter(F.col("_change_type").isin(["NEW", "CHANGED"]))
        .select([F.col(f"new.{c}").alias(c) for c in incoming_df.columns])
        .withColumn("valid_from",  F.lit(effective_date).cast("date"))
        .withColumn("valid_to",    F.lit(SCD_OPEN_DATE).cast("date"))
        .withColumn("is_current",  F.lit(True))
        .withColumn("_scd_updated_at", F.current_timestamp())
    )

    # ── Step 3: Union all parts back together ─────────────────────────────────
    final_df = unchanged_existing.unionByName(closed_records).unionByName(new_active_records)

    return final_df


if __name__ == "__main__":
    spark = get_spark_session(app_name="SCD2Demo", env="local")

    # Sample existing dimension table (what's currently in Silver)
    existing_data = [
        ("C001", "Priya Sharma",  "Mumbai",    "2023-01-01", "9999-12-31", True),
        ("C002", "Rahul Verma",   "Delhi",     "2023-01-01", "9999-12-31", True),
        ("C003", "Anita Nair",    "Chennai",   "2023-01-01", "9999-12-31", True),
    ]
    existing_df = spark.createDataFrame(
        existing_data,
        ["customer_id", "customer_name", "city", "valid_from", "valid_to", "is_current"]
    )

    # Incoming updates — Priya moved to Bangalore, new customer added
    incoming_data = [
        ("C001", "Priya Sharma",  "Bangalore"),   # CHANGED — city updated
        ("C002", "Rahul Verma",   "Delhi"),       # UNCHANGED
        ("C004", "Kiran Mehta",   "Pune"),        # NEW customer
    ]
    incoming_df = spark.createDataFrame(
        incoming_data,
        ["customer_id", "customer_name", "city"]
    )

    result_df = apply_scd2(
        existing_df=existing_df,
        incoming_df=incoming_df,
        primary_key="customer_id",
        tracked_columns=["city", "customer_name"],
        effective_date="2024-05-07"
    )

    print("\nFinal SCD2 dimension table:")
    result_df.orderBy("customer_id", "valid_from").show(truncate=False)

    spark.stop()
