"""
gold_layer/aggregations.py
---------------------------
Gold layer: business-ready aggregated datasets.

Gold = the final layer that BI tools, dashboards, and APIs query directly.
Rules:
- Read from Silver (clean, deduplicated)
- Apply business logic and aggregations
- Output is optimised for query performance (partitioned + small file count)
- Column names should be business-friendly — not technical

Two Gold tables built here:
1. daily_revenue_by_region  — for regional sales dashboards
2. customer_order_summary   — for customer 360 / CRM analytics
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from utils.spark_session import get_spark_session


def build_daily_revenue_by_region(silver_df: DataFrame) -> DataFrame:
    """
    Aggregates completed orders by region and date.
    Only includes COMPLETED orders — cancelled/pending orders don't count as revenue.

    Output columns:
    - order_date, region: dimensions (the "by what")
    - total_revenue: sum of order_amount for completed orders
    - total_orders: count of completed orders
    - avg_order_value: average order size (useful for trend analysis)
    """
    return (
        silver_df
        .filter(F.col("order_status") == "COMPLETED")          # revenue only from completed orders
        .filter(F.col("_dq_amount_valid") == True)             # exclude flagged bad amounts
        .groupBy("order_date", "region")
        .agg(
            F.sum("order_amount").alias("total_revenue"),
            F.count("order_id").alias("total_orders"),
            F.round(F.avg("order_amount"), 2).alias("avg_order_value"),
            F.max("_silver_processed_at").alias("_last_updated")  # lineage: when was this computed
        )
        .withColumn("total_revenue", F.round(F.col("total_revenue"), 2))
    )


def build_customer_order_summary(silver_df: DataFrame) -> DataFrame:
    """
    Customer-level order summary — one row per customer with lifetime stats.
    Used by CRM and customer analytics teams.

    Output columns:
    - customer_id: business key
    - total_lifetime_revenue: all-time completed order value
    - total_orders: total count of all orders (any status)
    - completed_orders: count of successful orders
    - first_order_date, last_order_date: customer tenure signals
    - favourite_region: region with most orders (mode)
    """
    # First compute region counts per customer for "favourite region"
    region_counts = (
        silver_df
        .groupBy("customer_id", "region")
        .agg(F.count("order_id").alias("region_order_count"))
    )

    # Use rank to pick the top region per customer
    from pyspark.sql.window import Window
    region_window = Window.partitionBy("customer_id").orderBy(F.col("region_order_count").desc())

    favourite_region = (
        region_counts
        .withColumn("_region_rank", F.rank().over(region_window))
        .filter(F.col("_region_rank") == 1)
        .select("customer_id", F.col("region").alias("favourite_region"))
    )

    # Main aggregation
    customer_summary = (
        silver_df
        .groupBy("customer_id")
        .agg(
            F.sum(
                F.when(F.col("order_status") == "COMPLETED", F.col("order_amount")).otherwise(0)
            ).alias("total_lifetime_revenue"),
            F.count("order_id").alias("total_orders"),
            F.sum(
                F.when(F.col("order_status") == "COMPLETED", 1).otherwise(0)
            ).alias("completed_orders"),
            F.min("order_date").alias("first_order_date"),
            F.max("order_date").alias("last_order_date"),
            F.current_timestamp().alias("_last_updated")
        )
        .withColumn("total_lifetime_revenue", F.round(F.col("total_lifetime_revenue"), 2))
    )

    # Join in favourite region
    return customer_summary.join(favourite_region, on="customer_id", how="left")


def write_gold(df: DataFrame, output_path: str, table_name: str) -> None:
    """
    Gold writes use overwrite mode — Gold tables are always fully recomputed.
    Unlike Bronze (append) and Silver (dynamic partition overwrite),
    Gold is rebuilt each run from the full Silver dataset.

    Why? Gold aggregations need global consistency — you can't partially update a SUM.
    """
    full_path = f"{output_path}/{table_name}"
    (
        df
        .coalesce(4)           # small number of output files — Gold tables are read frequently
        .write
        .mode("overwrite")
        .parquet(full_path)
    )
    print(f"Gold table '{table_name}' written to: {full_path}")


def run_gold_pipeline(silver_path: str, gold_path: str) -> None:
    """Main entry point. Called by Airflow after Silver pipeline succeeds."""
    spark = get_spark_session(app_name="GoldAggregations", env="local")

    silver_df = spark.read.parquet(silver_path)
    print(f"Silver records loaded: {silver_df.count()}")

    daily_revenue_df   = build_daily_revenue_by_region(silver_df)
    customer_summary_df = build_customer_order_summary(silver_df)

    write_gold(daily_revenue_df,    gold_path, "daily_revenue_by_region")
    write_gold(customer_summary_df, gold_path, "customer_order_summary")

    spark.stop()


if __name__ == "__main__":
    run_gold_pipeline(
        silver_path="data/output/silver/orders/",
        gold_path="data/output/gold/"
    )
