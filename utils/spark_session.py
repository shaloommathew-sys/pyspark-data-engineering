"""
utils/spark_session.py
----------------------
Reusable SparkSession factory.
Handles local, GCP Dataproc, and Databricks environments with a single entry point.
All pipeline files import get_spark_session() from here — avoids duplicate config code.
"""

from pyspark.sql import SparkSession
import os


def get_spark_session(app_name: str = "DataEngineering", env: str = "local") -> SparkSession:
    """
    Creates and returns a SparkSession configured for the target environment.

    Args:
        app_name: Name shown in Spark UI — use the pipeline name for easy debugging
        env: One of 'local', 'dataproc', 'databricks'

    Returns:
        Configured SparkSession
    """

    builder = SparkSession.builder.appName(app_name)

    if env == "local":
        builder = (
            builder
            .master("local[*]")                          # use all local CPU cores
            .config("spark.sql.shuffle.partitions", "8") # low partition count for local runs
            .config("spark.driver.memory", "2g")
        )

    elif env == "dataproc":
        # On GCP Dataproc, SparkSession picks up cluster config automatically.
        # We only override what's needed for our pipelines.
        builder = (
            builder
            .config("spark.sql.shuffle.partitions", "200")  # tune based on data volume
            .config("spark.sql.adaptive.enabled", "true")   # AQE: auto-optimises joins/shuffles
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            # GCS connector is pre-installed on Dataproc — no extra config needed
        )

    elif env == "databricks":
        # On Databricks, SparkSession already exists as 'spark' — just return it
        return SparkSession.getActiveSession()

    spark = builder.getOrCreate()

    # Suppress verbose INFO logs — only show warnings and errors
    spark.sparkContext.setLogLevel("WARN")

    return spark


# Quick smoke test — run this file directly to verify your Spark setup
if __name__ == "__main__":
    spark = get_spark_session(app_name="SmokeTest", env="local")
    print(f"Spark version: {spark.version}")
    print("SparkSession created successfully.")
    spark.stop()
