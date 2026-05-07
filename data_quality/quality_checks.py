"""
data_quality/quality_checks.py
--------------------------------
Reusable data quality check framework.

Runs after each layer write to catch issues before they propagate downstream.
In production this is called by Airflow — if any CRITICAL check fails,
the DAG stops and alerts the on-call engineer.

Check severity levels:
- CRITICAL: pipeline stops immediately (e.g. zero rows, null primary key)
- WARNING:  pipeline continues but alert is sent (e.g. row count drop > 20%)
- INFO:     logged only, no alert (e.g. column stats for monitoring)
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from dataclasses import dataclass
from typing import List
from utils.spark_session import get_spark_session


@dataclass
class QualityCheckResult:
    """Holds the result of a single data quality check."""
    check_name:  str
    status:      str    # PASSED / FAILED
    severity:    str    # CRITICAL / WARNING / INFO
    actual:      str    # what we measured
    expected:    str    # what we expected
    message:     str    # human-readable summary


def check_not_empty(df: DataFrame, table_name: str) -> QualityCheckResult:
    """
    CRITICAL: The table must have at least 1 row.
    An empty table almost always means a pipeline failure upstream.
    """
    row_count = df.count()
    passed = row_count > 0

    return QualityCheckResult(
        check_name=f"{table_name}.not_empty",
        status="PASSED" if passed else "FAILED",
        severity="CRITICAL",
        actual=str(row_count),
        expected="> 0",
        message=f"Row count is {row_count}" if passed else f"EMPTY TABLE — got 0 rows"
    )


def check_no_nulls_in_column(df: DataFrame, column: str, table_name: str,
                              severity: str = "CRITICAL") -> QualityCheckResult:
    """
    Checks that a column has no null values.
    Used for primary keys and mandatory business fields.
    """
    null_count = df.filter(F.col(column).isNull()).count()
    passed = null_count == 0

    return QualityCheckResult(
        check_name=f"{table_name}.{column}.no_nulls",
        status="PASSED" if passed else "FAILED",
        severity=severity,
        actual=f"{null_count} nulls",
        expected="0 nulls",
        message=f"Column '{column}' has {null_count} null values"
    )


def check_no_duplicates(df: DataFrame, primary_key: str, table_name: str) -> QualityCheckResult:
    """
    Checks for duplicate primary key values.
    Duplicates in dimension tables cause fan-out (row multiplication) in joins.
    """
    total_rows  = df.count()
    unique_keys = df.select(primary_key).distinct().count()
    duplicate_count = total_rows - unique_keys
    passed = duplicate_count == 0

    return QualityCheckResult(
        check_name=f"{table_name}.{primary_key}.no_duplicates",
        status="PASSED" if passed else "FAILED",
        severity="CRITICAL",
        actual=f"{duplicate_count} duplicates",
        expected="0 duplicates",
        message=f"Found {duplicate_count} duplicate {primary_key} values"
    )


def check_row_count_vs_previous(current_count: int, previous_count: int,
                                 table_name: str, threshold_pct: float = 20.0) -> QualityCheckResult:
    """
    WARNING if today's row count dropped by more than threshold_pct vs yesterday.
    A 50% drop could mean a source system stopped sending data.
    A 200% increase could mean a duplication bug.
    """
    if previous_count == 0:
        return QualityCheckResult(
            check_name=f"{table_name}.row_count_drift",
            status="PASSED",
            severity="INFO",
            actual=str(current_count),
            expected="N/A (no previous count)",
            message="First run — no previous count to compare"
        )

    pct_change = abs((current_count - previous_count) / previous_count * 100)
    passed = pct_change <= threshold_pct

    return QualityCheckResult(
        check_name=f"{table_name}.row_count_drift",
        status="PASSED" if passed else "FAILED",
        severity="WARNING",
        actual=f"{current_count} rows ({pct_change:.1f}% change)",
        expected=f"Within {threshold_pct}% of {previous_count}",
        message=f"Row count changed by {pct_change:.1f}% vs previous run"
    )


def check_value_in_set(df: DataFrame, column: str, allowed_values: list,
                        table_name: str) -> QualityCheckResult:
    """
    Checks that a categorical column only contains expected values.
    e.g. order_status should only be COMPLETED, PENDING, CANCELLED.
    Any other value means an upstream system changed its enum without telling us.
    """
    unexpected = (
        df
        .filter(~F.col(column).isin(allowed_values) & F.col(column).isNotNull())
        .select(column)
        .distinct()
        .rdd.flatMap(lambda x: x)
        .collect()
    )
    passed = len(unexpected) == 0

    return QualityCheckResult(
        check_name=f"{table_name}.{column}.valid_values",
        status="PASSED" if passed else "FAILED",
        severity="WARNING",
        actual=f"Unexpected: {unexpected}" if unexpected else "All values valid",
        expected=f"Only: {allowed_values}",
        message=f"Found unexpected values in '{column}': {unexpected}"
    )


def run_quality_checks(results: List[QualityCheckResult]) -> bool:
    """
    Prints a quality report and returns False if any CRITICAL check failed.
    Airflow uses the return value to decide whether to proceed to the next task.
    """
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)

    has_critical_failure = False

    for result in results:
        icon = "✅" if result.status == "PASSED" else "❌"
        print(f"{icon} [{result.severity}] {result.check_name}")
        print(f"   {result.message}")
        print(f"   Expected: {result.expected} | Actual: {result.actual}")

        if result.status == "FAILED" and result.severity == "CRITICAL":
            has_critical_failure = True

    print("="*60)
    if has_critical_failure:
        print("RESULT: CRITICAL checks failed — pipeline should stop")
    else:
        print("RESULT: All critical checks passed")

    return not has_critical_failure


if __name__ == "__main__":
    spark = get_spark_session(app_name="QualityChecks", env="local")

    # Demo with sample data
    sample_data = [
        ("O001", "C001", 150.0, "COMPLETED"),
        ("O002", "C002", 200.0, "PENDING"),
        ("O003", None,   100.0, "CANCELLED"),   # null customer_id — will fail check
        ("O001", "C001", 150.0, "COMPLETED"),   # duplicate order_id — will fail check
    ]
    df = spark.createDataFrame(sample_data, ["order_id", "customer_id", "amount", "status"])

    results = [
        check_not_empty(df, "orders"),
        check_no_nulls_in_column(df, "customer_id", "orders", severity="CRITICAL"),
        check_no_duplicates(df, "order_id", "orders"),
        check_row_count_vs_previous(4, 5, "orders", threshold_pct=30.0),
        check_value_in_set(df, "status", ["COMPLETED", "PENDING", "CANCELLED"], "orders")
    ]

    pipeline_can_continue = run_quality_checks(results)
    print(f"\nPipeline proceed: {pipeline_can_continue}")

    spark.stop()
