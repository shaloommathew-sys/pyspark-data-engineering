# pyspark-data-engineering
PySpark transformation patterns: Bronze→Silver→Gold layer processing, data quality validation, SCD implementations, schema evolution handling

pyspark-data-engineering/
├── README.md
├── bronze_layer/
│   └── raw_ingestion.py
├── silver_layer/
│   ├── data_cleaning.py
│   └── scd_type2.py
├── gold_layer/
│   └── aggregations.py
├── data_quality/
│   └── quality_checks.py
└── utils/
    └── spark_session.py


# PySpark Data Engineering Patterns

A collection of production-ready PySpark patterns for building Medallion Lakehouse 
architectures (Bronze → Silver → Gold), implemented using Apache Spark and designed 
for GCP Dataproc / Databricks environments.

## What's covered

- **Bronze Layer** — Raw ingestion from GCS with schema enforcement
- **Silver Layer** — Data cleaning, deduplication, SCD Type 2 slowly changing dimensions
- **Gold Layer** — Aggregations and business-ready datasets
- **Data Quality** — Row-level validation, null checks, anomaly detection
- **Utils** — Reusable SparkSession factory with GCP configs

## Tech stack

- Apache Spark / PySpark 3.x
- GCP: Dataproc, GCS, BigQuery
- Databricks (Delta Lake compatible)
- Python 3.9+

## Architecture

```
Raw Source Data
      ↓
 [Bronze Layer]  → Raw landing, no transforms, schema on read
      ↓
 [Silver Layer]  → Cleaned, deduplicated, SCD2 history tracked
      ↓
 [Gold Layer]    → Aggregated, business-ready, analytics-optimised
```

## How to run locally

```bash
pip install pyspark==3.5.0
python utils/spark_session.py    # test your Spark setup
python bronze_layer/raw_ingestion.py
```

## Author

Shaloo Merin Mathew — Senior Data Engineer  
[LinkedIn](https://linkedin.com/in/shaloo-mathew)
