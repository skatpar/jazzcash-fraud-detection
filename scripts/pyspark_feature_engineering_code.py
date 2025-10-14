from pyspark.sql import functions as F

# Load data from table
df = spark.table("public.stixor_iar_jul")

# Define event date for analysis
EVENT_DATE = "2024-07-30"

# Filter and select base columns
base = (
    df.filter(F.col("data_date") <= F.lit(EVENT_DATE))
    .select(
        "customer_msisdn", "data_date", "trx_amt", "trx_status", "trx_channel", "trx_type",
        "merchant_id", "reason_type", "pur_of_remit", "utility_company", "start_balance",
        "end_balance", "trans_initiate_time", "ac_from", "ac_to", "bill_ref_number",
        "fee", "fed", "trans_id", "ec"
    )
    .withColumn("data_date", F.to_date("data_date"))
)

# Create window flags and time-based features
event_date = F.to_date(F.lit(EVENT_DATE))
windowed = (
    base
    .withColumn("win_1d", F.col("data_date") >= F.date_sub(event_date, 1))
    .withColumn("win_3d", F.col("data_date") >= F.date_sub(event_date, 3))
    .withColumn("win_7d", F.col("data_date") >= F.date_sub(event_date, 7))
    .withColumn("win_15d", F.col("data_date") >= F.date_sub(event_date, 15))
    .withColumn("win_30d", F.col("data_date") >= F.date_sub(event_date, 30))
    .withColumn("trx_hour", F.hour("trans_initiate_time"))
    .withColumn(
        "trx_time_bucket",
        F.when((F.col("trx_hour") >= 0) & (F.col("trx_hour") < 6), "midnight")
        .when((F.col("trx_hour") >= 6) & (F.col("trx_hour") < 12), "morning")
        .when((F.col("trx_hour") >= 12) & (F.col("trx_hour") < 18), "afternoon")
        .otherwise("evening")
    )
)

# Optimize performance
windowed = windowed.repartition("ac_from")
windowed.cache()



def generate_window_aggs(window_flag: str, hours: int):
    """Generate aggregation functions for a specific time window."""
    return [
        # Basic transaction counts
        F.count(F.when(F.col(window_flag), True)).alias(f"tx_count_{window_flag}"),
        F.count(F.when(F.col(window_flag) & (F.col("trx_status") == "Completed"), True)).alias(f"tx_success_{window_flag}"),
        F.count(F.when(F.col(window_flag) & (F.col("trx_status") != "Completed"), True)).alias(f"tx_failed_{window_flag}"),
        F.countDistinct(F.when(F.col(window_flag), F.col("data_date"))).alias(f"active_days_{window_flag}"),
        (F.max(F.when(F.col(window_flag), F.col("trans_initiate_time"))) -
         F.min(F.when(F.col(window_flag), F.col("trans_initiate_time")))).alias(f"tx_span_{window_flag}"),

        # Transaction amount statistics
        F.sum(F.when(F.col(window_flag), F.col("trx_amt"))).alias(f"sum_trx_amt_{window_flag}"),
        F.avg(F.when(F.col(window_flag), F.col("trx_amt"))).alias(f"avg_trx_amt_{window_flag}"),
        F.max(F.when(F.col(window_flag), F.col("trx_amt"))).alias(f"max_trx_amt_{window_flag}"),
        F.min(F.when(F.col(window_flag), F.col("trx_amt"))).alias(f"min_trx_amt_{window_flag}"),
        F.stddev(F.when(F.col(window_flag), F.col("trx_amt"))).alias(f"stddev_trx_amt_{window_flag}"),

        # Balance statistics
        F.avg(F.when(F.col(window_flag), F.col("start_balance"))).alias(f"avg_start_balance_{window_flag}"),
        F.avg(F.when(F.col(window_flag), F.col("end_balance"))).alias(f"avg_end_balance_{window_flag}"),
        F.avg(F.when(F.col(window_flag), F.col("end_balance") - F.col("start_balance"))).alias(f"avg_balance_change_{window_flag}"),
        F.sum(F.when(F.col(window_flag), F.col("end_balance") - F.col("start_balance"))).alias(f"total_balance_change_{window_flag}"),

        # Unique counts for diversity metrics
        F.countDistinct(F.when(F.col(window_flag), F.col("trx_channel"))).alias(f"unique_channels_{window_flag}"),
        F.countDistinct(F.when(F.col(window_flag), F.col("trx_type"))).alias(f"unique_types_{window_flag}"),
        F.countDistinct(F.when(F.col(window_flag), F.col("merchant_id"))).alias(f"unique_merchants_{window_flag}"),
        F.countDistinct(F.when(F.col(window_flag), F.col("reason_type"))).alias(f"unique_reason_types_{window_flag}"),
        F.countDistinct(F.when(F.col(window_flag), F.col("pur_of_remit"))).alias(f"unique_purposes_{window_flag}"),
        F.countDistinct(F.when(F.col(window_flag), F.col("utility_company"))).alias(f"unique_utilities_{window_flag}"),

        # Failure and risk ratios
        (F.count(F.when(F.col(window_flag) & (F.col("trx_status") != "Completed"), True)).cast("float") /
         F.when(F.count(F.when(F.col(window_flag), True)) != 0,
                F.count(F.when(F.col(window_flag), True)))).alias(f"failure_ratio_{window_flag}"),

        # High-value transaction metrics
        F.count(F.when(F.col(window_flag) & (F.col("trx_amt") > 100000), True)).alias(f"high_value_count_{window_flag}"),
        (F.count(F.when(F.col(window_flag) & (F.col("trx_amt") > 100000), True)).cast("float") /
         F.when(F.count(F.when(F.col(window_flag), True)) != 0,
                F.count(F.when(F.col(window_flag), True)))).alias(f"high_value_ratio_{window_flag}"),

        # Average hourly metrics
        (F.count(F.when(F.col(window_flag), True)) / F.lit(hours)).alias(f"avg_hourly_tx_count_{window_flag}"),
        (F.sum(F.when(F.col(window_flag), F.col("trx_amt"))) / F.lit(hours)).alias(f"avg_hourly_tx_amt_{window_flag}")
    ]


# Generate aggregations for all time windows
aggregations = []
windows = [("win_1d", 24), ("win_3d", 72), ("win_7d", 168), ("win_15d", 360), ("win_30d", 720)]
for win_flag, hrs in windows:
    aggregations.extend(generate_window_aggs(win_flag, hrs))

# Apply aggregations and write results
agg = windowed.groupBy("ac_from").agg(*aggregations)
agg.write.mode("overwrite").parquet("/mnt/fraud/features_agg/")