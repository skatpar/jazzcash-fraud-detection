"""
Spark utility functions for fraud detection pipeline
"""
import yaml
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from typing import Dict, Any, Optional
from datetime import datetime


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file

    Args:
        config_path: Path to config file

    Returns:
        Dict containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_spark_session(
    app_name: str = "JazzCash-Fraud-Detection",
    config_path: str = "config/spark_config.yaml",
    jdbc_driver_path: Optional[str] = None
) -> SparkSession:
    """
    Create and configure Spark session

    Args:
        app_name: Application name
        config_path: Path to Spark configuration file
        jdbc_driver_path: Path to PostgreSQL JDBC driver

    Returns:
        SparkSession: Configured Spark session
    """
    # Load Spark configuration
    spark_config = load_config(config_path)

    # Get active profile
    active_profile = spark_config.get('active_profile', 'development')
    profile = spark_config['profiles'][active_profile]

    # Build Spark session
    builder = SparkSession.builder.appName(app_name)

    # Set master
    if 'master' in spark_config['spark']:
        builder = builder.master(spark_config['spark']['master'])

    # Add JDBC driver if provided
    if jdbc_driver_path:
        builder = builder.config("spark.jars", jdbc_driver_path)

    # Executor configuration
    builder = builder.config("spark.executor.memory", profile['executor_memory'])
    builder = builder.config("spark.executor.cores", profile['executor_cores'])
    builder = builder.config("spark.executor.instances", profile['executor_instances'])

    # Driver configuration
    driver_config = spark_config['spark']['driver']
    builder = builder.config("spark.driver.memory", driver_config['memory'])
    builder = builder.config("spark.driver.maxResultSize", driver_config['max_result_size'])

    # SQL configuration
    sql_config = spark_config['spark']['sql']
    builder = builder.config("spark.sql.shuffle.partitions", profile['shuffle_partitions'])
    builder = builder.config("spark.sql.adaptive.enabled", sql_config['adaptive_enabled'])
    builder = builder.config("spark.sql.adaptive.coalescePartitions.enabled",
                           sql_config['adaptive_coalesce_partitions'])
    builder = builder.config("spark.sql.broadcastTimeout", sql_config['broadcast_timeout'])
    builder = builder.config("spark.sql.autoBroadcastJoinThreshold",
                           sql_config['autoBroadcastJoinThreshold'])

    # Serialization
    builder = builder.config("spark.serializer", spark_config['spark']['serializer'])
    builder = builder.config("spark.kryoserializer.buffer.max",
                           spark_config['spark']['kryoserializer_buffer_max'])

    # Network timeout
    builder = builder.config("spark.network.timeout",
                           spark_config['spark']['network']['timeout'])

    # Dynamic allocation
    dynamic = spark_config['spark']['dynamic_allocation']
    if dynamic['enabled']:
        builder = builder.config("spark.dynamicAllocation.enabled", True)
        builder = builder.config("spark.dynamicAllocation.minExecutors", dynamic['min_executors'])
        builder = builder.config("spark.dynamicAllocation.maxExecutors", dynamic['max_executors'])
        builder = builder.config("spark.dynamicAllocation.initialExecutors",
                               dynamic['initial_executors'])

    # Create session
    spark = builder.getOrCreate()

    # Set checkpoint directory
    checkpoint_dir = spark_config['spark'].get('checkpoint_dir', '/tmp/spark-checkpoints')
    spark.sparkContext.setCheckpointDir(checkpoint_dir)

    return spark


def get_jdbc_url(config_path: str = "config/db_config.yaml") -> str:
    """
    Get JDBC URL from configuration

    Args:
        config_path: Path to database configuration file

    Returns:
        str: JDBC URL
    """
    db_config = load_config(config_path)
    pg_config = db_config['postgresql']

    jdbc_url = f"jdbc:postgresql://{pg_config['host']}:{pg_config['port']}/{pg_config['database']}"
    return jdbc_url


def get_connection_properties(config_path: str = "config/db_config.yaml") -> Dict[str, str]:
    """
    Get JDBC connection properties

    Args:
        config_path: Path to database configuration file

    Returns:
        Dict: Connection properties
    """
    db_config = load_config(config_path)
    pg_config = db_config['postgresql']

    properties = {
        "user": pg_config['user'],
        "password": pg_config['password'],
        "driver": pg_config['connection_properties']['driver']
    }

    return properties


def read_table_from_postgres(
    spark: SparkSession,
    table_name: str,
    predicates: Optional[list] = None,
    config_path: str = "config/db_config.yaml"
) -> DataFrame:
    """
    Read table from PostgreSQL database

    Args:
        spark: Spark session
        table_name: Name of the table to read
        predicates: List of predicates for parallel read
        config_path: Path to database configuration

    Returns:
        DataFrame: Loaded data
    """
    jdbc_url = get_jdbc_url(config_path)
    properties = get_connection_properties(config_path)

    if predicates:
        df = spark.read.jdbc(
            url=jdbc_url,
            table=table_name,
            predicates=predicates,
            properties=properties
        )
    else:
        df = spark.read.jdbc(
            url=jdbc_url,
            table=table_name,
            properties=properties
        )

    return df


def create_date_predicates(
    date_column: str,
    start_date: str,
    end_date: str,
    num_partitions: int = 10
) -> list:
    """
    Create date-based predicates for parallel reading

    Args:
        date_column: Name of date column
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        num_partitions: Number of partitions

    Returns:
        List of predicate strings
    """
    from datetime import datetime, timedelta

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    date_diff = (end - start).days
    partition_size = max(1, date_diff // num_partitions)

    predicates = []
    current_date = start

    while current_date <= end:
        next_date = min(current_date + timedelta(days=partition_size), end)

        predicate = f"{date_column} >= '{current_date.strftime('%Y-%m-%d')}' " \
                   f"AND {date_column} <= '{next_date.strftime('%Y-%m-%d')}'"
        predicates.append(predicate)

        current_date = next_date + timedelta(days=1)

    return predicates


def write_parquet(
    df: DataFrame,
    output_path: str,
    mode: str = "overwrite",
    partition_by: Optional[list] = None,
    coalesce: Optional[int] = None
):
    """
    Write DataFrame to Parquet format

    Args:
        df: DataFrame to write
        output_path: Output path
        mode: Write mode (overwrite, append, etc.)
        partition_by: Columns to partition by
        coalesce: Number of files to coalesce to
    """
    writer = df.write.mode(mode)

    if partition_by:
        writer = writer.partitionBy(*partition_by)

    if coalesce:
        df = df.coalesce(coalesce)
        writer = df.write.mode(mode)
        if partition_by:
            writer = writer.partitionBy(*partition_by)

    writer.parquet(output_path)


def optimize_dataframe(df: DataFrame, cache: bool = True) -> DataFrame:
    """
    Optimize DataFrame by caching and repartitioning

    Args:
        df: Input DataFrame
        cache: Whether to cache

    Returns:
        DataFrame: Optimized DataFrame
    """
    # Repartition if needed
    current_partitions = df.rdd.getNumPartitions()
    optimal_partitions = max(200, current_partitions)

    if current_partitions != optimal_partitions:
        df = df.repartition(optimal_partitions)

    if cache:
        df = df.cache()

    return df


def get_table_config(
    table_key: str,
    config_path: str = "config/db_config.yaml"
) -> Dict[str, Any]:
    """
    Get table configuration

    Args:
        table_key: Key for table in config
        config_path: Path to database configuration

    Returns:
        Dict: Table configuration
    """
    db_config = load_config(config_path)
    return db_config['tables'].get(table_key, {})


def log_df_stats(df: DataFrame, name: str, logger):
    """
    Log DataFrame statistics

    Args:
        df: DataFrame
        name: Name of DataFrame
        logger: Logger instance
    """
    count = df.count()
    num_partitions = df.rdd.getNumPartitions()

    logger.info(f"{name}:")
    logger.info(f"  Rows: {count:,}")
    logger.info(f"  Columns: {len(df.columns)}")
    logger.info(f"  Partitions: {num_partitions}")
