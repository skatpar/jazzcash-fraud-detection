# Libraries
Import all required libraries for data processing, machine learning, and model explainability including PySpark for distributed computing, scikit-learn for modeling, and SHAP for interpretability.


```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import variance, stddev, col, count, isnan, when
from pyspark.sql.types import NumericType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

```


```python
fraud_channels =  ['API', 'ATM', 'BIO', 'BISP', 'Business App', 'Business App API', 'Cheetay', 'JC Keyboard', 'Merchant Payment', 'Mobile App', 'NEW_JC_APP', 'PAYPAK', 'PGW', 'Payment Gateway', 'QR Payment', 'Self Care App', 'THIRD_PARTY_WEB', 'USSD', 'USSD API', 'USSD_API', 'VRG']
fraud_types =  ['Online Payment', 'PTS ATM Withdrawal', 'PTS Purchase Payment', 'Transfer(C2C)', 'MFS Card Withdraw', 'Cash in', 'Cash out', 'IBFT Outgoing Customer', 'IBFT Outgoing OTC', 'Merchant Payment', 'Transfer(B2C)', 'Transfer(C2B)', 'Jazz Load (Prepaid top-up)', 'Business Cash Out', 'Customer Remit To CNIC', 'Donation', 'Get Loan', 'IBFT Credit', 'Others', 'Utility Bills Payment', 'Purchase Payment', 'Auto Debit', 'Indigo Bills (Postpaid payment)']

channels_in_clause = ", ".join([f"'{ch}'" for ch in fraud_channels])
types_in_clause = ", ".join([f"'{tp}'" for tp in fraud_types])

CLICKHOUSE_CONFIG = {
    'host': 'localhost',
    'port': 9000,
    'database': 'public',
    'user': 'default',
    'password': 'DfsTeChB1'
}

url = f"jdbc:ch://{CLICKHOUSE_CONFIG['host']}:8123/{CLICKHOUSE_CONFIG['database']}"
user = CLICKHOUSE_CONFIG['user'] 
password = CLICKHOUSE_CONFIG['password']
driver = "com.clickhouse.jdbc.ClickHouseDriver"

packages = [
    "com.clickhouse.spark:clickhouse-spark-runtime-3.5_2.12:0.8.1",
    "com.clickhouse:clickhouse-client:0.9.4",
    "com.clickhouse:clickhouse-http-client:0.9.4",
    "org.apache.httpcomponents.client5:httpclient5:5.2.1"
]

spark = (SparkSession.builder
    .appName("poc-model-training")
    # .master("spark://10.205.161.118:7077")
    .master("local[*]")
    .config("spark.jars.packages", ",".join(packages))
    .config("spark.executor.memory", "150g")
    .config("spark.executor.memoryOverhead", "5g")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.cores", "32")
    .config("spark.executor.instances", "2")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.default.parallelism", "96")
    .getOrCreate()
)
    
# Configure ClickHouse catalog
spark.conf.set("spark.sql.catalog.clickhouse", "com.clickhouse.spark.ClickHouseCatalog")
spark.conf.set("spark.sql.catalog.clickhouse.host", "localhost")
spark.conf.set("spark.sql.catalog.clickhouse.protocol", "http")
spark.conf.set("spark.sql.catalog.clickhouse.http_port", "8123")
spark.conf.set("spark.sql.catalog.clickhouse.user", "default")
spark.conf.set("spark.sql.catalog.clickhouse.password", "DfsTeChB1")
spark.conf.set("spark.sql.catalog.clickhouse.database", "public")
spark.conf.set("spark.clickhouse.write.format", "json")           
```

    :: loading settings :: url = jar:file:/root/miniconda3/envs/fraud-spark/lib/python3.10/site-packages/pyspark/jars/ivy-2.5.1.jar!/org/apache/ivy/core/settings/ivysettings.xml


    Ivy Default Cache set to: /root/.ivy2/cache
    The jars for the packages stored in: /root/.ivy2/jars
    com.clickhouse.spark#clickhouse-spark-runtime-3.5_2.12 added as a dependency
    com.clickhouse#clickhouse-client added as a dependency
    com.clickhouse#clickhouse-http-client added as a dependency
    org.apache.httpcomponents.client5#httpclient5 added as a dependency
    :: resolving dependencies :: org.apache.spark#spark-submit-parent-0df457c4-ebaa-4014-9418-615fdfb5fc3e;1.0
    	confs: [default]
    	found com.clickhouse.spark#clickhouse-spark-runtime-3.5_2.12;0.8.1 in central
    	found com.clickhouse#clickhouse-client;0.9.4 in central
    	found com.clickhouse#clickhouse-data;0.9.4 in central
    	found commons-codec#commons-codec;1.17.1 in central
    	found commons-io#commons-io;2.16.1 in central
    	found org.apache.commons#commons-lang3;3.18.0 in central
    	found org.apache.commons#commons-compress;1.27.1 in central
    	found com.clickhouse#clickhouse-http-client;0.9.4 in central
    	found org.apache.httpcomponents.client5#httpclient5;5.4.4 in central
    	found org.apache.httpcomponents.core5#httpcore5;5.3.4 in central
    	found org.apache.httpcomponents.core5#httpcore5-h2;5.3.4 in central
    	found org.slf4j#slf4j-api;2.0.7 in central
    :: resolution report :: resolve 349ms :: artifacts dl 14ms
    	:: modules in use:
    	com.clickhouse#clickhouse-client;0.9.4 from central in [default]
    	com.clickhouse#clickhouse-data;0.9.4 from central in [default]
    	com.clickhouse#clickhouse-http-client;0.9.4 from central in [default]
    	com.clickhouse.spark#clickhouse-spark-runtime-3.5_2.12;0.8.1 from central in [default]
    	commons-codec#commons-codec;1.17.1 from central in [default]
    	commons-io#commons-io;2.16.1 from central in [default]
    	org.apache.commons#commons-compress;1.27.1 from central in [default]
    	org.apache.commons#commons-lang3;3.18.0 from central in [default]
    	org.apache.httpcomponents.client5#httpclient5;5.4.4 from central in [default]
    	org.apache.httpcomponents.core5#httpcore5;5.3.4 from central in [default]
    	org.apache.httpcomponents.core5#httpcore5-h2;5.3.4 from central in [default]
    	org.slf4j#slf4j-api;2.0.7 from central in [default]
    	:: evicted modules:
    	org.apache.httpcomponents.client5#httpclient5;5.2.1 by [org.apache.httpcomponents.client5#httpclient5;5.4.4] in [default]
    	---------------------------------------------------------------------
    	|                  |            modules            ||   artifacts   |
    	|       conf       | number| search|dwnlded|evicted|| number|dwnlded|
    	---------------------------------------------------------------------
    	|      default     |   13  |   0   |   0   |   1   ||   12  |   0   |
    	---------------------------------------------------------------------
    :: retrieving :: org.apache.spark#spark-submit-parent-0df457c4-ebaa-4014-9418-615fdfb5fc3e
    	confs: [default]
    	0 artifacts copied, 12 already retrieved (0kB/10ms)
    25/12/01 16:20:47 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
    25/12/01 16:20:48 WARN Utils: Service 'SparkUI' could not bind on port 4040. Attempting port 4041.
    25/12/01 16:20:48 WARN Utils: Service 'SparkUI' could not bind on port 4041. Attempting port 4042.


# Spark Session & ClickHouse Configuration
Initialize Spark session with optimized memory settings and configure ClickHouse catalog for distributed data access.

# Data Extraction
Define fraud-related channels and transaction types for filtering the dataset from ClickHouse.


```python
START_DATE = '2025-06-01'
END_DATE = '2025-06-30'
```

# Date Range Configuration
Set the training data date range (June 2025) for extracting transaction features.


```python
selected_cols = [
'cutoff_date',
'fraud_flag',
 'trx_channel',
 'trx_type',
 'start_balance',
 'trx_amt',
 'mbar_registered_channel',
 'mbar_a_c_status',
 'mbar_a_c_level',
 'mbar_account_type_name',
 'hour_of_day',
 'day_of_week',
 'is_weekend',
 'is_night',
 'is_business_hours',
 'is_unusual_hour',
 'night_weekend_combo',
 'start_balance_log',
 'txn_txns_3d',
 'txn_total_amount_3d',
 'txn_avg_amount_3d',
 'txn_max_amount_3d',
 'txn_min_amount_3d',
 'txn_unique_recipients_3d',
 'txn_unique_channels_3d',
 'txn_unique_types_3d',
 'txn_is_high_activity_3d',
 'txn_multi_channel_recent',
 'txn_amount_deviation_from_avg',
 'txn_night_txns_3d',
 'txn_weekend_txns_3d',
 'channel_new_jc_app',
 'channel_ussd',
 'channel_ussd_api',
 'channel_payment_gateway',
 'channel_mobile_app',
 'type_transfer_c2c',
 'type_transfer_c2b',
 'type_bill_payment',
 'type_mobile_load',
 'user_total_txns_3d',
 'user_total_amount_3d',
 'user_avg_amount_3d',
 'user_median_amount_3d',
 'user_max_amount_3d',
 'user_min_amount_3d',
 'user_unique_recipients_3d',
 'user_unique_channels_3d',
 'user_unique_types_3d',
 'user_total_txns_7d',
 'user_total_amount_7d',
 'user_avg_amount_7d',
 'user_median_amount_7d',
 'user_max_amount_7d',
 'user_min_amount_7d',
 'user_unique_recipients_7d',
 'user_unique_channels_7d',
 'user_unique_types_7d',
 'user_most_used_channel_7d',
 'user_last_used_channel',
 'user_channel_diversity_score_7d',
 'user_most_used_type_7d',
 'user_last_used_type',
 'user_type_diversity_score_7d',
 'user_night_txns_7d',
 'user_weekend_txns_7d',
 'user_peak_hour_txns_7d',
 'user_off_peak_hour_txns_7d',
 'user_avg_start_balance_7d',
 'user_avg_end_balance_7d',
 'user_min_balance_7d',
 'user_max_balance_7d',
 'user_balance_volatility_7d',
 'user_avg_amount_per_recipient_7d',
 'user_max_amount_to_single_recipient_7d',
 'user_recipient_concentration_ratio_7d',
 'user_avg_time_between_txns_7d',
 'user_txn_frequency_score_7d',
 'user_days_since_last_txn']


query = f"""
    SELECT {', '.join(selected_cols)}
    FROM clickhouse.public.stixor_fraud_features_distributed
    WHERE cutoff_date BETWEEN '{START_DATE}' AND '{END_DATE}'
        AND trx_channel IN ({channels_in_clause})
        AND trx_type IN ({types_in_clause})
        AND mbar_account_type_name = 'Customer Account'
"""


df = spark.sql(query)
```

# Query & Load Data
Execute SQL query to fetch selected features from the distributed fraud features table, filtering by date range, channels, types, and customer accounts only.


```python
df.show()
```

    25/12/01 16:20:52 WARN SparkStringUtils: Truncated the string representation of a plan since it was too large. This behavior can be adjusted by setting 'spark.sql.debug.maxToStringFields'.


    +-----------+----------+-----------+--------------------+-------------+-------+-----------------------+---------------+--------------+----------------------+-----------+-----------+----------+--------+-----------------+---------------+-------------------+------------------+-----------+-------------------+------------------+-----------------+-----------------+------------------------+----------------------+-------------------+-----------------------+------------------------+-----------------------------+-----------------+-------------------+------------------+------------+----------------+-----------------------+------------------+-----------------+-----------------+-----------------+----------------+------------------+--------------------+------------------+---------------------+------------------+------------------+-------------------------+-----------------------+--------------------+------------------+--------------------+------------------+---------------------+------------------+------------------+-------------------------+-----------------------+--------------------+-------------------------+----------------------+-------------------------------+----------------------+--------------------+----------------------------+------------------+--------------------+----------------------+--------------------------+-------------------------+-----------------------+-------------------+-------------------+--------------------------+--------------------------------+--------------------------------------+-------------------------------------+-----------------------------+---------------------------+------------------------+
    |cutoff_date|fraud_flag|trx_channel|            trx_type|start_balance|trx_amt|mbar_registered_channel|mbar_a_c_status|mbar_a_c_level|mbar_account_type_name|hour_of_day|day_of_week|is_weekend|is_night|is_business_hours|is_unusual_hour|night_weekend_combo| start_balance_log|txn_txns_3d|txn_total_amount_3d| txn_avg_amount_3d|txn_max_amount_3d|txn_min_amount_3d|txn_unique_recipients_3d|txn_unique_channels_3d|txn_unique_types_3d|txn_is_high_activity_3d|txn_multi_channel_recent|txn_amount_deviation_from_avg|txn_night_txns_3d|txn_weekend_txns_3d|channel_new_jc_app|channel_ussd|channel_ussd_api|channel_payment_gateway|channel_mobile_app|type_transfer_c2c|type_transfer_c2b|type_bill_payment|type_mobile_load|user_total_txns_3d|user_total_amount_3d|user_avg_amount_3d|user_median_amount_3d|user_max_amount_3d|user_min_amount_3d|user_unique_recipients_3d|user_unique_channels_3d|user_unique_types_3d|user_total_txns_7d|user_total_amount_7d|user_avg_amount_7d|user_median_amount_7d|user_max_amount_7d|user_min_amount_7d|user_unique_recipients_7d|user_unique_channels_7d|user_unique_types_7d|user_most_used_channel_7d|user_last_used_channel|user_channel_diversity_score_7d|user_most_used_type_7d| user_last_used_type|user_type_diversity_score_7d|user_night_txns_7d|user_weekend_txns_7d|user_peak_hour_txns_7d|user_off_peak_hour_txns_7d|user_avg_start_balance_7d|user_avg_end_balance_7d|user_min_balance_7d|user_max_balance_7d|user_balance_volatility_7d|user_avg_amount_per_recipient_7d|user_max_amount_to_single_recipient_7d|user_recipient_concentration_ratio_7d|user_avg_time_between_txns_7d|user_txn_frequency_score_7d|user_days_since_last_txn|
    +-----------+----------+-----------+--------------------+-------------+-------+-----------------------+---------------+--------------+----------------------+-----------+-----------+----------+--------+-----------------+---------------+-------------------+------------------+-----------+-------------------+------------------+-----------------+-----------------+------------------------+----------------------+-------------------+-----------------------+------------------------+-----------------------------+-----------------+-------------------+------------------+------------+----------------+-----------------------+------------------+-----------------+-----------------+-----------------+----------------+------------------+--------------------+------------------+---------------------+------------------+------------------+-------------------------+-----------------------+--------------------+------------------+--------------------+------------------+---------------------+------------------+------------------+-------------------------+-----------------------+--------------------+-------------------------+----------------------+-------------------------------+----------------------+--------------------+----------------------------+------------------+--------------------+----------------------+--------------------------+-------------------------+-----------------------+-------------------+-------------------+--------------------------+--------------------------------+--------------------------------------+-------------------------------------+-----------------------------+---------------------------+------------------------+
    | 2025-06-19|         0| NEW_JC_APP|       Transfer(C2C)|     12476.84| 2040.0|                    BIO|         Active|       Level 1|      Customer Account|          6|          4|         0|       1|                0|              0|                  0| 9.431629405906346|          1|             6000.0|            6000.0|           6000.0|           6000.0|                       1|                     1|                  1|                      0|                       0|                        -0.66|                0|                  0|                 1|           0|               0|                      0|                 0|                1|                0|                0|               0|                 1|             1589.51|           1589.51|              1589.51|           1589.51|           1589.51|                        1|                      1|                   1|                 1|             1589.51|           1589.51|              1589.51|           1589.51|           1589.51|                        1|                      1|                   1|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|       Transfer(C2C)|                         0.0|                 0|                   0|                     0|                         1|                  1589.51|                7589.51|            1589.51|            7589.51|                       0.0|                         1589.51|                               1589.51|                                  0.2|                         5.75|                        2.5|                       0|
    | 2025-06-19|         0| NEW_JC_APP|       Transfer(C2C)|     11436.37| 1000.0|                    BIO|         Active|       Level 1|      Customer Account|         16|          4|         0|       0|                1|              0|                  0| 9.344553908318396|          1|             6000.0|            6000.0|           6000.0|           6000.0|                       1|                     1|                  1|                      0|                       0|          -0.8333333333333334|                0|                  0|                 1|           0|               0|                      0|                 0|                1|                0|                0|               0|                 1|             1589.51|           1589.51|              1589.51|           1589.51|           1589.51|                        1|                      1|                   1|                 1|             1589.51|           1589.51|              1589.51|           1589.51|           1589.51|                        1|                      1|                   1|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|       Transfer(C2C)|                         0.0|                 0|                   0|                     0|                         1|                  1589.51|                7589.51|            1589.51|            7589.51|                       0.0|                         1589.51|                               1589.51|                                  0.2|                         5.75|                        2.5|                       0|
    | 2025-06-20|         0| NEW_JC_APP|       Transfer(C2C)|         0.05| 1000.0|                    BIO|         Active|       Level 1|      Customer Account|         12|          5|         0|       0|                1|              0|                  0|               0.0|          5|            11340.0|            2268.0|           6000.0|            300.0|                       5|                     1|                  1|                      0|                       0|          -0.5590828924162258|                1|                  0|                 1|           0|               0|                      0|                 0|                1|                0|                0|               0|                 5|            25906.27|          5181.254|              1589.51|          12476.84|              0.84|                        5|                      1|                   1|                 5|            25906.27|          5181.254|              1589.51|          12476.84|              0.84|                        5|                      1|                   1|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|       Transfer(C2C)|                         0.0|                 1|                   0|                     2|                         3|                 5181.254|      7449.253999999999|               0.84|           14516.84|          5566.39646790848|                        5181.254|                              12476.84|                  0.16666666666666666|                          8.0|                        2.0|                       0|
    | 2025-06-21|         0| NEW_JC_APP|       Transfer(C2B)|      8569.52| 1000.0|                    BIO|         Active|       Level 1|      Customer Account|         20|          6|         1|       0|                0|              0|                  0| 9.055967000890089|          6|            12340.0|2056.6666666666665|           6000.0|            300.0|                       6|                     1|                  1|                      0|                       0|          -0.5137763371150729|                1|                  0|                 1|           0|               0|                      0|                 0|                0|                1|                0|               0|                 6|            25906.32|           4317.72|               996.11|          12476.84|              0.05|                        6|                      1|                   1|                 6|            25906.32|           4317.72|               996.11|          12476.84|              0.05|                        6|                      1|                   1|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|       Transfer(C2B)|          0.8378378378378378|                 1|                   0|                     3|                         3|                  4317.72|      6374.386666666666|               0.05|           14516.84|          5435.90803920866|                         4317.72|                              12476.84|                  0.14285714285714285|                         12.0|                       1.75|                       0|
    | 2025-06-24|         0| NEW_JC_APP|       Transfer(C2C)|          0.0|    1.0|                    BIO|         Active|       Level 1|      Customer Account|          7|          2|         0|       0|                0|              0|                  0|               0.0|          3|             2530.0| 843.3333333333334|           1020.0|            510.0|                       2|                     1|                  2|                      0|                       0|          -0.9988142292490119|                0|                  2|                 1|           0|               0|                      0|                 0|                1|                0|                0|               0|                 3|            38590.18|12863.393333333333|              8569.52|          23737.09|           6283.57|                        2|                      1|                   2|                 9|             64496.5| 7166.277777777777|              6283.57|          23737.09|              0.05|                        7|                      1|                   2|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|       Transfer(C2C)|          0.9047619047619048|                 1|                   2|                     3|                         6|        7166.277777777777|      8596.277777777777|               0.05|           24247.09|        7478.2673662030365|               7166.277777777777|                              23737.09|                  0.23076923076923078|                        11.75|         1.8571428571428572|                       0|
    | 2025-06-24|         0| NEW_JC_APP|       Transfer(C2C)|         41.9| 1000.0|                    BIO|         Active|       Level 1|      Customer Account|         17|          2|         0|       0|                1|              0|                  0|3.7352858270344256|          3|             2530.0| 843.3333333333334|           1020.0|            510.0|                       2|                     1|                  2|                      0|                       0|          0.18577075098814225|                0|                  2|                 1|           0|               0|                      0|                 0|                1|                0|                0|               0|                 3|            38590.18|12863.393333333333|              8569.52|          23737.09|           6283.57|                        2|                      1|                   2|                 9|             64496.5| 7166.277777777777|              6283.57|          23737.09|              0.05|                        7|                      1|                   2|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|       Transfer(C2C)|          0.9047619047619048|                 1|                   2|                     3|                         6|        7166.277777777777|      8596.277777777777|               0.05|           24247.09|        7478.2673662030365|               7166.277777777777|                              23737.09|                  0.23076923076923078|                        11.75|         1.8571428571428572|                       0|
    | 2025-06-25|         0| NEW_JC_APP|Jazz Load (Prepai...|      9188.52|  900.0|                    BIO|         Active|       Level 1|      Customer Account|         13|          3|         0|       0|                1|              0|                  0| 9.125710157622516|          6|             5681.0| 946.8333333333334|           3000.0|              1.0|                       5|                     1|                  3|                      0|                       0|         -0.04946312268966735|                0|                  1|                 1|           0|               0|                      0|                 0|                0|                0|                0|               1|                 6|             41989.6| 6998.266666666666|              5963.52|          23737.09|               0.0|                        5|                      1|                   3|                12|            74875.93| 6239.660833333332|              5963.52|          23737.09|               0.0|                        9|                      1|                   3|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|Utility Bills Pay...|          0.9289617486338798|                 1|                   2|                     5|                         7|        6239.660833333332|      6633.077499999999|                0.0|           24247.09|        6873.4596896837675|               6257.946363636364|                              23737.09|                  0.21052631578947367|             8.88888888888889|         2.7142857142857144|                       0|
    | 2025-06-25|         0| NEW_JC_APP|       Transfer(C2C)|          6.2| 1000.0|                    BIO|         Active|       Level 1|      Customer Account|         16|          3|         0|       0|                1|              0|                  0|1.8245492931885376|          6|             5681.0| 946.8333333333334|           3000.0|              1.0|                       5|                     1|                  3|                      0|                       0|          0.05615208590036961|                0|                  1|                 1|           0|               0|                      0|                 0|                1|                0|                0|               0|                 6|             41989.6| 6998.266666666666|              5963.52|          23737.09|               0.0|                        5|                      1|                   3|                12|            74875.93| 6239.660833333332|              5963.52|          23737.09|               0.0|                        9|                      1|                   3|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|Utility Bills Pay...|          0.9289617486338798|                 1|                   2|                     5|                         7|        6239.660833333332|      6633.077499999999|                0.0|           24247.09|        6873.4596896837675|               6257.946363636364|                              23737.09|                  0.21052631578947367|             8.88888888888889|         2.7142857142857144|                       0|
    | 2025-06-25|         0| NEW_JC_APP|       Transfer(C2C)|      1798.51| 4000.0|                    BIO|         Active|       Level 1|      Customer Account|         18|          3|         0|       0|                0|              0|                  0| 7.494713823017072|          6|             5681.0| 946.8333333333334|           3000.0|              1.0|                       5|                     1|                  3|                      0|                       0|           3.2246083436014783|                0|                  1|                 1|           0|               0|                      0|                 0|                1|                0|                0|               0|                 6|             41989.6| 6998.266666666666|              5963.52|          23737.09|               0.0|                        5|                      1|                   3|                12|            74875.93| 6239.660833333332|              5963.52|          23737.09|               0.0|                        9|                      1|                   3|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|Utility Bills Pay...|          0.9289617486338798|                 1|                   2|                     5|                         7|        6239.660833333332|      6633.077499999999|                0.0|           24247.09|        6873.4596896837675|               6257.946363636364|                              23737.09|                  0.21052631578947367|             8.88888888888889|         2.7142857142857144|                       0|
    | 2025-06-25|         0| NEW_JC_APP|       Transfer(C2C)|       619.58| 1000.0|                    BIO|         Active|       Level 1|      Customer Account|         19|          3|         0|       0|                0|              0|                  0| 6.429041828880801|          6|             5681.0| 946.8333333333334|           3000.0|              1.0|                       5|                     1|                  3|                      0|                       0|          0.05615208590036961|                0|                  1|                 1|           0|               0|                      0|                 0|                1|                0|                0|               0|                 6|             41989.6| 6998.266666666666|              5963.52|          23737.09|               0.0|                        5|                      1|                   3|                12|            74875.93| 6239.660833333332|              5963.52|          23737.09|               0.0|                        9|                      1|                   3|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|Utility Bills Pay...|          0.9289617486338798|                 1|                   2|                     5|                         7|        6239.660833333332|      6633.077499999999|                0.0|           24247.09|        6873.4596896837675|               6257.946363636364|                              23737.09|                  0.21052631578947367|             8.88888888888889|         2.7142857142857144|                       0|
    | 2025-06-26|         0| NEW_JC_APP|Utility Bills Pay...|      5009.52| 4305.0|                    BIO|         Active|       Level 1|      Customer Account|          9|          4|         0|       0|                1|              0|                  0| 8.519095381208183|         12|            14565.0|           1213.75|           4000.0|              1.0|                       8|                     1|                  4|                      1|                       0|           2.5468589083419158|                1|                  0|                 1|           0|               0|                      0|                 0|                0|                0|                0|               0|                12|            42948.33|         3579.0275|              2113.52|           9188.52|               0.0|                        8|                      1|                   4|                15|            75254.99| 5016.999333333334|              2288.52|          23737.09|               0.0|                        9|                      1|                   4|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|       Transfer(C2B)|          0.9371980676328502|                 0|                   2|                     5|                        10|        5016.999333333334|               5242.466|                0.0|           24247.09|         6030.069779725017|               4811.619166666666|                              23737.09|                  0.21739130434782608|            6.909090909090909|         3.2857142857142856|                       0|
    | 2025-06-26|         0| NEW_JC_APP|       Transfer(C2B)|     40699.52| 2000.0|                    BIO|         Active|       Level 1|      Customer Account|         13|          4|         0|       0|                1|              0|                  0|10.613971578405973|         12|            14565.0|           1213.75|           4000.0|              1.0|                       8|                     1|                  4|                      1|                       0|           0.6477857878475798|                1|                  0|                 1|           0|               0|                      0|                 0|                0|                1|                0|               0|                12|            42948.33|         3579.0275|              2113.52|           9188.52|               0.0|                        8|                      1|                   4|                15|            75254.99| 5016.999333333334|              2288.52|          23737.09|               0.0|                        9|                      1|                   4|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|       Transfer(C2B)|          0.9371980676328502|                 0|                   2|                     5|                        10|        5016.999333333334|               5242.466|                0.0|           24247.09|         6030.069779725017|               4811.619166666666|                              23737.09|                  0.21739130434782608|            6.909090909090909|         3.2857142857142856|                       0|
    | 2025-06-26|         0| NEW_JC_APP|       Transfer(C2C)|      5098.51| 7000.0|                    BIO|         Active|       Level 1|      Customer Account|         18|          4|         0|       0|                0|              0|                  0|  8.53670361989121|         12|            14565.0|           1213.75|           4000.0|              1.0|                       8|                     1|                  4|                      1|                       0|             4.76725025746653|                1|                  0|                 1|           0|               0|                      0|                 0|                1|                0|                0|               0|                12|            42948.33|         3579.0275|              2113.52|           9188.52|               0.0|                        8|                      1|                   4|                15|            75254.99| 5016.999333333334|              2288.52|          23737.09|               0.0|                        9|                      1|                   4|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|       Transfer(C2B)|          0.9371980676328502|                 0|                   2|                     5|                        10|        5016.999333333334|               5242.466|                0.0|           24247.09|         6030.069779725017|               4811.619166666666|                              23737.09|                  0.21739130434782608|            6.909090909090909|         3.2857142857142856|                       0|
    | 2025-06-26|         0| NEW_JC_APP|       Transfer(C2B)|     34999.52| 2000.0|                    BIO|         Active|       Level 1|      Customer Account|         20|          4|         0|       0|                0|              0|                  0|10.463089626379434|         12|            14565.0|           1213.75|           4000.0|              1.0|                       8|                     1|                  4|                      1|                       0|           0.6477857878475798|                1|                  0|                 1|           0|               0|                      0|                 0|                0|                1|                0|               0|                12|            42948.33|         3579.0275|              2113.52|           9188.52|               0.0|                        8|                      1|                   4|                15|            75254.99| 5016.999333333334|              2288.52|          23737.09|               0.0|                        9|                      1|                   4|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|       Transfer(C2B)|          0.9371980676328502|                 0|                   2|                     5|                        10|        5016.999333333334|               5242.466|                0.0|           24247.09|         6030.069779725017|               4811.619166666666|                              23737.09|                  0.21739130434782608|            6.909090909090909|         3.2857142857142856|                       0|
    | 2025-06-26|         0| NEW_JC_APP|       Transfer(C2B)|     32999.52| 6000.0|                    BIO|         Active|       Level 1|      Customer Account|         20|          4|         0|       0|                0|              0|                  0|10.404248294895082|         12|            14565.0|           1213.75|           4000.0|              1.0|                       8|                     1|                  4|                      1|                       0|           3.9433573635427392|                1|                  0|                 1|           0|               0|                      0|                 0|                0|                1|                0|               0|                12|            42948.33|         3579.0275|              2113.52|           9188.52|               0.0|                        8|                      1|                   4|                15|            75254.99| 5016.999333333334|              2288.52|          23737.09|               0.0|                        9|                      1|                   4|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|       Transfer(C2B)|          0.9371980676328502|                 0|                   2|                     5|                        10|        5016.999333333334|               5242.466|                0.0|           24247.09|         6030.069779725017|               4811.619166666666|                              23737.09|                  0.21739130434782608|            6.909090909090909|         3.2857142857142856|                       0|
    | 2025-06-27|         0| NEW_JC_APP|       Transfer(C2C)|     22723.16| 5000.0|                    BIO|         Active|       Level 1|      Customer Account|          7|          5|         0|       0|                0|              0|                  0|10.031139948418641|         19|            39030.0|2054.2105263157896|           7000.0|              1.0|                       9|                     1|                  4|                      1|                       0|           1.4340251088905969|                2|                  0|                 1|           0|               0|                      0|                 0|                1|                0|                0|               0|                19|            175850.2| 9255.273684210528|              5009.52|          40699.52|               0.0|                        9|                      1|                   4|                22|           214440.38|           9747.29|              5493.51|          40699.52|               0.0|                        9|                      1|                   4|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|       Transfer(C2C)|                        0.95|                 1|                   2|                     7|                        15|                  9747.29|                9713.29|                0.0|           40699.52|        12172.897168965159|              10364.464210526316|                              40699.52|                  0.23333333333333334|            4.896551724137931|          4.285714285714286|                       0|
    | 2025-06-27|         0| NEW_JC_APP|       Transfer(C2B)|     25277.52|  500.0|                    BIO|         Active|       Level 1|      Customer Account|          8|          5|         0|       0|                0|              0|                  0|10.137670743380466|         19|            39030.0|2054.2105263157896|           7000.0|              1.0|                       9|                     1|                  4|                      1|                       0|          -0.7565974891109403|                2|                  0|                 1|           0|               0|                      0|                 0|                0|                1|                0|               0|                19|            175850.2| 9255.273684210528|              5009.52|          40699.52|               0.0|                        9|                      1|                   4|                22|           214440.38|           9747.29|              5493.51|          40699.52|               0.0|                        9|                      1|                   4|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|       Transfer(C2C)|                        0.95|                 1|                   2|                     7|                        15|                  9747.29|                9713.29|                0.0|           40699.52|        12172.897168965159|              10364.464210526316|                              40699.52|                  0.23333333333333334|            4.896551724137931|          4.285714285714286|                       0|
    | 2025-06-27|         0| NEW_JC_APP|Jazz Load (Prepai...|     24777.52|  300.0|                    BIO|         Active|       Level 1|      Customer Account|          9|          5|         0|       0|                1|              0|                  0|10.117692070484242|         19|            39030.0|2054.2105263157896|           7000.0|              1.0|                       9|                     1|                  4|                      1|                       0|          -0.8539584934665642|                2|                  0|                 1|           0|               0|                      0|                 0|                0|                0|                0|               1|                19|            175850.2| 9255.273684210528|              5009.52|          40699.52|               0.0|                        9|                      1|                   4|                22|           214440.38|           9747.29|              5493.51|          40699.52|               0.0|                        9|                      1|                   4|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|       Transfer(C2C)|                        0.95|                 1|                   2|                     7|                        15|                  9747.29|                9713.29|                0.0|           40699.52|        12172.897168965159|              10364.464210526316|                              40699.52|                  0.23333333333333334|            4.896551724137931|          4.285714285714286|                       0|
    | 2025-06-27|         0| NEW_JC_APP|       Transfer(C2C)|          6.2| 2000.0|                    BIO|         Active|       Level 1|      Customer Account|         16|          5|         0|       0|                1|              0|                  0|1.8245492931885376|         19|            39030.0|2054.2105263157896|           7000.0|              1.0|                       9|                     1|                  4|                      1|                       0|         -0.02638995644376...|                2|                  0|                 1|           0|               0|                      0|                 0|                1|                0|                0|               0|                19|            175850.2| 9255.273684210528|              5009.52|          40699.52|               0.0|                        9|                      1|                   4|                22|           214440.38|           9747.29|              5493.51|          40699.52|               0.0|                        9|                      1|                   4|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|       Transfer(C2C)|                        0.95|                 1|                   2|                     7|                        15|                  9747.29|                9713.29|                0.0|           40699.52|        12172.897168965159|              10364.464210526316|                              40699.52|                  0.23333333333333334|            4.896551724137931|          4.285714285714286|                       0|
    | 2025-06-29|         0| NEW_JC_APP|       Transfer(C2C)|      3080.14| 2000.0|                    BIO|         Active|       Level 1|      Customer Account|          7|          7|         1|       0|                0|              0|                  0| 8.032730330307384|         19|            65802.0|3463.2631578947367|          10200.0|            300.0|                      11|                     1|                  5|                      1|                       0|          -0.4225099541047384|                1|                  3|                 1|           0|               0|                      0|                 0|                1|                0|                0|               0|                19|           330095.01|17373.421578947367|             20775.97|          40699.52|               6.2|                       11|                      1|                   5|                31|           373043.34|12033.656129032259|              6038.52|          40699.52|               0.0|                       12|                      1|                   5|               NEW_JC_APP|            NEW_JC_APP|                            0.0|         Transfer(C2C)|       Transfer(C2C)|          0.9594017094017094|                 1|                   3|                    13|                        18|       12033.656129032259|     11401.993225806451|                0.0|           40699.52|        12597.722370692887|               12250.00962962963|                              40699.52|                                 0.25|            4.428571428571429|          5.142857142857143|                       0|
    +-----------+----------+-----------+--------------------+-------------+-------+-----------------------+---------------+--------------+----------------------+-----------+-----------+----------+--------+-----------------+---------------+-------------------+------------------+-----------+-------------------+------------------+-----------------+-----------------+------------------------+----------------------+-------------------+-----------------------+------------------------+-----------------------------+-----------------+-------------------+------------------+------------+----------------+-----------------------+------------------+-----------------+-----------------+-----------------+----------------+------------------+--------------------+------------------+---------------------+------------------+------------------+-------------------------+-----------------------+--------------------+------------------+--------------------+------------------+---------------------+------------------+------------------+-------------------------+-----------------------+--------------------+-------------------------+----------------------+-------------------------------+----------------------+--------------------+----------------------------+------------------+--------------------+----------------------+--------------------------+-------------------------+-----------------------+-------------------+-------------------+--------------------------+--------------------------------+--------------------------------------+-------------------------------------+-----------------------------+---------------------------+------------------------+
    only showing top 20 rows
    


# Data Preview
Display sample rows and basic statistics to verify data loaded correctly.


```python
df.count()
```

                                                                                    




    180883125




```python
df.printSchema()
```

    root
     |-- cutoff_date: date (nullable = false)
     |-- fraud_flag: short (nullable = false)
     |-- trx_channel: string (nullable = false)
     |-- trx_type: string (nullable = false)
     |-- start_balance: double (nullable = false)
     |-- trx_amt: double (nullable = false)
     |-- mbar_registered_channel: string (nullable = false)
     |-- mbar_a_c_status: string (nullable = false)
     |-- mbar_a_c_level: string (nullable = false)
     |-- mbar_account_type_name: string (nullable = false)
     |-- hour_of_day: short (nullable = false)
     |-- day_of_week: short (nullable = false)
     |-- is_weekend: short (nullable = false)
     |-- is_night: short (nullable = false)
     |-- is_business_hours: short (nullable = false)
     |-- is_unusual_hour: short (nullable = false)
     |-- night_weekend_combo: short (nullable = false)
     |-- start_balance_log: double (nullable = false)
     |-- txn_txns_3d: long (nullable = false)
     |-- txn_total_amount_3d: double (nullable = false)
     |-- txn_avg_amount_3d: double (nullable = false)
     |-- txn_max_amount_3d: double (nullable = false)
     |-- txn_min_amount_3d: double (nullable = false)
     |-- txn_unique_recipients_3d: long (nullable = false)
     |-- txn_unique_channels_3d: long (nullable = false)
     |-- txn_unique_types_3d: long (nullable = false)
     |-- txn_is_high_activity_3d: short (nullable = false)
     |-- txn_multi_channel_recent: short (nullable = false)
     |-- txn_amount_deviation_from_avg: double (nullable = false)
     |-- txn_night_txns_3d: long (nullable = false)
     |-- txn_weekend_txns_3d: long (nullable = false)
     |-- channel_new_jc_app: short (nullable = false)
     |-- channel_ussd: short (nullable = false)
     |-- channel_ussd_api: short (nullable = false)
     |-- channel_payment_gateway: short (nullable = false)
     |-- channel_mobile_app: short (nullable = false)
     |-- type_transfer_c2c: short (nullable = false)
     |-- type_transfer_c2b: short (nullable = false)
     |-- type_bill_payment: short (nullable = false)
     |-- type_mobile_load: short (nullable = false)
     |-- user_total_txns_3d: long (nullable = false)
     |-- user_total_amount_3d: double (nullable = false)
     |-- user_avg_amount_3d: double (nullable = false)
     |-- user_median_amount_3d: double (nullable = false)
     |-- user_max_amount_3d: double (nullable = false)
     |-- user_min_amount_3d: double (nullable = false)
     |-- user_unique_recipients_3d: long (nullable = false)
     |-- user_unique_channels_3d: long (nullable = false)
     |-- user_unique_types_3d: long (nullable = false)
     |-- user_total_txns_7d: long (nullable = false)
     |-- user_total_amount_7d: double (nullable = false)
     |-- user_avg_amount_7d: double (nullable = false)
     |-- user_median_amount_7d: double (nullable = false)
     |-- user_max_amount_7d: double (nullable = false)
     |-- user_min_amount_7d: double (nullable = false)
     |-- user_unique_recipients_7d: long (nullable = false)
     |-- user_unique_channels_7d: long (nullable = false)
     |-- user_unique_types_7d: long (nullable = false)
     |-- user_most_used_channel_7d: string (nullable = false)
     |-- user_last_used_channel: string (nullable = false)
     |-- user_channel_diversity_score_7d: double (nullable = false)
     |-- user_most_used_type_7d: string (nullable = false)
     |-- user_last_used_type: string (nullable = false)
     |-- user_type_diversity_score_7d: double (nullable = false)
     |-- user_night_txns_7d: long (nullable = false)
     |-- user_weekend_txns_7d: long (nullable = false)
     |-- user_peak_hour_txns_7d: long (nullable = false)
     |-- user_off_peak_hour_txns_7d: long (nullable = false)
     |-- user_avg_start_balance_7d: double (nullable = false)
     |-- user_avg_end_balance_7d: double (nullable = false)
     |-- user_min_balance_7d: double (nullable = false)
     |-- user_max_balance_7d: double (nullable = false)
     |-- user_balance_volatility_7d: double (nullable = false)
     |-- user_avg_amount_per_recipient_7d: double (nullable = false)
     |-- user_max_amount_to_single_recipient_7d: double (nullable = false)
     |-- user_recipient_concentration_ratio_7d: double (nullable = false)
     |-- user_avg_time_between_txns_7d: double (nullable = false)
     |-- user_txn_frequency_score_7d: double (nullable = false)
     |-- user_days_since_last_txn: long (nullable = false)
    



```python
all_columns = df.columns
features = [c for c in all_columns if c not in ['cutoff_date','fraud_flag']]
numerical_cols = [
 'start_balance',
 'trx_amt',
 'hour_of_day',
 'day_of_week',
 'is_weekend',
 'is_night',
 'is_business_hours',
 'is_unusual_hour',
 'night_weekend_combo',
 'start_balance_log',
 'txn_txns_3d',
 'txn_total_amount_3d',
 'txn_avg_amount_3d',
 'txn_max_amount_3d',
 'txn_min_amount_3d',
 'txn_unique_recipients_3d',
 'txn_unique_channels_3d',
 'txn_unique_types_3d',
 'txn_is_high_activity_3d',
 'txn_multi_channel_recent',
 'txn_amount_deviation_from_avg',
 'txn_night_txns_3d',
 'txn_weekend_txns_3d',
 'user_total_txns_3d',
 'user_total_amount_3d',
 'user_avg_amount_3d',
 'user_median_amount_3d',
 'user_max_amount_3d',
 'user_min_amount_3d',
 'user_unique_recipients_3d',
 'user_unique_channels_3d',
 'user_unique_types_3d',
 'user_total_txns_7d',
 'user_total_amount_7d',
 'user_avg_amount_7d',
 'user_median_amount_7d',
 'user_max_amount_7d',
 'user_min_amount_7d',
 'user_unique_recipients_7d',
 'user_unique_channels_7d',
 'user_unique_types_7d',
 'user_most_used_channel_7d',
 'user_last_used_channel',
 'user_channel_diversity_score_7d',
 'user_most_used_type_7d',
 'user_last_used_type',
 'user_type_diversity_score_7d',
 'user_night_txns_7d',
 'user_weekend_txns_7d',
 'user_peak_hour_txns_7d',
 'user_off_peak_hour_txns_7d',
 'user_avg_start_balance_7d',
 'user_avg_end_balance_7d',
 'user_min_balance_7d',
 'user_max_balance_7d',
 'user_balance_volatility_7d',
 'user_avg_amount_per_recipient_7d',
 'user_max_amount_to_single_recipient_7d',
 'user_recipient_concentration_ratio_7d',
 'user_avg_time_between_txns_7d',
 'user_txn_frequency_score_7d',
 'user_days_since_last_txn']
feature_name_mapping = {
    'trx_amt': 'Transaction Amount (PKR)',
    'start_balance': 'Account Balance Before Transaction',
    'start_balance_log': 'Log(Account Balance)',
    'hour_of_day': 'Hour of Transaction (0-23)',
    'day_of_week': 'Day of Week (1=Mon, 7=Sun)',
    'is_weekend': 'Weekend Transaction Flag',
    'is_night': 'Night Hours Transaction Flag',
    'is_business_hours': 'Business Hours Transaction Flag',
    'is_unusual_hour': 'Unusual Time Transaction Flag',
    'night_weekend_combo': 'Night+Weekend Transaction Flag',
    'txn_txns_3d': 'Transaction Count (Last 3 Days)',
    'txn_total_amount_3d': 'Total Amount Transacted (3 Days)',
    'txn_avg_amount_3d': 'Average Transaction Amount (3 Days)',
    'txn_max_amount_3d': 'Maximum Transaction Amount (3 Days)',
    'txn_min_amount_3d': 'Minimum Transaction Amount (3 Days)',
    'txn_unique_recipients_3d': 'Unique Recipients Count (3 Days)',
    'txn_unique_channels_3d': 'Unique Channels Used (3 Days)',
    'txn_unique_types_3d': 'Unique Transaction Types (3 Days)',
    'txn_night_txns_3d': 'Night Transactions Count (3 Days)',
    'txn_weekend_txns_3d': 'Weekend Transactions Count (3 Days)',
    'txn_is_high_activity_3d': 'High Activity Period Flag',
    'txn_multi_channel_recent': 'Multiple Channels Used Recently Flag',
    'txn_amount_deviation_from_avg': 'Amount Deviation from User Average',
    'channel_new_jc_app': 'Uses New JazzCash App',
    'channel_ussd': 'Uses USSD Channel',
    'channel_ussd_api': 'Uses USSD API Channel',
    'channel_payment_gateway': 'Uses Payment Gateway',
    'channel_mobile_app': 'Uses Mobile App',
    'type_transfer_c2c': 'Customer-to-Customer Transfer',
    'type_transfer_c2b': 'Customer-to-Business Transfer',
    'type_bill_payment': 'Bill Payment Transaction',
    'type_mobile_load': 'Mobile Top-up Transaction',
    'user_total_txns_3d': 'User Total Transactions (3 Days)',
    'user_total_amount_3d': 'User Total Amount (3 Days)',
    'user_avg_amount_3d': 'User Average Amount (3 Days)',
    'user_median_amount_3d': 'User Median Amount (3 Days)',
    'user_max_amount_3d': 'User Maximum Amount (3 Days)',
    'user_min_amount_3d': 'User Minimum Amount (3 Days)',
    'user_unique_recipients_3d': 'User Unique Recipients (3 Days)',
    'user_unique_channels_3d': 'User Unique Channels (3 Days)',
    'user_unique_types_3d': 'User Unique Types (3 Days)',
    'user_total_txns_7d': 'User Total Transactions (7 Days)',
    'user_total_amount_7d': 'User Total Amount (7 Days)',
    'user_avg_amount_7d': 'User Average Amount (7 Days)',
    'user_median_amount_7d': 'User Median Amount (7 Days)',
    'user_max_amount_7d': 'User Maximum Amount (7 Days)',
    'user_min_amount_7d': 'User Minimum Amount (7 Days)',
    'user_unique_recipients_7d': 'User Unique Recipients (7 Days)',
    'user_unique_channels_7d': 'User Unique Channels (7 Days)',
    'user_unique_types_7d': 'User Unique Types (7 Days)',
    'user_channel_diversity_score_7d': 'User Channel Diversity Score',
    'user_type_diversity_score_7d': 'User Transaction Type Diversity',
    'user_night_txns_7d': 'User Night Transactions (7 Days)',
    'user_weekend_txns_7d': 'User Weekend Transactions (7 Days)',
    'user_peak_hour_txns_7d': 'User Peak Hour Transactions (7 Days)',
    'user_off_peak_hour_txns_7d': 'User Off-Peak Transactions (7 Days)',
    'user_avg_start_balance_7d': 'User Average Starting Balance',
    'user_avg_end_balance_7d': 'User Average Ending Balance',
    'user_min_balance_7d': 'User Minimum Balance (7 Days)',
    'user_max_balance_7d': 'User Maximum Balance (7 Days)',
    'user_balance_volatility_7d': 'User Balance Volatility',
    'user_avg_amount_per_recipient_7d': 'User Avg Amount per Recipient',
    'user_max_amount_to_single_recipient_7d': 'User Max Amount to One Recipient',
    'user_recipient_concentration_ratio_7d': 'User Recipient Concentration',
    'user_avg_time_between_txns_7d': 'User Avg Time Between Transactions',
    'user_txn_frequency_score_7d': 'User Transaction Frequency Score',
    'user_days_since_last_txn': 'Days Since User Last Transaction',
    'fraud_flag': 'Fraud Flag (0=Safe, 1=Fraud)'
}

```

# Feature Definitions & Mappings
Define numerical feature columns and create human-readable feature name mappings for visualization and reporting.

# DATA SAMPLING
Handle extreme class imbalance by downsampling the majority class (non-fraud) to create a balanced training dataset.


```python
df.groupBy("fraud_flag").count().show()
```

    [Stage 2:=================================================>       (14 + 2) / 16]

    +----------+---------+
    |fraud_flag|    count|
    +----------+---------+
    |         1|     6847|
    |         0|180876278|
    +----------+---------+
    


                                                                                    

# Class Distribution Analysis
Analyze the fraud vs non-fraud distribution to understand the severe class imbalance (~0.004% fraud rate).


```python
fraud_count=6847
non_fraud_count=180876278
fraud_percentage = (fraud_count / (fraud_count + non_fraud_count)) * 100
print(f"Fraud Records: {fraud_count:,} ({fraud_percentage:.4f}%)")
print(f"Non-Fraud Records: {non_fraud_count:,} ({100 - fraud_percentage:.4f}%)")
print(f"Imbalance Ratio: 1:{non_fraud_count//fraud_count}")
```

    Fraud Records: 6,847 (0.0038%)
    Non-Fraud Records: 180,876,278 (99.9962%)
    Imbalance Ratio: 1:26416



```python
downsample_ratio = 10000  # Adjust this based on your preference
target_non_fraud_count = fraud_count * downsample_ratio
print(f"Downsampling Ratio: 1:{downsample_ratio} (fraud:non-fraud)")
print(f"Target Non-Fraud Count: {target_non_fraud_count:,}")
print(f"Sampling Fraction: {target_non_fraud_count / non_fraud_count:.6f}")
```

    Downsampling Ratio: 1:10000 (fraud:non-fraud)
    Target Non-Fraud Count: 68,470,000
    Sampling Fraction: 0.378546


# Downsampling Strategy
Calculate sampling parameters to achieve target fraud:non-fraud ratio of 1:10000.


```python
df_fraud = df.filter("fraud_flag = 1")
df_non_fraud = df.filter("fraud_flag = 0")
print(f"Fraud DataFrame Count: {df_fraud.count():,}")
print(f"Non-Fraud DataFrame Count: {df_non_fraud.count():,}")
```

    Fraud DataFrame Count: 6,847


    [Stage 8:=================================================>       (14 + 2) / 16]

    Non-Fraud DataFrame Count: 180,876,278


                                                                                    

# Separate Fraud and Non-Fraud Data
Split the dataset into fraud and non-fraud subsets for stratified sampling.


```python
sampling_fraction = target_non_fraud_count / non_fraud_count
df_non_fraud_sampled = df_non_fraud.sample(
    withReplacement=False, 
    fraction=sampling_fraction,
    seed=42  
)

actual_sampled_count = df_non_fraud_sampled.count()
print(f"Sampling Fraction: {sampling_fraction:.6f}")
print(f"Expected Sampled Count: ~{target_non_fraud_count:,}")
print(f"Actual Sampled Count: {actual_sampled_count:,}")
```

    [Stage 11:=================================================>      (14 + 2) / 16]

    Sampling Fraction: 0.378546
    Expected Sampled Count: ~68,470,000
    Actual Sampled Count: 68,471,379


                                                                                    

# Apply Downsampling
Randomly sample non-fraud records using the calculated fraction to reduce class imbalance.


```python
df_balanced_cached = df_fraud.union(df_non_fraud_sampled)
balanced_total = df_balanced_cached.count()
balanced_fraud = df_balanced_cached.filter("fraud_flag = 1").count()
balanced_non_fraud = df_balanced_cached.filter("fraud_flag = 0").count()
```

                                                                                    

# Create Balanced Dataset
Combine all fraud cases with downsampled non-fraud cases to create the balanced training dataset.


```python
feature_variance_results = pd.read_csv("/root/research-dir/dev/jazzcash-fraud-detection/analysis/feature_variance_results.csv")
display(feature_variance_results.head(10))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>variance</th>
      <th>std_dev</th>
      <th>count</th>
      <th>null_count</th>
      <th>completeness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>user_total_amount_7d</td>
      <td>2.922059e+17</td>
      <td>5.405607e+08</td>
      <td>68474149</td>
      <td>0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>user_total_amount_3d</td>
      <td>1.261987e+17</td>
      <td>3.552446e+08</td>
      <td>68474149</td>
      <td>0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>user_max_amount_to_single_recipient_7d</td>
      <td>5.829966e+16</td>
      <td>2.414532e+08</td>
      <td>68474149</td>
      <td>0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>user_max_balance_7d</td>
      <td>5.829921e+16</td>
      <td>2.414523e+08</td>
      <td>68474149</td>
      <td>0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>user_max_amount_7d</td>
      <td>5.829904e+16</td>
      <td>2.414519e+08</td>
      <td>68474149</td>
      <td>0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>user_max_amount_3d</td>
      <td>3.423436e+16</td>
      <td>1.850253e+08</td>
      <td>68474149</td>
      <td>0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>start_balance</td>
      <td>2.554187e+16</td>
      <td>1.598182e+08</td>
      <td>68474149</td>
      <td>0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>user_median_amount_7d</td>
      <td>1.247727e+16</td>
      <td>1.117017e+08</td>
      <td>68474149</td>
      <td>0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>user_median_amount_3d</td>
      <td>1.124278e+16</td>
      <td>1.060320e+08</td>
      <td>68474149</td>
      <td>0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>user_min_balance_7d</td>
      <td>8.288571e+15</td>
      <td>9.104159e+07</td>
      <td>68474149</td>
      <td>0</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
</div>


# Feature Selection (Variance-Based)
Load pre-computed feature variance analysis to select top 30 most informative features for modeling.


```python
top_30_features_by_variance = feature_variance_results[
    feature_variance_results['feature'] != 'fraud_flag'
]['feature'].tolist()

print(f"   Selected top 30 features based on variance:")
print(f"   Total features available: {len(feature_variance_results)}")
print(f"   Selected features: {len(top_30_features_by_variance)}")
```

       Selected top 30 features based on variance:
       Total features available: 67
       Selected features: 67



```python
model_features = top_30_features_by_variance + ['fraud_flag']
```


```python
top_30_features_by_variance = feature_variance_results[
    feature_variance_results['feature'] != 'fraud_flag'
]['feature'].tolist()

print(f"  Selected top 30 features based on variance:")
print(f"   Total features available: {len(feature_variance_results)}")
print(f"   Selected features: {len(top_30_features_by_variance)}")
```

    ✅ Selected top 30 features based on variance:
       Total features available: 67
       Selected features: 67



```python
total_records = df_balanced_cached.count()
fraud_records = df_balanced_cached.filter(col('fraud_flag') == 1).count()
non_fraud_records = df_balanced_cached.filter(col('fraud_flag') == 0).count()

print(f" Original dataset composition:")
print(f"   Total records: {total_records}")
print(f"   Fraud cases: {fraud_records}")
print(f"   Non-fraud cases: {non_fraud_records}")
print(f"   Fraud rate: {fraud_records/total_records} ({fraud_records/total_records*100}%)")
```

    [Stage 29:====================================================>   (30 + 2) / 32]

    📊 Original dataset composition:
       Total records: 68478226
       Fraud cases: 6847
       Non-fraud cases: 68471379
       Fraud rate: 9.998798742245454e-05 (0.009998798742245454%)


                                                                                    

# Stratified Sampling for Modeling
Apply stratified sampling to create a manageable dataset (max 1M records) while preserving all fraud cases.


```python
modeling_sample_size = min(1000000, total_records)
target_non_fraud_count = modeling_sample_size - fraud_records
non_fraud_fraction = min(1.0, target_non_fraud_count / non_fraud_records)
```


```python
print(f"   Target sample size: {modeling_sample_size:,}")
print(f"   Keep all fraud cases: {fraud_records:,}")
print(f"   Target non-fraud cases: {target_non_fraud_count:,}")
print(f"   Non-fraud sampling fraction: {non_fraud_fraction:.6f}")
```

       Target sample size: 1,000,000
       Keep all fraud cases: 6,847
       Target non-fraud cases: 993,153
       Non-fraud sampling fraction: 0.014505



```python
modeling_fraction = modeling_sample_size / total_records
```


```python
df_fraud = df_balanced_cached.filter(col('fraud_flag') == 1).select(model_features)
df_non_fraud = df_balanced_cached.filter(col('fraud_flag') == 0).select(model_features).sample(
    withReplacement=False,
    fraction=non_fraud_fraction,
    seed=42
)
df_modeling = df_fraud.union(df_non_fraud).toPandas()
print(f" Stratified sampling completed:")
print(f"   Fraud records: {df_fraud.count():,} (100% kept)")
print(f"   Non-fraud records: {df_non_fraud.count():,} ({non_fraud_fraction:.4f} fraction)")
print(f"   Total modeling sample: {len(df_modeling):,}")
```

                                                                                    

    ✅ Stratified sampling completed:


                                                                                    

       Fraud records: 6,847 (100% kept)


    [Stage 36:====================================================>   (30 + 2) / 32]

       Non-fraud records: 993,137 (0.0145 fraction)
       Total modeling sample: 999,984


                                                                                    


```python
df_modeling = df_modeling.fillna(df_modeling.median())
```

# Handle Missing Values
Fill missing values with column medians to ensure complete data for model training.


```python
X = df_modeling[top_30_features_by_variance]
y = df_modeling['fraud_flag']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = SKStandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

# Train/Test Split & Feature Scaling
Split data into 80% train / 20% test with stratification, then apply StandardScaler normalization.


```python
lr_model = SKLogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'  
)

lr_model.fit(X_train_scaled, y_train)
y_pred = lr_model.predict(X_test_scaled)
y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

auc_score = roc_auc_score(y_test, y_pred_proba)
classification_rep = classification_report(y_test, y_pred, output_dict=True)

print(f"   AUC-ROC Score: {auc_score:.4f}")
print(f"   Accuracy: {classification_rep['accuracy']:.4f}")
print(f"   Precision (Fraud): {classification_rep['1']['precision']:.4f}")
print(f"   Recall (Fraud): {classification_rep['1']['recall']:.4f}")
print(f"   F1-Score (Fraud): {classification_rep['1']['f1-score']:.4f}")
print(f"   Precision (Legitimate): {classification_rep['0']['precision']:.4f}")
print(f"   Recall (Legitimate): {classification_rep['0']['recall']:.4f}")
print(f"   F1-Score (Legitimate): {classification_rep['0']['f1-score']:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\n  CONFUSION MATRIX:")
print(f"                 Predicted")
print(f"                 Legit  Fraud")
print(f"Actual Legit     {cm[0,0]:>5}  {cm[0,1]:>5}")
print(f"       Fraud     {cm[1,0]:>5}  {cm[1,1]:>5}")
```

       AUC-ROC Score: 0.8814
       Accuracy: 0.8111
       Precision (Fraud): 0.0287
       Recall (Fraud): 0.8086
       F1-Score (Fraud): 0.0554
       Precision (Legitimate): 0.9984
       Recall (Legitimate): 0.8111
       F1-Score (Legitimate): 0.8950
    
      CONFUSION MATRIX:
                     Predicted
                     Legit  Fraud
    Actual Legit     161104  37524
           Fraud       262   1107


# Logistic Regression Model Training & Evaluation
Train a balanced Logistic Regression model and evaluate with AUC-ROC, precision, recall, F1-score, and confusion matrix.


```python
tn, fp, fn, tp = cm.ravel()
false_positive_rate = fp / (fp + tn)
false_negative_rate = fn / (fn + tp)

print(f"   False Positive Rate: {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
print(f"   False Negative Rate: {false_negative_rate:.4f} ({false_negative_rate*100:.2f}%)")
print(f"   True Positives (Frauds Caught): {tp:,}")
print(f"   False Negatives (Frauds Missed): {fn:,}")
print(f"   False Positives (Legitimate Flagged): {fp:,}")
```

       False Positive Rate: 0.1889 (18.89%)
       False Negative Rate: 0.1914 (19.14%)
       True Positives (Frauds Caught): 1,107
       False Negatives (Frauds Missed): 262
       False Positives (Legitimate Flagged): 37,524


# Error Rate Analysis
Calculate false positive and false negative rates to understand model's fraud detection and false alarm performance.


```python
feature_importance = pd.DataFrame({
    'Feature': top_30_features_by_variance,
    'Feature_Name': [feature_name_mapping.get(f, f) for f in top_30_features_by_variance],
    'Coefficient': lr_model.coef_[0],
    'Abs_Coefficient': np.abs(lr_model.coef_[0])
})

feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
```

# Feature Importance (Coefficients)
Extract and rank feature importance based on logistic regression coefficients with human-readable names.


```python
feature_importance
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Feature_Name</th>
      <th>Coefficient</th>
      <th>Abs_Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>user_peak_hour_txns_7d</td>
      <td>User Peak Hour Transactions (7 Days)</td>
      <td>-1.685300</td>
      <td>1.685300</td>
    </tr>
    <tr>
      <th>18</th>
      <td>user_total_txns_7d</td>
      <td>User Total Transactions (7 Days)</td>
      <td>-1.362529</td>
      <td>1.362529</td>
    </tr>
    <tr>
      <th>21</th>
      <td>txn_txns_3d</td>
      <td>Transaction Count (Last 3 Days)</td>
      <td>1.038702</td>
      <td>1.038702</td>
    </tr>
    <tr>
      <th>22</th>
      <td>user_total_txns_3d</td>
      <td>User Total Transactions (3 Days)</td>
      <td>1.038702</td>
      <td>1.038702</td>
    </tr>
    <tr>
      <th>52</th>
      <td>type_mobile_load</td>
      <td>Mobile Top-up Transaction</td>
      <td>-0.856027</td>
      <td>0.856027</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>user_max_amount_to_single_recipient_7d</td>
      <td>User Max Amount to One Recipient</td>
      <td>-0.014065</td>
      <td>0.014065</td>
    </tr>
    <tr>
      <th>4</th>
      <td>user_max_amount_7d</td>
      <td>User Maximum Amount (7 Days)</td>
      <td>-0.013607</td>
      <td>0.013607</td>
    </tr>
    <tr>
      <th>54</th>
      <td>user_recipient_concentration_ratio_7d</td>
      <td>User Recipient Concentration</td>
      <td>-0.004643</td>
      <td>0.004643</td>
    </tr>
    <tr>
      <th>59</th>
      <td>type_bill_payment</td>
      <td>Bill Payment Transaction</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>60</th>
      <td>user_days_since_last_txn</td>
      <td>Days Since User Last Transaction</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>67 rows × 4 columns</p>
</div>




```python
shap_sample_size = min(1000, len(X_test))
X_shap = X_test_scaled[:shap_sample_size]
y_shap = y_test.iloc[:shap_sample_size].values
print(f" Using {shap_sample_size} samples for SHAP analysis")
```

     Using 1000 samples for SHAP analysis


# SHAP Analysis
Use SHAP (SHapley Additive exPlanations) to provide interpretable explanations of feature contributions to predictions.


```python
explainer = shap.LinearExplainer(lr_model, X_train_scaled)
shap_values = explainer.shap_values(X_shap)
plt.style.use('default')
print(f" Creating SHAP Feature Importance Plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_shap, feature_names=top_30_features_by_variance, 
                  show=False, plot_type="bar")
plt.title(' SHAP Feature Importance\n(Mean Absolute SHAP Values)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

     Creating SHAP Feature Importance Plot...



    
![png](fraud_files/fraud_51_1.png)
    


# SHAP Feature Importance Plot
Bar chart showing mean absolute SHAP values - which features have the largest overall impact on predictions.


```python
print(f" Creating SHAP Summary Plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_shap, feature_names=top_30_features_by_variance, show=False)
plt.title('🎯 SHAP Summary Plot\n(Feature Impact on Predictions)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

     Creating SHAP Summary Plot...


    /tmp/ipykernel_3650597/168295226.py:5: UserWarning: Glyph 127919 (\N{DIRECT HIT}) missing from font(s) DejaVu Sans.
      plt.tight_layout()
    /root/miniconda3/envs/fraud-spark/lib/python3.10/site-packages/IPython/core/pylabtools.py:170: UserWarning: Glyph 127919 (\N{DIRECT HIT}) missing from font(s) DejaVu Sans.
      fig.canvas.print_figure(bytes_io, **kw)



    
![png](fraud_files/fraud_53_2.png)
    


# SHAP Summary Plot
Beeswarm plot showing how each feature's value (high/low) pushes predictions toward fraud or legitimate classification.

# SHAP Case Comparison
Side-by-side comparison of SHAP values for individual fraud vs legitimate transactions, showing top 15 features driving each prediction.


```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
fraud_indices = np.where(y_shap == 1)[0]
if len(fraud_indices) > 0:
    fraud_idx = fraud_indices[0]
    base_value = explainer.expected_value
    shap_vals = shap_values[fraud_idx]
    sorted_indices = np.argsort(np.abs(shap_vals))[-15:]
    sorted_shap = shap_vals[sorted_indices]
    sorted_features = [top_30_features_by_variance[i] for i in sorted_indices]
    sorted_meaningful_names = [feature_name_mapping.get(f, f)[:25] for f in sorted_features]
    colors = ['red' if val > 0 else 'blue' for val in sorted_shap]
    ax1.barh(range(len(sorted_shap)), sorted_shap, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(sorted_shap)))
    ax1.set_yticklabels(sorted_meaningful_names, fontsize=10)
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax1.set_xlabel('SHAP Value (Impact on Prediction)')
    ax1.set_title(f' SHAP Values for Fraud Case\n(Top 15 Contributing Features)', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
else:
    ax1.text(0.5, 0.5, 'No fraud cases in sample', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title(' SHAP Values for Fraud Case\n(No Fraud Cases Available)', fontsize=14, fontweight='bold')

legit_indices = np.where(y_shap == 0)[0]
if len(legit_indices) > 0:
    legit_idx = legit_indices[0]
    
    base_value = explainer.expected_value
    shap_vals = shap_values[legit_idx]
    feature_vals = X_shap[legit_idx]
    
    sorted_indices = np.argsort(np.abs(shap_vals))[-15:]  # Top 15
    sorted_shap = shap_vals[sorted_indices]
    sorted_features = [top_30_features_by_variance[i] for i in sorted_indices]
    sorted_meaningful_names = [feature_name_mapping.get(f, f)[:25] for f in sorted_features]
    
    colors = ['red' if val > 0 else 'blue' for val in sorted_shap]
    ax2.barh(range(len(sorted_shap)), sorted_shap, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(sorted_shap)))
    ax2.set_yticklabels(sorted_meaningful_names, fontsize=10)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('SHAP Value (Impact on Prediction)')
    ax2.set_title(f' SHAP Values for Legitimate Case\n(Top 15 Contributing Features)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
else:
    ax2.text(0.5, 0.5, 'No legitimate cases in sample', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title(' SHAP Values for Legitimate Case\n(No Legitimate Cases Available)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

```


    
![png](fraud_files/fraud_56_0.png)
    

