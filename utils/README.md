# PostgreSQL Database Connection Testing

This directory contains utilities for testing PostgreSQL database connections in the JazzCash fraud detection project.

## Files Overview

### 1. `test_db_conn.py` (Full-featured test)
- Comprehensive PostgreSQL connection testing
- Database information retrieval
- Basic operations testing (CREATE, INSERT, SELECT)
- Performance testing
- Table listing functionality

### 2. `simple_db_test.py` (Basic connectivity test)
- Network connectivity test (uses only standard library)
- PostgreSQL driver availability check
- Minimal dependencies

### 3. `requirements.txt`
- Required Python packages for database connectivity

### 4. `.env.example`
- Example environment configuration file

## Setup Instructions

### 1. Install Dependencies

```bash
# Install required Python packages
pip install -r requirements.txt

# Or install individually
pip install psycopg2-binary python-dotenv
```

### 2. Configure Database Connection

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual database credentials
nano .env
```

Update the `.env` file with your PostgreSQL connection details:
```env
DB_HOST=your_postgres_host
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password
```

### 3. Alternative Configuration Methods

If you don't want to use a `.env` file, you can set environment variables directly:

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=fraud_detection
export DB_USER=postgres
export DB_PASSWORD=your_password
```

Or modify the scripts to use hardcoded values (not recommended for production).

## Usage

### Quick Network Test (No Dependencies)

```bash
# Test basic network connectivity and driver availability
python simple_db_test.py
```

This will:
- Test network connectivity to the PostgreSQL server
- Check if psycopg2 driver is available
- Provide setup guidance

### Full Database Connection Test

```bash
# Comprehensive database connection and functionality test
python test_db_conn.py
```

This will:
1. Test database connection
2. Retrieve database information (version, size, etc.)
3. List available tables
4. Test basic SQL operations
5. Run performance benchmarks
6. Provide detailed output

## Example Output

### Simple Test Output
```
==================================================
PostgreSQL Simple Connection Test
==================================================

Connection Parameters:
  Host: localhost
  Port: 5432
  Database: fraud_detection
  User: postgres
  Password: ********

1. Network Connectivity Test:
Testing network connectivity to localhost:5432...
✓ Network connection to localhost:5432 successful!

2. PostgreSQL Driver Test:
✓ psycopg2 library is available

==================================================
Test Summary:
  Network Connectivity: ✓ PASS
  PostgreSQL Driver:    ✓ PASS

✓ Ready to test full PostgreSQL connection!
  Run: python test_db_conn.py
==================================================
```

### Full Test Output
```
============================================================
PostgreSQL Database Connection Test
============================================================

1. Testing Database Connection:
2024-10-22 10:30:45,123 - INFO - Attempting to connect to PostgreSQL database...
2024-10-22 10:30:45,123 - INFO - Host: localhost
2024-10-22 10:30:45,123 - INFO - Port: 5432
2024-10-22 10:30:45,123 - INFO - Database: fraud_detection
2024-10-22 10:30:45,123 - INFO - User: postgres
2024-10-22 10:30:45,150 - INFO - ✓ Successfully connected to PostgreSQL database!

2. Database Information:
   Postgresql Version: PostgreSQL 14.2 on x86_64-pc-linux-gnu
   Current Database: fraud_detection
   Current User: postgres
   Server Time: 2024-10-22 10:30:45.165432
   Database Size: 156 MB

3. Available Tables:
   - fraud_transactions
   - customer_accounts
   - transaction_logs
   ... and 15 more tables

4. Testing Basic Operations:
2024-10-22 10:30:45,200 - INFO - Testing CREATE operation...
2024-10-22 10:30:45,205 - INFO - Testing INSERT operation...
2024-10-22 10:30:45,210 - INFO - Testing SELECT operation...
2024-10-22 10:30:45,215 - INFO - ✓ Basic operations test successful!

5. Performance Test:
   Iterations: 100
   Total Time: 0.052 seconds
   Queries/Second: 1923.08
   Avg Query Time: 0.520 ms

============================================================
Database connection test completed successfully!
============================================================
```

## Troubleshooting

### Common Issues

1. **Connection Refused Error**
   ```
   psycopg2.OperationalError: could not connect to server: Connection refused
   ```
   - Check if PostgreSQL server is running
   - Verify host and port configuration
   - Check firewall settings

2. **Authentication Failed**
   ```
   psycopg2.OperationalError: FATAL: password authentication failed for user
   ```
   - Verify username and password
   - Check PostgreSQL `pg_hba.conf` configuration
   - Ensure user has necessary permissions

3. **Database Does Not Exist**
   ```
   psycopg2.OperationalError: FATAL: database "xyz" does not exist
   ```
   - Create the database or use an existing one
   - Verify database name spelling

4. **psycopg2 Import Error**
   ```
   ImportError: No module named 'psycopg2'
   ```
   - Install psycopg2: `pip install psycopg2-binary`
   - Use `simple_db_test.py` to check prerequisites

### Docker PostgreSQL Setup (for testing)

If you need a PostgreSQL instance for testing:

```bash
# Run PostgreSQL in Docker
docker run --name test-postgres \
  -e POSTGRES_PASSWORD=testpass \
  -e POSTGRES_DB=fraud_detection \
  -p 5432:5432 \
  -d postgres:14

# Set environment variables for testing
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=fraud_detection
export DB_USER=postgres
export DB_PASSWORD=testpass
```

## Integration with Spark

Since the project includes `postgresql-42.7.1.jar`, you can also test PostgreSQL connectivity from Spark:

```python
# Example Spark PostgreSQL connection test
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("PostgreSQL Test") \
    .config("spark.jars", "utils/postgresql-42.7.1.jar") \
    .getOrCreate()

# Test reading from PostgreSQL
df = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/fraud_detection") \
    .option("dbtable", "information_schema.tables") \
    .option("user", "postgres") \
    .option("password", "your_password") \
    .load()

df.show(5)
```

## Security Notes

- Never commit `.env` files with actual credentials to version control
- Use environment variables or secure credential management in production
- Consider using connection pooling for high-throughput applications
- Enable SSL for production database connections