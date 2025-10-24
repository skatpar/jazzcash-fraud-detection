#!/usr/bin/env python3
"""
PostgreSQL Database Connection Test Script

This script tests the connection to a PostgreSQL database and provides
various utilities for validating database connectivity and basic operations.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    print("psycopg2 library not found. Please install it using:")
    print("pip install psycopg2-binary")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


DB_CONFIG = {
    'host': '10.205.161.118',
    'port': '5432',
    'database': 'db_fraud',
    'user': 'dfstechbi',
    'password': 'DfsTeChB1@923'
}

class PostgreSQLTester:
    """PostgreSQL database connection tester and utility class."""
    
    def __init__(self, connection_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the PostgreSQL tester.
        
        Args:
            connection_params: Dictionary containing connection parameters
        """
        self.connection_params = connection_params or self._get_default_params()
        self.connection = None
        
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default connection parameters from environment variables."""
        return {
            'host': os.getenv('DB_HOST', DB_CONFIG['host']),
            'port': int(os.getenv('DB_PORT', DB_CONFIG['port'])),
            'database': os.getenv('DB_NAME', DB_CONFIG['database']),
            'user': os.getenv('DB_USER', DB_CONFIG['user']),
            'password': os.getenv('DB_PASSWORD',DB_CONFIG['password']),
        }
    
    def test_connection(self) -> bool:
        """
        Test the database connection.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info("Attempting to connect to PostgreSQL database...")
            logger.info(f"Host: {self.connection_params['host']}")
            logger.info(f"Port: {self.connection_params['port']}")
            logger.info(f"Database: {self.connection_params['database']}")
            logger.info(f"User: {self.connection_params['user']}")
            
            self.connection = psycopg2.connect(**self.connection_params)
            logger.info("✓ Successfully connected to PostgreSQL database!")
            return True
            
        except psycopg2.OperationalError as e:
            logger.error(f"✗ Connection failed: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Unexpected error: {e}")
            return False
    
    def get_database_info(self) -> Optional[Dict[str, Any]]:
        """
        Get basic database information.
        
        Returns:
            Dictionary containing database information or None if connection fails
        """
        if not self.connection:
            if not self.test_connection():
                return None
        
        try:
            with self.connection.cursor() as cursor:
                # Get PostgreSQL version
                cursor.execute("SELECT version();")
                pg_version = cursor.fetchone()[0]
                
                # Get current database name
                cursor.execute("SELECT current_database();")
                current_db = cursor.fetchone()[0]
                
                # Get current user
                cursor.execute("SELECT current_user;")
                current_user = cursor.fetchone()[0]
                
                # Get server time
                cursor.execute("SELECT NOW();")
                server_time = cursor.fetchone()[0]
                
                # Get database size
                cursor.execute("""
                    SELECT pg_size_pretty(pg_database_size(current_database()));
                """)
                db_size = cursor.fetchone()[0]
                
                return {
                    'postgresql_version': pg_version,
                    'current_database': current_db,
                    'current_user': current_user,
                    'server_time': server_time,
                    'database_size': db_size
                }
                
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return None
    
    def list_tables(self, schema: str = 'public') -> Optional[list]:
        """
        List all tables in the specified schema.
        
        Args:
            schema: Schema name (default: 'public')
            
        Returns:
            List of table names or None if error occurs
        """
        if not self.connection:
            if not self.test_connection():
                return None
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = %s 
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name;
                """, (schema,))
                
                tables = [row[0] for row in cursor.fetchall()]
                return tables
                
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return None
    
    def test_basic_operations(self) -> bool:
        """
        Test basic database operations (CREATE, INSERT, SELECT, DROP).
        
        Returns:
            bool: True if all operations successful, False otherwise
        """
        if not self.connection:
            if not self.test_connection():
                return False
        
        test_table = f"test_table_{int(datetime.now().timestamp())}"
        
        try:
            with self.connection.cursor() as cursor:
                # Create test table
                logger.info("Testing CREATE operation...")
                cursor.execute(f"""
                    CREATE TEMP TABLE {test_table} (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(100),
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                
                # Insert test data
                logger.info("Testing INSERT operation...")
                cursor.execute(f"""
                    INSERT INTO {test_table} (name) 
                    VALUES ('Test Record 1'), ('Test Record 2');
                """)
                
                # Select test data
                logger.info("Testing SELECT operation...")
                cursor.execute(f"SELECT * FROM {test_table};")
                results = cursor.fetchall()
                
                if len(results) == 2:
                    logger.info("✓ Basic operations test successful!")
                    logger.info(f"Inserted and retrieved {len(results)} records")
                    return True
                else:
                    logger.error(f"✗ Expected 2 records, got {len(results)}")
                    return False
                    
        except Exception as e:
            logger.error(f"✗ Basic operations test failed: {e}")
            return False
        finally:
            # Commit any pending transactions
            try:
                self.connection.commit()
            except:
                pass
    
    def test_performance(self, iterations: int = 1000) -> Optional[Dict[str, float]]:
        """
        Test database performance with simple queries.
        
        Args:
            iterations: Number of iterations to run
            
        Returns:
            Dictionary with performance metrics or None if error occurs
        """
        if not self.connection:
            if not self.test_connection():
                return None
        
        try:
            start_time = datetime.now()
            
            with self.connection.cursor() as cursor:
                for i in range(iterations):
                    cursor.execute("SELECT 1;")
                    cursor.fetchone()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                'iterations': iterations,
                'total_time_seconds': duration,
                'queries_per_second': iterations / duration if duration > 0 else 0,
                'avg_query_time_ms': (duration * 1000) / iterations if iterations > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return None
    
    def close_connection(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed.")


def main():
    """Main function to run the database connection tests."""
    print("=" * 60)
    print("PostgreSQL Database Connection Test")
    print("=" * 60)
    
    # Initialize tester
    tester = PostgreSQLTester()
    
    # Test connection
    print("\n1. Testing Database Connection:")
    if not tester.test_connection():
        print("Connection test failed. Please check your database configuration.")
        return False
    
    # Get database information
    print("\n2. Database Information:")
    db_info = tester.get_database_info()
    if db_info:
        for key, value in db_info.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
    
    # List tables
    print("\n3. Available Tables:")
    tables = tester.list_tables()
    if tables:
        if tables:
            for table in tables[:10]:  # Show first 10 tables
                print(f"   - {table}")
            if len(tables) > 10:
                print(f"   ... and {len(tables) - 10} more tables")
        else:
            print("   No tables found in public schema")
    
    # Test basic operations
    print("\n4. Testing Basic Operations:")
    tester.test_basic_operations()
    
    # Test performance
    print("\n5. Performance Test:")
    perf_results = tester.test_performance(100)
    if perf_results:
        print(f"   Iterations: {perf_results['iterations']}")
        print(f"   Total Time: {perf_results['total_time_seconds']:.3f} seconds")
        print(f"   Queries/Second: {perf_results['queries_per_second']:.2f}")
        print(f"   Avg Query Time: {perf_results['avg_query_time_ms']:.3f} ms")
    
    # Close connection
    tester.close_connection()
    
    print("\n" + "=" * 60)
    print("Database connection test completed successfully!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        sys.exit(1)