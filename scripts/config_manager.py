"""
Configuration Manager for Fraud Detection Scripts
=================================================

Handles loading and accessing configuration from clickhouse_config.ini
"""

import os
import configparser
from pathlib import Path


class Config:
    """Configuration manager for ClickHouse and feature generation settings."""
    
    def __init__(self, config_file=None):
        """
        Initialize configuration from file.
        
        Args:
            config_file: Path to config file. If None, looks in standard locations.
        """
        self.config = configparser.ConfigParser()
        
        # Try to find config file in standard locations
        if config_file is None:
            config_file = self._find_config_file()
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}\n"
                f"Please create config/clickhouse_config.ini with ClickHouse credentials"
            )
        
        self.config.read(config_file)
        self.config_file = config_file
    
    def _find_config_file(self):
        """Find configuration file in standard locations."""
        # Standard locations to search
        search_paths = [
            # Current directory
            'clickhouse_config.ini',
            'config/clickhouse_config.ini',
            # Script directory
            os.path.join(os.path.dirname(__file__), 'clickhouse_config.ini'),
            os.path.join(os.path.dirname(__file__), '../config/clickhouse_config.ini'),
            # Project root
            '/root/research-dir/dev/jazzcash-fraud-detection/config/clickhouse_config.ini',
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        # Default to project root location
        return '/root/research-dir/dev/jazzcash-fraud-detection/config/clickhouse_config.ini'
    
    # ClickHouse Connection Settings
    @property
    def host(self):
        return self.config.get('clickhouse', 'host', fallback='localhost')
    
    @property
    def port(self):
        return self.config.getint('clickhouse', 'port', fallback=9000)
    
    @property
    def database(self):
        return self.config.get('clickhouse', 'database', fallback='public')
    
    @property
    def user(self):
        return self.config.get('clickhouse', 'user', fallback='default')
    
    @property
    def password(self):
        return self.config.get('clickhouse', 'password', fallback='')
    
    @property
    def cluster(self):
        return self.config.get('clickhouse', 'cluster', fallback='my_cluster_2shards')
    
    # Table Names
    @property
    def source_table(self):
        return self.config.get('tables', 'source_table', fallback='stixor_iar_distributed')
    
    @property
    def source_mbar_table(self):
        return self.config.get('tables', 'source_mbar_table', fallback='stixor_mbar_v')
    
    @property
    def user_features_table(self):
        return self.config.get('tables', 'user_features_table', fallback='ac_from_features_distributed')
    
    @property
    def transaction_features_table(self):
        return self.config.get('tables', 'transaction_features_table', fallback='transaction_features_distributed')
    
    @property
    def combined_features_table(self):
        return self.config.get('tables', 'combined_features_table', fallback='combined_features_distributed')
    
    # Fraud Label Tables
    @property
    def fraud_accounts_table(self):
        return self.config.get('fraud_labels', 'fraud_accounts_table', fallback='fraud_accounts_with_types')
    
    @property
    def victim_accounts_table(self):
        return self.config.get('fraud_labels', 'victim_accounts_table', fallback='victim_accounts_with_types')
    
    @property
    def complaint_accounts_table(self):
        return self.config.get('fraud_labels', 'complaint_accounts_table', fallback='complaint_accounts_with_types')
    
    # Feature Generation Settings
    @property
    def default_lookback_days(self):
        return self.config.getint('features', 'default_lookback_days', fallback=7)
    
    @property
    def transaction_lookback_days(self):
        return self.config.getint('features', 'transaction_lookback_days', fallback=3)
    
    def get_connection_params(self):
        """Get dictionary of connection parameters for ClickHouse client."""
        return {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.user,
            'password': self.password
        }
    
    def __str__(self):
        """String representation of config (without password)."""
        return (
            f"ClickHouse Config:\n"
            f"  Host: {self.host}:{self.port}\n"
            f"  Database: {self.database}\n"
            f"  User: {self.user}\n"
            f"  Cluster: {self.cluster}\n"
            f"  Source Table: {self.source_table}\n"
            f"  Config File: {self.config_file}"
        )


# Singleton instance
_config_instance = None


def get_config(config_file=None):
    """
    Get configuration instance (singleton pattern).
    
    Args:
        config_file: Path to config file (optional)
        
    Returns:
        Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_file)
    
    return _config_instance


if __name__ == '__main__':
    # Test configuration loading
    try:
        config = get_config()
        print(config)
        print("\n✅ Configuration loaded successfully!")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
