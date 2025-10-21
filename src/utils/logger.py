"""
Logging utility for fraud detection pipeline
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


class PipelineLogger:
    """Custom logger for fraud detection pipeline"""

    def __init__(self, name: str, log_dir: str = "logs", level=logging.INFO):
        """
        Initialize logger

        Args:
            name: Logger name
            log_dir: Directory to store log files
            level: Logging level
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Remove existing handlers
        self.logger.handlers = []

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        """Get logger instance"""
        return self.logger


def get_logger(name: str, log_dir: str = "logs", level=logging.INFO):
    """
    Get or create a logger

    Args:
        name: Logger name
        log_dir: Directory to store log files
        level: Logging level

    Returns:
        logging.Logger: Logger instance
    """
    pipeline_logger = PipelineLogger(name, log_dir, level)
    return pipeline_logger.get_logger()


def log_dataframe_info(logger, df, name: str = "DataFrame"):
    """
    Log information about a PySpark DataFrame

    Args:
        logger: Logger instance
        df: PySpark DataFrame
        name: Name of the DataFrame
    """
    count = df.count()
    logger.info(f"{name} - Row count: {count:,}")
    logger.info(f"{name} - Columns: {len(df.columns)}")
    logger.debug(f"{name} - Schema: {df.schema}")


def log_step(logger, step_name: str):
    """
    Log a pipeline step

    Args:
        logger: Logger instance
        step_name: Name of the step
    """
    separator = "=" * 80
    logger.info(f"\n{separator}")
    logger.info(f"STEP: {step_name}")
    logger.info(f"{separator}\n")
