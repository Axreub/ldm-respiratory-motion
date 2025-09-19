import logging
import logging.handlers
from pathlib import Path


def setup_training_logging(
    logfile_path: str,
    log_level: str = "INFO",
    log_format: str = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
    console_output: bool = True,
) -> logging.Logger:
    """
    Sets up logging configuration for training runs.

    Args:
        logfile_path (str): Path for saving logs
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format (str): Format string for log messages
        max_bytes (int): Maximum size of each log file
        backup_count (int): Number of backup files to keep
        console_output (bool): Whether to also log to console

    Returns:
        logging.Logger: Logger instance
    """

    log_dir = Path(logfile_path) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"training_logs.log"

    logger = logging.getLogger("training")
    logger.setLevel(getattr(logging, log_level.upper()))

    logger.handlers = []

    formatter = logging.Formatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Rotating allows long files to go to a new file if max bytes is reached
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
