import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


def get_logger(
    name: str,
    log_dir: Optional[str] = "outputs/logs",
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """
    Create and configure a logger with console and file handlers.

    Args:
        name:      Logger name (usually __name__ of the calling module).
        log_dir:   Directory where log files are saved.
        level:     Logging level (DEBUG, INFO, WARNING, ERROR).
        console:   Whether to log to stdout.
        file:      Whether to log to a file.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if file and log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{name.replace('.', '_')}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class EpochLogger:
    """
    Lightweight metric tracker that logs and stores epoch-level metrics.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.history: dict = {}

    def log_epoch(self, epoch: int, metrics: dict) -> None:
        """
        Log metrics at the end of an epoch and append to history.

        Args:
            epoch:   Current epoch number.
            metrics: Dict of metric_name -> float value.
        """
        parts = [f"Epoch {epoch:04d}"]
        for key, val in metrics.items():
            parts.append(f"{key}: {val:.5f}")
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(val)
        self.logger.info(" | ".join(parts))

    def get_history(self, metric: str) -> list:
        return self.history.get(metric, [])