#!/usr/bin/env python3
"""
Logging Configuration for Arm Control Module

Provides consistent logging across all components with
file and console output, timestamps, and log levels.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# Log directory
LOG_DIR = Path(__file__).parent.parent / "logs"


def setup_logging(
    name: str = "arm_control",
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_dir: Optional[Path] = None
) -> logging.Logger:
    """
    Set up logging for a module.

    Args:
        name: Logger name (typically module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Write logs to file
        log_to_console: Write logs to console
        log_dir: Directory for log files (default: logs/)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Format with timestamp, level, module, and message
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        if log_dir is None:
            log_dir = LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # File gets all details
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default configuration.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name)
    return logger


# Pre-configured loggers for main components
class Loggers:
    """Container for component loggers."""

    @staticmethod
    def controller() -> logging.Logger:
        return get_logger("arm.controller")

    @staticmethod
    def hardware() -> logging.Logger:
        return get_logger("arm.hardware")

    @staticmethod
    def simulation() -> logging.Logger:
        return get_logger("arm.simulation")

    @staticmethod
    def can_bus() -> logging.Logger:
        return get_logger("arm.can_bus")

    @staticmethod
    def signal_bridge() -> logging.Logger:
        return get_logger("arm.signal_bridge")

    @staticmethod
    def vision() -> logging.Logger:
        return get_logger("arm.vision")

    @staticmethod
    def playback() -> logging.Logger:
        return get_logger("arm.playback")


# Initialize default logging on import
_root_logger = setup_logging("arm_control", log_to_file=True, log_to_console=True)
