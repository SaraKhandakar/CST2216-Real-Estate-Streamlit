# =========================
# Logging Configuration Module
# =========================
# This file sets up a centralized logger for the application.
# Logging helps track execution flow, debug errors, and monitor usage.

import logging

def setup_logger():
    """
    Configure and return a logger for the application.

    Returns:
    logging.Logger: Configured logger instance.

    Purpose:
    This function standardizes logging across the entire project,
    ensuring that all modules write logs in a consistent format
    to the same log file.
    """

    # Configure logging settings
    logging.basicConfig(
        filename="app.log",                      # Log file name
        level=logging.INFO,                      # Log level (INFO, WARNING, ERROR)
        format="%(asctime)s - %(levelname)s - %(message)s"  # Log message format
    )

    # Return the root logger instance
    return logging.getLogger()