import os
import sys
import logging
from datetime import datetime
from zoneinfo import ZoneInfo


def to_datetime(date_str):
    """
    Return a datetime object from a date string in MMDDYYYY or YYYYMMDD format.
    Set the HH:MM:SS time to 00:00:00 Pacific Time.
    """
    # If it's already a datetime, we're good to go
    if isinstance(date_str, datetime):
        return date_str

    # Try ISO format first (e.g. "2024-01-22T00:00:00-08:00")
    if "T" in str(date_str):
        try:
            dt = datetime.fromisoformat(str(date_str))
            return dt.astimezone(ZoneInfo("America/Los_Angeles"))
        except (ValueError, TypeError, AttributeError):
            pass  # If ISO parsing fails, continue with other formats

    # Remove slashes and dashes so we can handle formats like MM/DD/YYYY, MM-DD-YYYY, etc
    date_str = str(date_str).replace("/", "").replace("-", "").strip()

    # Add a leading 0 if needed (if date_str was specified as an int, leading 0s get clipped)
    date_str = date_str.zfill(8)

    if len(date_str) != 8:
        raise ValueError("Date string must be exactly 8 characters long. "
                         f"Got date = {date_str} ({len(date_str)} characters)")

    # Auto-detect the date format: if date starts with "20", it must be YYYYMMDD format
    if date_str.startswith("20"):
        date_format = "%Y%m%d"
    # Otherwise, assume MMDDYYYY
    else:
        date_format = "%m%d%Y"

    # Convert to datetime and set timezone to Pacific 
    dt = datetime.strptime(date_str, date_format)
    dt = dt.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
    return dt


def setup_logger(log_name, path_logfile_info, path_logfile_warn, path_logfile_debug) -> logging.Logger:
    """
    Sets up a logger that outputs to 3 different files:
    - File for all general logs (log level INFO and above).
    - File for warnings and errors (log level WARNING and above).
    - File for detailed debug output (log level DEBUG and above)

    Parameters:
        log_name: Name of the logfile (for logger identification)
        path_logfile_info: Path to the logfile for info messages
        path_logfile_warn: Path to the logfile for warning and error messages
        path_logfile_debug: Path to the logfile for debug messages

    Returns:
        logging.Logger
    """

    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)  # Capture all levels (DEBUG and above)

    # Define format for log messages
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%d-%b-%y %H:%M:%S")

    # Handler for logging messages INFO and above to a file
    fileHandler_info = logging.FileHandler(path_logfile_info, mode="w")
    fileHandler_info.setFormatter(formatter)
    fileHandler_info.setLevel(logging.INFO)

    # Handler for logging messages WARNING and above to a file
    fileHandler_warn = logging.FileHandler(path_logfile_warn, mode="w")
    fileHandler_warn.setFormatter(formatter)
    fileHandler_warn.setLevel(logging.WARNING)

    # Handler for logging messages DEBUG and above to a file
    fileHandler_debug = logging.FileHandler(path_logfile_debug, mode="w")
    fileHandler_debug.setFormatter(formatter)
    fileHandler_debug.setLevel(logging.DEBUG)

    # Add handlers to the logger
    logger.addHandler(fileHandler_info)
    logger.addHandler(fileHandler_warn)
    logger.addHandler(fileHandler_debug)

    return logger


def setup_stdout_logger(log_name: str) -> logging.Logger:
    """
    Sets up a logger that outputs to stdout. 
    Useful for running functions that expect a logger, 
    but we don't actually care to create a logfile (e.g. when in a jupyter notebook)
    
    Parameters:
        log_name: Name of the logfile (for logger identification)

    Returns:
        logging.Logger
    """

    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers 
    # (needed if we accidentally create multiple loggers of same name to avoid duplicate prints)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define format for log messages
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%d-%b-%y %H:%M:%S")

    # Single handler for all levels to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger


def get_logger_directory(logger: logging.Logger) -> str:
    """
    Helper to get the directory path where the first FileHandler of the logger writes logs.

    Parameters:
        logger (logging.Logger): Logger to track conversion progress

    Returns:
        str: Path to the log directory
    """
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return os.path.dirname(handler.baseFilename)
    raise ValueError("Logger has no FileHandler with a valid log file path.")