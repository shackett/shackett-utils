"""Time-related utility functions."""

from datetime import datetime


def get_timestamp() -> str:
    """Get current timestamp formatted as string.

    Returns
    -------
    str
        Current timestamp in format 'YYYY-MM-DD HH:MM:SS'
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
