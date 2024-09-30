# utils/filters.py

from flask import current_app
from pytz import timezone
import logging

logger = logging.getLogger(__name__)

def to_user_timezone(value, tz_string):
    """
    Converts a datetime object to the user's timezone.
    """
    try:
        user_tz = timezone(tz_string)
        return value.astimezone(user_tz)
    except Exception as e:
        logger.error(f"Error converting timezone: {str(e)}")
        return value  # Fallback to original time if conversion fails
