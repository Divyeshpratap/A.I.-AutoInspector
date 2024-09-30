# utils/helpers.py

import os
import requests
import logging

from pytz import timezone

logger = logging.getLogger(__name__)

def basename_filter(path):
    """
    Returns the base name of the given file path.
    """
    return os.path.basename(path)

