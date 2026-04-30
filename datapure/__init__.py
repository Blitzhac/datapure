"""
datapure — AI-powered data cleaning for data scientists.

Usage:
    from datapure.cleaners import MissingValueCleaner
    from datapure.core import Pipeline
"""

__version__ = "0.1.0"
__author__ = "Basil Anil"

# Package-level logger — users can configure this
import logging

logging.getLogger("datapure").addHandler(logging.NullHandler())
