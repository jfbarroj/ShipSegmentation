"""
Defines the settings that the project will internally use.

This module loads environment variables from .env (which must be carefully
placed for it to work correctly). Alternatively, default values are given
to each relevant setting of the project.
"""

import os
from dotenv import load_dotenv


load_dotenv(override=True)


DATASET_PATH = os.getenv('DATASET_PATH', 'data/MASATI')
