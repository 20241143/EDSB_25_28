
import pandas as pd
import numpy as np
import unicodedata
import re
import os
from dotenv import load_dotenv
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import datetime as dt

# Load environment variables
load_dotenv()

BRONZE_PATH = os.getenv("BRONZE_PATH")
SILVER_PATH = os.getenv("SILVER_PATH")
GOLD_PATH = os.getenv("GOLD_PATH")
FIG_DIR = os.getenv("FIG_DIR")

# Utility functions to use across notebooks

def normalizeString(string, sep = '_'):
  """
  Normalizes a string name to a standardized format.

  This function performs the following transformations on the input string:
  1. Normalizes Unicode characters to ASCII using NFKD form.
  2. Converts all characters to lowercase.
  3. Replaces any non-alphanumeric characters with underscores.
  4. Removes leading and trailing underscores.

  Parameters:
      string (str): The original column name to normalize.

  Returns:
      str: A normalized column name containing only lowercase letters, numbers, and underscores.
  """
  if string is np.nan or None:
    return np.nan

  string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8')
  string = string.lower()
  string = re.sub(r'[^a-z0-9]+', '_', string)
  string = string.strip('_')

  return string

def convert_target(df, target_col='y'):
    """
    Converts the target column from 'yes'/'no' to 1/0.
    """
    if target_col in df.columns:
        df[target_col] = df[target_col].map({'yes': 1, 'no': 0}).astype(int)
    return df