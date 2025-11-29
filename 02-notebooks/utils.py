
import csv
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
MODEL_DIR = os.getenv("MODEL_DIR")

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
    try:
        if target_col in df.columns:
            df[target_col] = df[target_col].map({'yes': 1, 'no': 0}).astype(int)
    except Exception as e:
        print(f"Error converting target column: {e}")
    return df

def save_metrics(model, model_name, metrics_dict):
    """
    Save the model's metrics in a single CSV file for comparison.
    
    Parameters:
    -----------
    model_name : str
        Identifier for the model (e.g. "baseline_glm", "baseline_lr", "random_forest").
    metrics_dict : dict
        Dictionary of metrics such as:
        {
            "AUC": 0.85,
            "Accuracy": 0.91,
            "Precision": 0.40,
            "Recall": 0.30,
            "F1": 0.34,
            "LogLoss": 0.57
        }
    """

    # Always add model name and timestamp
    metrics_dict = {
        "Model": model_name,
        "Timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **metrics_dict
    }

    path = MODEL_DIR+model
    os.makedirs(path, exist_ok=True)

    file_exists = os.path.isfile(os.path.join(path, model_name + "_metrics.csv"))

    with open(os.path.join(path, model_name + "_metrics.csv"), "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_dict.keys(), delimiter=';')

        # Write header only for first-time creation
        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics_dict)

    print(f"Metrics saved for {model_name} → {path}/{model_name}_metrics.csv")