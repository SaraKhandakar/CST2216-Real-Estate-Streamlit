# =========================
# Data Loading Module
# =========================
# This file contains functions responsible for loading datasets
# used in the machine learning pipeline.

import pandas as pd

def load_data(filepath):
    """
    Load dataset from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file containing the dataset.

    Returns:
    pandas.DataFrame: Loaded dataset as a DataFrame.

    Purpose:
    This function centralizes data loading so it can be reused
    across training, testing, and prediction modules.
    """

    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(filepath)

    # Return the loaded dataset
    return df