# =========================
# Data Preprocessing Module
# =========================
# This file contains functions used to prepare the dataset
# before training the machine learning model.

from sklearn.model_selection import train_test_split

def split_data(df, target_column):
    """
    Split dataset into features and target, and then into
    training and testing sets.

    Parameters:
    df (pandas.DataFrame): The complete dataset
    target_column (str): Name of the target (dependent variable)

    Returns:
    tuple:
        X (DataFrame): Feature variables
        y (Series): Target variable
        x_train, x_test, y_train, y_test: Split datasets

    Purpose:
    - Separate input features (X) and output variable (y)
    - Split data into training and testing sets
    - Ensure model is evaluated on unseen data
    """

    # =========================
    # Separate Features and Target
    # =========================
    # X contains all independent variables (inputs)
    # y contains the dependent variable (output to predict)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # =========================
    # Train-Test Split
    # =========================
    # Split data into training (80%) and testing (20%)
    # random_state ensures reproducibility of results
    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,        # 20% data for testing
        random_state=42       # ensures same split every run
    )

    # =========================
    # Return Split Data
    # =========================
    return X, y, x_train, x_test, y_train, y_test