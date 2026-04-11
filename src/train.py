# =========================
# Model Training & Evaluation Module
# =========================
# This file contains functions for training machine learning models
# and evaluating their performance using appropriate metrics.

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_models(x_train, y_train, x_test, y_test):
    """
    Train multiple machine learning models and evaluate their performance.

    Parameters:
    x_train (DataFrame): Training feature set
    y_train (Series): Training target values
    x_test (DataFrame): Testing feature set
    y_test (Series): Testing target values

    Returns:
    dict: Dictionary containing trained models and their evaluation metrics

    Purpose:
    - Train different regression models
    - Compare their performance using Mean Absolute Error (MAE)
    - Help select the best model for prediction
    """

    # Dictionary to store models and their evaluation results
    results = {}

    # =========================
    # Linear Regression Model
    # =========================
    # A simple baseline model that assumes a linear relationship
    lr_model = LinearRegression()

    # Train model on training data
    lr_model.fit(x_train, y_train)

    # Predict on unseen test data
    lr_pred = lr_model.predict(x_test)

    # Evaluate model using Mean Absolute Error
    # MAE measures average prediction error (lower is better)
    lr_mae = mean_absolute_error(y_test, lr_pred)

    # Store model and its performance
    results["Linear Regression"] = {
        "model": lr_model,
        "mae": lr_mae
    }

    # =========================
    # Random Forest Model
    # =========================
    # An ensemble model that builds multiple decision trees
    # and averages their predictions for better accuracy
    rf_model = RandomForestRegressor(random_state=42)

    # Train model
    rf_model.fit(x_train, y_train)

    # Predict on test data
    rf_pred = rf_model.predict(x_test)

    # Evaluate performance using MAE
    rf_mae = mean_absolute_error(y_test, rf_pred)

    # Store model and results
    results["Random Forest"] = {
        "model": rf_model,
        "mae": rf_mae
    }

    # =========================
    # Return Results
    # =========================
    return results