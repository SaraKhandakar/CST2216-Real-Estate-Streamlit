# =========================
# Model Persistence & Prediction Module
# =========================
# This file handles saving/loading machine learning models
# and generating predictions based on user input.

import joblib
import pandas as pd

def save_model(model, filepath):
    """
    Save a trained machine learning model to disk.

    Parameters:
    model: Trained machine learning model object
    filepath (str): Path where the model will be saved

    Purpose:
    This allows the trained model to be reused later
    without retraining.
    """

    # Save model using joblib for efficient serialization
    joblib.dump(model, filepath)


def load_model(filepath):
    """
    Load a trained machine learning model from disk.

    Parameters:
    filepath (str): Path to the saved model file

    Returns:
    model: Loaded machine learning model

    Purpose:
    This function retrieves the trained model so it can
    be used for making predictions in the application.
    """

    # Load model from file
    return joblib.load(filepath)


def make_prediction(model, input_data):
    """
    Generate a prediction using the trained model.

    Parameters:
    model: Loaded machine learning model
    input_data (dict): Dictionary of user input features

    Returns:
    float: Predicted value (house price)

    Purpose:
    Converts user input into the correct format and
    passes it to the model for prediction.
    """

    # Convert input dictionary into a DataFrame
    # Model expects tabular input with column names
    input_df = pd.DataFrame([input_data])

    # Generate prediction using the trained model
    prediction = model.predict(input_df)

    # Return the first prediction (single input case)
    return prediction[0]