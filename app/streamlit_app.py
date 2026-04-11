# =========================
# Import Required Libraries
# =========================
import streamlit as st
import sys
import os

# =========================
# Set Project Base Directory
# =========================
# This ensures that we can import modules from the parent directory (src folder)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# Import custom functions for prediction and logging
from src.predict import load_model, make_prediction
from src.logger import setup_logger

# =========================
# Initialize Logger
# =========================
# Logger helps track application flow and debug errors
logger = setup_logger()
logger.info("Real Estate Streamlit app started")

# =========================
# Model Path Configuration
# =========================
# Path to the trained machine learning model
MODEL_PATH = os.path.join(BASE_DIR, "models", "real_estate_model.pkl")

# =========================
# Streamlit Page Configuration
# =========================
# Set title, icon, and layout of the web app
st.set_page_config(
    page_title="Real Estate Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# =========================
# Custom UI Styling (CSS)
# =========================
# Improves visual appearance of the app
st.markdown(
    """
    <style>
    .main {
        padding-top: 1.5rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-size: 16px;
        font-weight: 600;
    }
    .title-box {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 14px;
        margin-bottom: 20px;
        border: 1px solid #e6eaf0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# App Title Section
# =========================
# Displays main heading and short description for user guidance
st.markdown(
    """
    <div class="title-box">
        <h1 style="margin-bottom: 0.3rem;">🏠 Real Estate Price Prediction App</h1>
        <p style="margin-bottom: 0;">
            Enter the property details below to estimate the predicted house price.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

try:
    # =========================
    # Load Trained Model
    # =========================
    # Load the pre-trained model from the saved file
    logger.info(f"Loading model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully")

    # =========================
    # User Input Section
    # =========================
    # Collect property-related inputs from the user
    st.subheader("Property Information")

    # Create two-column layout for better UI organization
    col1, col2 = st.columns(2)

    # Left column inputs
    with col1:
        year_sold = st.number_input("Year Sold", min_value=2000, max_value=2100, value=2020)
        property_tax = st.number_input("Property Tax", min_value=0.0, value=5000.0)
        insurance = st.number_input("Insurance", min_value=0.0, value=1500.0)
        beds = st.number_input("Beds", min_value=0, value=3)
        baths = st.number_input("Baths", min_value=0, value=2)
        sqft = st.number_input("Square Feet", min_value=0.0, value=2000.0)

    # Right column inputs
    with col2:
        year_built = st.number_input("Year Built", min_value=1800, max_value=2100, value=2005)
        lot_size = st.number_input("Lot Size", min_value=0.0, value=5000.0)
        basement = st.selectbox("Basement", ["No", "Yes"])
        popular = st.selectbox("Popular Area", ["No", "Yes"])
        recession = st.selectbox("Recession", ["No", "Yes"])
        property_type_condo = st.selectbox("Property Type", ["Non-Condo", "Condo"])

    # =========================
    # Feature Engineering
    # =========================
    # Calculate property age based on year sold and year built
    property_age = year_sold - year_built
    st.info(f"Calculated Property Age: {property_age} years")

    # =========================
    # Prediction Button Logic
    # =========================
    # When user clicks button, prepare input and make prediction
    if st.button("Predict Price"):

        # Convert user input into model-compatible format
        # Categorical values are encoded into numeric (0/1)
        input_data = {
            "year_sold": year_sold,
            "property_tax": property_tax,
            "insurance": insurance,
            "beds": beds,
            "baths": baths,
            "sqft": sqft,
            "year_built": year_built,
            "lot_size": lot_size,
            "basement": 1 if basement == "Yes" else 0,
            "popular": 1 if popular == "Yes" else 0,
            "recession": 1 if recession == "Yes" else 0,
            "property_age": property_age,
            "property_type_Condo": 1 if property_type_condo == "Condo" else 0
        }

        # Log user input for debugging and tracking
        logger.info(f"Prediction requested with input: {input_data}")

        # Generate prediction using trained model
        prediction = make_prediction(model, input_data)

        # Log prediction result
        logger.info(f"Prediction generated: {prediction}")

        # Display prediction result to user
        st.success(f"Predicted Price: ${prediction:,.2f}")
        st.caption("This prediction is based on the trained machine learning model.")

# =========================
# Error Handling
# =========================
# Handle cases where model file is missing
except FileNotFoundError:
    logger.error("Model file not found")
    st.error("Model file not found. Please run train_model.py first.")

# Handle any unexpected runtime errors
except Exception as e:
    logger.error(f"Application error: {e}")
    st.error(f"An error occurred: {e}")