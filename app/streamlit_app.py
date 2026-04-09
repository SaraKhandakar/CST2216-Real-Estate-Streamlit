import streamlit as st
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.predict import load_model, make_prediction
from src.logger import setup_logger

logger = setup_logger()
logger.info("Real Estate Streamlit app started")

MODEL_PATH = os.path.join(BASE_DIR, "models", "real_estate_model.pkl")

st.set_page_config(
    page_title="Real Estate Price Prediction",
    page_icon="🏠",
    layout="centered"
)

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
    logger.info(f"Loading model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully")

    st.subheader("Property Information")

    col1, col2 = st.columns(2)

    with col1:
        year_sold = st.number_input("Year Sold", min_value=2000, max_value=2100, value=2020)
        property_tax = st.number_input("Property Tax", min_value=0.0, value=5000.0)
        insurance = st.number_input("Insurance", min_value=0.0, value=1500.0)
        beds = st.number_input("Beds", min_value=0, value=3)
        baths = st.number_input("Baths", min_value=0, value=2)
        sqft = st.number_input("Square Feet", min_value=0.0, value=2000.0)

    with col2:
        year_built = st.number_input("Year Built", min_value=1800, max_value=2100, value=2005)
        lot_size = st.number_input("Lot Size", min_value=0.0, value=5000.0)
        basement = st.selectbox("Basement", ["No", "Yes"])
        popular = st.selectbox("Popular Area", ["No", "Yes"])
        recession = st.selectbox("Recession", ["No", "Yes"])
        property_type_condo = st.selectbox("Property Type", ["Non-Condo", "Condo"])

    property_age = year_sold - year_built
    st.info(f"Calculated Property Age: {property_age} years")

    if st.button("Predict Price"):
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

        logger.info(f"Prediction requested with input: {input_data}")
        prediction = make_prediction(model, input_data)
        logger.info(f"Prediction generated: {prediction}")

        st.success(f"Predicted Price: ${prediction:,.2f}")
        st.caption("This prediction is based on the trained machine learning model.")

except FileNotFoundError:
    logger.error("Model file not found")
    st.error("Model file not found. Please run train_model.py first.")
except Exception as e:
    logger.error(f"Application error: {e}")
    st.error(f"An error occurred: {e}")