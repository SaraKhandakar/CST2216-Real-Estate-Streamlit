import streamlit as st
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.predict import load_model, make_prediction

MODEL_PATH = os.path.join(BASE_DIR, "models", "real_estate_model.pkl")

st.set_page_config(page_title="Real Estate Price Prediction", layout="centered")

st.title("Real Estate Price Prediction App")
st.write("Enter the property details below to predict the house price.")

try:
    model = load_model(MODEL_PATH)

    year_sold = st.number_input("Year Sold", min_value=2000, max_value=2100, value=2020)
    property_tax = st.number_input("Property Tax", min_value=0.0, value=5000.0)
    insurance = st.number_input("Insurance", min_value=0.0, value=1500.0)
    beds = st.number_input("Beds", min_value=0, value=3)
    baths = st.number_input("Baths", min_value=0, value=2)
    sqft = st.number_input("Square Feet (sqft)", min_value=0.0, value=2000.0)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2100, value=2005)
    lot_size = st.number_input("Lot Size", min_value=0.0, value=5000.0)

    basement = st.selectbox("Basement", [0, 1])
    popular = st.selectbox("Popular Area", [0, 1])
    recession = st.selectbox("Recession", [0, 1])
    property_age = st.number_input("Property Age", min_value=0, value=15)
    property_type_condo = st.selectbox("Property Type Condo", [0, 1])

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
            "basement": basement,
            "popular": popular,
            "recession": recession,
            "property_age": property_age,
            "property_type_Condo": property_type_condo
        }

        prediction = make_prediction(model, input_data)
        st.success(f"Predicted Price: ${prediction:,.2f}")

except FileNotFoundError:
    st.error("Model file not found. Please run train_model.py first.")
except Exception as e:
    st.error(f"An error occurred: {e}")