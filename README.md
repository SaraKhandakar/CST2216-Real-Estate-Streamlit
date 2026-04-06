# Real Estate Price Prediction App

## Overview
- This project predicts real estate prices using machine learning models
- The original Jupyter Notebook was converted into a modular Python project
- The app is deployed using Streamlit

## Dataset
- File: final.csv
- Target variable: price
- Features include:
  - year_sold
  - property_tax
  - insurance
  - beds
  - baths
  - sqft
  - year_built
  - lot_size
  - basement
  - popular
  - recession
  - property_age
  - property_type_Condo

## Models Used
- Linear Regression
- Random Forest Regressor
- Best model selected based on Mean Absolute Error (MAE)

## Key Insight
- Property_age was expected to reduce price
- Analysis showed almost no relationship with price
- Random Forest still uses it due to non-linear feature interactions
- Other features like sqft, tax, and beds have stronger impact on price

## Project Structure
- app/
  - streamlit_app.py
- src/
  - data_loader.py
  - preprocess.py
  - train.py
  - predict.py
  - logger.py
- models/
  - real_estate_model.pkl
- data/
  - final.csv
- notebooks/
  - Real_Estate.ipynb
- requirements.txt
- README.md
- train_model.py

## Installation
- Install required libraries
- pip install -r requirements.txt

## Run Locally
- Train the model
- python train_model.py

- Run the Streamlit app
- streamlit run app/streamlit_app.py

## Deployment
- This app is deployed using Streamlit Community Cloud
- Streamlit App Link: https://cst2216-real-estate-app-ryaw4bqztbrmretiqv88zp.streamlit.app/

## GitHub Repository
- https://github.com/SaraKhandakar/CST2216-Real-Estate-Streamlit

## Author
- Shara Khandakar
- Algonquin College
- Business Intelligence Systems Infrastructure
