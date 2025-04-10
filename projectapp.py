import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# --- Load model and scalers ---
model = joblib.load("gdp_predictor_model.pkl")
scaler = joblib.load("scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")

# --- Load dataset to rebuild label encoder ---
df = pd.read_excel(r"C:\Users\joshin joseph\Desktop\project\world_bank_dataset.xlsx")
le = LabelEncoder()
le.fit(df['Country'])  # Fit on original names, not encoded numbers

# --- UI Setup ---
st.set_page_config(page_title="🌍 GDP Predictor", layout="centered")
st.title("🌍 GDP Prediction App")
st.markdown("Provide the country's key indicators below to predict **GDP**.")

# --- Input Form ---
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        country_name = st.selectbox("Country", sorted(df['Country'].unique()))
        year = st.number_input("Year", min_value=2010, max_value=2030, value=2020)
        population = st.number_input("Population (in millions)", min_value=0.0, value=50.0)
        life_expectancy = st.slider("Life Expectancy (years)", 40.0, 90.0, 70.0)

    with col2:
        unemployment_rate = st.slider("Unemployment Rate (%)", 0.1, 25.0, 5.0)
        co2_emissions = st.number_input("CO₂ Emissions (metric tons per capita)", 0.0, value=4.5)
        electricity_access = st.slider("Access to Electricity (%)", 0.0, 100.0, 90.0)
        gdp_per_capita = st.number_input("GDP per Capita (USD)", min_value=0.0, value=10000.0)

    gdp_growth_rate = st.slider("GDP Growth Rate (%)", -20.0, 15.0, 2.5)
    inverse_unemployment = 1 / (unemployment_rate + 1e-3)

    submitted = st.form_submit_button("📈 Predict GDP")

# --- Predict on Submit ---
if submitted:
    try:
        encoded_country = le.transform([country_name])[0]
    except:
        st.error("Country encoding failed.")
        st.stop()

    input_data = pd.DataFrame([[
        encoded_country, year, population, life_expectancy, unemployment_rate,
        co2_emissions, electricity_access, gdp_per_capita, gdp_growth_rate, inverse_unemployment
    ]], columns=[
        'Country', 'Year', 'Population', 'Life Expectancy', 'Unemployment Rate (%)',
        'CO2 Emissions (metric tons per capita)', 'Access to Electricity (%)',
        'GDP_per_Capita', 'GDP_Growth_Rate', 'Inverse_Unemployment'
    ])

    input_scaled = scaler.transform(input_data)
    prediction_scaled = model.predict(input_scaled).reshape(-1, 1)
    prediction = target_scaler.inverse_transform(prediction_scaled)[0][0]

    st.success(f"🌐 Predicted GDP: **₹ {prediction:,.2f}**")
    st.caption("Model: Tuned Gradient Boosting | Built with 💡 by Joshin")
