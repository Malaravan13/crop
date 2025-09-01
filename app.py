import streamlit as st
import pickle
import numpy as np

# Load trained model and scaler
with open("trained_model.sav", "rb") as f:
    rf_model, scaler = pickle.load(f)

# Streamlit page setup
st.set_page_config(page_title="Crop Yield Predictor", layout="centered")
st.title("🌾 Crop Yield Predictor")
st.write("Enter details below to estimate the **Crop Yield**.")

# Inputs
Year = st.number_input("Year", min_value=1900, max_value=2100, value=2020, step=1)
average_rain_fall_mm_per_year = st.number_input("Average Rainfall (mm per year)", 
                                                min_value=-2.0, max_value=5000.0, value=1000.0, step=0.01)
pesticides_tonnes = st.number_input("Pesticides used (tonnes)", 
                                    min_value=-2.0, max_value=10000.0, value=500.0, step=0.01)
avg_temp = st.number_input("Average Temperature (°C)", 
                           min_value=-10.0, max_value=50.0, value=25.0, step=0.1)

# Dropdowns for categorical values
area_options = {
    0: "Albania",
    1: "Algeria",
    2: "Angola",
    3: "Argentina",
    4: "Armenia",
    5: "Australia",
    6: "Austria"
}
Area_encoded = st.selectbox("Location", options=list(area_options.keys()), 
                            format_func=lambda x: area_options[x])

item_options = {
    1: "Maize",
    2: "Potatoes",
    3: "Sorghum",
    4: "Soybeans",
    6: "Wheat"
}
Item_encoded = st.selectbox("Crop Type", options=list(item_options.keys()), 
                            format_func=lambda x: item_options[x])

features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area_encoded, Item_encoded]])
features_scaled = scaler.transform(features)
prediction = rf_model.predict(features_scaled)


if st.button("Predict Yield"):
    prediction = rf_model.predict(features_scaled)
    st.success(f"🌱 Estimated Crop Yield: {prediction[0]:.2f}")
