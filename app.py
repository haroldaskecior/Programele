import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder

# Load the trained Gradient Boosting model
model_path = './data/tuned_gradient_boosting_model.pkl'
model = joblib.load(model_path)

# Load the dataset to populate dropdowns
data_path = './data/df2.csv'
df = pd.read_csv(data_path)

# Initialize label encoders for all categorical features
encoders = {}
categorical_features = {
    'oem': 'Atomobilio markė',
    'model': 'Automobilio modelis',
    'City': 'AUtomobilio lokacija',
    'Engine Type': 'Variklio tipas',
    'Color': 'Automobilio Spalva',
    'Tyre Type': 'Padangu tipas',
    'Transmission': 'Transmisijos tipas'
}

# Fit encoders on the dataset
for feature, display_name in categorical_features.items():
    if feature in df.columns:
        encoder = LabelEncoder()
        df[feature] = df[feature].astype(str).str.strip()  # Standardize
        df[feature] = encoder.fit_transform(df[feature])
        encoders[feature] = encoder

# Title of the application
st.title("AUTOMOBILIŲ KAINOS PROGNOZAVIMO ĮRANKIS NAUDOJANTIS MAŠININĮ MOKYMĄ")

# Input Data Collection
input_values = []

# Step 1: Select OEM (Make)
selected_oem = st.selectbox(categorical_features['oem'], encoders['oem'].classes_)
encoded_oem = encoders['oem'].transform([selected_oem])[0]
input_values.append(encoded_oem)

# Step 2: Select Model (Filtered by OEM)
filtered_models = df[df['oem'] == encoded_oem]
model_classes = encoders['model'].inverse_transform(filtered_models['model'].unique())
selected_model = st.selectbox(categorical_features['model'], model_classes)
encoded_model = encoders['model'].transform([selected_model])[0]
input_values.append(encoded_model)

# Step 3: Select City
if st.checkbox(categorical_features['City']):
    selected_city = st.selectbox(categorical_features['City'], encoders['City'].classes_)
    encoded_city = encoders['City'].transform([selected_city])[0]
else:
    # Use the most frequent value as the default
    default_city = df['City'].mode()[0]
    encoded_city = encoders['City'].transform([default_city])[0]
input_values.append(encoded_city)

# Step 4: Select Engine Type
if st.checkbox(f"{categorical_features['Engine Type']}"):
    selected_engine = st.selectbox(categorical_features['Engine Type'], encoders['Engine Type'].classes_)
    encoded_engine = encoders['Engine Type'].transform([selected_engine])[0]
else:
    # Use the most frequent value as the default
    default_engine = df['Engine Type'].mode()[0]
    encoded_engine = encoders['Engine Type'].transform([default_engine])[0]
input_values.append(encoded_engine)

# Step 5: Select Other Features
numeric_features = {
    'Wheel Size': 'Ratų dydis',
    'modelYear': 'Pagaminimo metai',
    'Width': 'Plotis (mm)',
    'Max Power': 'Automobilio galia (hp)',
    'Length': 'Ilgis (mm)',
    'Wheel Base': 'Ratų bazė (mm)',
    'km': 'Kilometražas (km)',
    'Acceleration': 'Akseleracija/pagreitėjimas (0-100 km/h)',
    'Kerb Weight': 'Originalus automobilio svoris (kg)',
    'Torque': 'Sukimo momentas (Nm)',
    'Cargo Volumn': 'Automobilio talpa (L)',
    'Mileage': 'Kuro sąnaudos (km/l)',
    'Height': 'Aukštis (mm)',
    'Gear Box': 'Pavarų dežė',
    'Gross Weight': 'Maksimalus svoris (kg)'
}

for feature, display_name in numeric_features.items():
    if feature in df.columns:
        if st.checkbox(f"{display_name}"):
            selected_value = st.selectbox(f"{display_name}", sorted(df[feature].unique()))
        else:
            # Use the median value as the default
            selected_value = df[feature].median()
        input_values.append(selected_value)
    else:
        st.warning(f"{feature} not found in dataset. Defaulting to 0.")
        input_values.append(0)

# Add placeholders for excluded features to match model input size
excluded_features_defaults = [0] * 10  # Placeholder for excluded features
input_values.extend(excluded_features_defaults)

# Convert input values to numpy array
input_data = np.array([input_values])

# Function to fetch exchange rate
def get_exchange_rate(from_currency="INR", to_currency="EUR"):
    api_url = "https://v6.exchangerate-api.com/v6/c5186f953a752d1b79ef7e7f/latest/INR"
    response = requests.get(api_url)
    if response.status_code == 200:
        rates = response.json().get("conversion_rates", {})
        return rates.get(to_currency, None)
    else:
        st.error("Failed to fetch exchange rates.")
        return None

# Predict button
if st.button('Predict Price'):
    try:
        # Make the prediction
        predicted_price = model.predict(input_data)[0]
        
        # Fetch exchange rate and convert to EUR
        exchange_rate = get_exchange_rate()
        if exchange_rate:
            converted_price = predicted_price * exchange_rate
            st.success(f"The estimated car price is: ₹{predicted_price:,.2f} (≈ €{converted_price:,.2f})")
        else:
            st.success(f"The estimated car price is: ₹{predicted_price:,.2f}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

