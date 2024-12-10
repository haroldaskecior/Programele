import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained Gradient Boosting model
model_path = './data/tuned_gradient_boosting_model.pkl'
model = joblib.load(model_path)

# Load the dataset to populate dropdowns
data_path = './data/df2.csv'
df = pd.read_csv(data_path)

# Initialize label encoders for all categorical features
encoders = {}
categorical_features = ['oem', 'model', 'City', 'Engine Type', 'Color', 'Tyre Type', 'Transmission']

# Fit encoders on the dataset
for feature in categorical_features:
    if feature in df.columns:
        encoder = LabelEncoder()
        df[feature] = df[feature].astype(str).str.strip()  # Standardize
        df[feature] = encoder.fit_transform(df[feature])
        encoders[feature] = encoder

# Title of the application
st.title("Car Price Prediction with Selected Features")

# Input Data Collection
input_values = []

# Step 1: Select OEM (Make)
selected_oem = st.selectbox('Select Car OEM (Make)', encoders['oem'].classes_)
encoded_oem = encoders['oem'].transform([selected_oem])[0]
input_values.append(encoded_oem)

# Step 2: Select Model (Filtered by OEM)
filtered_models = df[df['oem'] == encoded_oem]
model_classes = encoders['model'].inverse_transform(filtered_models['model'].unique())
selected_model = st.selectbox('Select Car Model', model_classes)
encoded_model = encoders['model'].transform([selected_model])[0]
input_values.append(encoded_model)

# Step 3: Select City
selected_city = st.selectbox('Select City', encoders['City'].classes_)
encoded_city = encoders['City'].transform([selected_city])[0]
input_values.append(encoded_city)

# Step 4: Select Engine Type
selected_engine = st.selectbox('Select Engine Type', encoders['Engine Type'].classes_)
encoded_engine = encoders['Engine Type'].transform([selected_engine])[0]
input_values.append(encoded_engine)

# Step 5: Select Other Features
numeric_features = ['Wheel Size', 'modelYear', 'Width', 'Max Power', 'Length', 'Wheel Base', 'km',
                    'Acceleration', 'Kerb Weight', 'Torque', 'Cargo Volumn', 'Mileage', 'Height', 'Gear Box', 'Gross Weight']

for feature in numeric_features:
    if feature in df.columns:
        selected_value = st.selectbox(f'Select {feature}', sorted(df[feature].unique()))
        input_values.append(selected_value)
    else:
        st.warning(f"{feature} not found in dataset. Defaulting to 0.")
        input_values.append(0)

# Handle Categorical Features with Cleaned Dropdowns
if 'Transmission' in df.columns:
    transmission_classes = encoders['Transmission'].classes_
    selected_transmission = st.selectbox('Select Transmission', transmission_classes)
    encoded_transmission = encoders['Transmission'].transform([selected_transmission])[0]
    input_values.append(encoded_transmission)

# Add placeholders for excluded features to match model input size
excluded_features_defaults = [0] * 10  # Placeholder for Insurance Validity, Alloy Wheel Size, Turning Radius, Front Tread, Rear Tread, Displacement, and other excluded features
input_values.extend(excluded_features_defaults)

# Convert input values to numpy array
input_data = np.array([input_values])

# Predict button
if st.button('Predict Price'):
    try:
        # Make the prediction
        predicted_price = model.predict(input_data)[0]

        # Display the result
        st.success(f"The estimated car price is: ₹{predicted_price:,.2f}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
