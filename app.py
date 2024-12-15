import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and data
def load_model():
    try:
        model = joblib.load('programele\data\tuned_gradient_boosting_model.pkl')
        if model is None:
            raise ValueError("The loaded model is None.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
data = pd.read_csv(r'.\data\df2.csv')

# Encoding mappings (example; update based on training preprocessing)
make_mapping = {value: idx for idx, value in enumerate(data['Make'].unique())}
model_mapping = {value: idx for idx, value in enumerate(data['Model'].unique())}
body_type_mapping = {value: idx for idx, value in enumerate(data['Body type'].unique())}
gearbox_mapping = {value: idx for idx, value in enumerate(data['Gearbox'].unique())}
driven_wheels_mapping = {value: idx for idx, value in enumerate(data['Driven wheels'].unique())}
registration_country_mapping = {value: idx for idx, value in enumerate(data['First registration country'].unique())}
color_mapping = {value: idx for idx, value in enumerate(data['Color'].unique())}

# Feature names for the model
input_features = [
    'First registration', 'Engine Power KW', 'Mileage', 'Model',
    'CO2 emisija, g/km', 'Body type', 'Gearbox', 'Make',
    'Driven wheels', 'First registration country', 'LED headlights',
    'Number of seats', 'Fuel consuption out-of-city l/100 km',
    'Fuel consuption Urban l/100 km', 'Color'
]

# Streamlit App
st.title("Vehicle Price Prediction App")
st.write("Predict vehicle prices based on specific features.")

# Dropdowns
make = st.selectbox('Select Make', data['Make'].unique())
filtered_models = data[data['Make'] == make]['Model'].unique()
model_choice = st.selectbox('Select Model', filtered_models)
year_of_make = st.selectbox('First Registration Year', sorted(data['First registration'].unique()))
body_type = st.selectbox('Body Type', data['Body type'].unique())
gearbox = st.selectbox('Gearbox', data['Gearbox'].unique())
driven_wheels = st.selectbox('Driven Wheels', data['Driven wheels'].unique())
registration_country = st.selectbox('First Registration Country', data['First registration country'].unique())
color = st.selectbox('Color', data['Color'].unique())
number_of_seats = st.selectbox('Number of Seats', sorted(data['Number of seats'].unique()))

# Numerical Inputs
engine_power = st.number_input('Engine Power (KW)', min_value=0)
co2_emissions = st.number_input('CO2 Emissions (g/km)', min_value=0)
mileage = st.number_input('Mileage', min_value=0)
fuel_out_city = st.number_input('Fuel Consumption (Out of City, l/100 km)', min_value=0.0)
fuel_urban = st.number_input('Fuel Consumption (Urban, l/100 km)', min_value=0.0)

# LED Headlights
led_headlights = st.checkbox('LED Headlights')

# Predict Button
if st.button('Predict Price'):
    if model is not None:
        try:
            # Encode categorical fields
            encoded_make = make_mapping.get(make, -1)
            encoded_model = model_mapping.get(model_choice, -1)
            encoded_body_type = body_type_mapping.get(body_type, -1)
            encoded_gearbox = gearbox_mapping.get(gearbox, -1)
            encoded_driven_wheels = driven_wheels_mapping.get(driven_wheels, -1)
            encoded_registration_country = registration_country_mapping.get(registration_country, -1)
            encoded_color = color_mapping.get(color, -1)

            # Prepare input data with correct feature names
            input_data = pd.DataFrame([[
                year_of_make, engine_power, mileage, encoded_model, co2_emissions,
                encoded_body_type, encoded_gearbox, encoded_make, encoded_driven_wheels,
                encoded_registration_country, int(led_headlights), number_of_seats,
                fuel_out_city, fuel_urban, encoded_color
            ]], columns=input_features)

            # Perform prediction
            prediction = model.predict(input_data)[0]
            st.success(f"The predicted price of the vehicle is €{prediction:.2f}")

                        # Find top 3 most correlated URLs
            data['price_diff'] = np.abs(data['Price'] - prediction)
            correlated_ads = data.nsmallest(3, 'price_diff')[['URL', 'Price']]
            st.write("Top 3 similar ads:")
            st.table(correlated_ads)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.error("Model could not be loaded. Please check the model file.")
