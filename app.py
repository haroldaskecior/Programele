import streamlit as st
import pandas as pd
import pickle

# Load the model and data
def load_model():
    with open('.data/tuned_gradient_boosting_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
data = pd.read_csv('.data/df2.csv')

# Field titles customization
titles = {
    'First registration': 'Year of Make',
    'Engine Power KW': 'Engine Power (KW)',
    'CO2 emisija, g/km': 'CO2 Emissions (g/km)',
    'Fuel consuption out-of-city l/100 km': 'Fuel Consumption (Out of City)',
    'Fuel consuption combined l/100 km': 'Fuel Consumption (Combined)',
    'Fuel consuption Urban l/100 km': 'Fuel Consumption (Urban)',
    'Engine size in Cm3': 'Engine Size (Cm3)',
    'Battery capacity, kWh': 'Battery Capacity (kWh)',
    'Electric range': 'Electric Range'
}

# Streamlit App
st.title("Vehicle Price Prediction App")
st.write("Predict vehicle prices based on various features.")

# Dropdowns
make = st.selectbox('Select Make', data['Make'].unique())
filtered_models = data[data['Make'] == make]['Model'].unique()
model_choice = st.selectbox('Select Model', filtered_models)
year_of_make = st.selectbox(titles['First registration'], sorted(data['First registration'].unique()))
color = st.selectbox('Color', data['Color'].unique())
gearbox = st.selectbox('Gearbox', data['Gearbox'].unique())
fuel_type = st.selectbox('Fuel Type', data['Fuel type'].unique())
driven_wheels = st.selectbox('Driven Wheels', data['Driven wheels'].unique())
number_of_seats = st.selectbox('Number of Seats', sorted(data['Number of seats'].unique()))
damage = st.selectbox('Damage', data['Damage'].unique())
euro_standard = st.selectbox('Euro Standard', data['Euro standard'].unique())
pollution_tax = st.selectbox('Pollution Tax in Lithuania', data['Pollution tax in Lithuania'].unique())

# Date Input
mot_expiry = st.date_input('MOT Test Expiry')

# Numerical Inputs
engine_power = st.number_input(titles['Engine Power KW'], min_value=0)
co2_emissions = st.number_input(titles['CO2 emisija, g/km'], min_value=0)
mileage = st.number_input('Mileage', min_value=0)
fuel_out_city = st.number_input(titles['Fuel consuption out-of-city l/100 km'], min_value=0.0)
fuel_combined = st.number_input(titles['Fuel consuption combined l/100 km'], min_value=0.0)
fuel_urban = st.number_input(titles['Fuel consuption Urban l/100 km'], min_value=0.0)
engine_size = st.number_input(titles['Engine size in Cm3'], min_value=0)

# Conditional Fields
if "electricity" in fuel_type.lower():
    battery_capacity = st.number_input(titles['Battery capacity, kWh'], min_value=0.0)
    electric_range = st.number_input(titles['Electric range'], min_value=0.0)

# Checkbox Inputs
binary_features = [
    'Rear view camera', '360 degree camera', 'Front view camera',
    'Ventilated seats', 'Blind Spot Detection', 'Electrically adjustable steering wheel',
    'Touch screen', 'Automated Parking', 'Electric seats with memory', 'Pneumatic suspension',
    'Matrix headlights', 'Paddle shifters', 'HiFi audio system', 'Electric seats',
    'Apple CarPlay / Android Auto', 'Sunroof', 'ESP', 'Traction control system',
    'Reclining seats', 'Fog lights', 'Boot cover', 'Lane Departure Warning',
    'Dynamic cornering lights', 'Soft close doors', 'LED headlights',
    'Heated mirrors', 'Leather seats'
]

user_inputs = {}
for feature in binary_features:
    user_inputs[feature] = 1 if st.checkbox(feature) else 0

# Predict Button
if st.button('Predict Price'):
    # Prepare input data for prediction
    input_data = [
        year_of_make, engine_power, user_inputs['Rear view camera'], mileage,
        number_of_seats, user_inputs['Battery capacity, kWh'] if "electricity" in fuel_type.lower() else 0,
        co2_emissions, pollution_tax, make, driven_wheels, model_choice,
        user_inputs['Electrically adjustable steering wheel'], gearbox, user_inputs['360 degree camera'],
        user_inputs['Front view camera'], color, fuel_type, mot_expiry, damage,
        fuel_out_city, fuel_combined, fuel_urban, engine_size
    ] + [user_inputs[feature] for feature in binary_features]

    prediction = model.predict([input_data])[0]
    st.success(f"The predicted price of the vehicle is €{prediction:.2f}")
