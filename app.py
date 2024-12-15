import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Įkeliamas modelis ir duomenys
def load_model():
    try:
        model = joblib.load('data/tuned_gradient_boosting_model.pkl')  # Įkeliamas išsaugotas modelis
        if model is None:
            raise ValueError("The loaded model is None.")  # Tikrinama, ar modelis nėra tuščias
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")  # Klaidos pranešimas, jei nepavyksta įkelti modelio
        return None

model = load_model()
data = pd.read_csv('data/df2.csv')  # Įkeliamas duomenų rinkinys

# Kategorialių laukų kodavimo žemėlapiai (pagal mokymo duomenis)
make_mapping = {value: idx for idx, value in enumerate(data['Make'].unique())}  # Automobilio markė
model_mapping = {value: idx for idx, value in enumerate(data['Model'].unique())}  # Automobilio modelis
body_type_mapping = {value: idx for idx, value in enumerate(data['Body type'].unique())}  # Kėbulo tipas
gearbox_mapping = {value: idx for idx, value in enumerate(data['Gearbox'].unique())}  # Pavarų dėžė
driven_wheels_mapping = {value: idx for idx, value in enumerate(data['Driven wheels'].unique())}  # Varomieji ratai
registration_country_mapping = {value: idx for idx, value in enumerate(data['First registration country'].unique())}  # Registracijos šalis
color_mapping = {value: idx for idx, value in enumerate(data['Color'].unique())}  # Spalva

# Modelio laukų pavadinimai
input_features = [
    'First registration', 'Engine Power KW', 'Mileage', 'Model',
    'CO2 emisija, g/km', 'Body type', 'Gearbox', 'Make',
    'Driven wheels', 'First registration country', 'LED headlights',
    'Number of seats', 'Fuel consuption out-of-city l/100 km',
    'Fuel consuption Urban l/100 km', 'Color'
]

# Streamlit aplikacija
st.title("AUTOMOBILIŲ KAINOS PROGNOZAVIMO ĮRANKIS NAUDOJANTIS MAŠININĮ MOKYMĄ")  # Programėlės pavadinimas
st.write("Haroldas Kečioris PS1")  # Autoriaus vardas ir kodas

# Išskleidžiami sąrašai (dropdown)
make = st.selectbox('Select Make', data['Make'].unique())  # Automobilio markės pasirinkimas
filtered_models = data[data['Make'] == make]['Model'].unique()  # Modeliai, priklausantys pasirinktai markei
model_choice = st.selectbox('Select Model', filtered_models)  # Modelio pasirinkimas
year_of_make = st.selectbox('First Registration Year', sorted(data['First registration'].unique()))  # Registracijos metai
body_type = st.selectbox('Body Type', data['Body type'].unique())  # Kėbulo tipas
gearbox = st.selectbox('Gearbox', data['Gearbox'].unique())  # Pavarų dėžė
driven_wheels = st.selectbox('Driven Wheels', data['Driven wheels'].unique())  # Varomieji ratai
registration_country = st.selectbox('First Registration Country', data['First registration country'].unique())  # Registracijos šalis
color = st.selectbox('Color', data['Color'].unique())  # Spalvos pasirinkimas
number_of_seats = st.selectbox('Number of Seats', sorted(data['Number of seats'].unique()))  # Sėdimų vietų skaičius

# Skaitinių reikšmių įvedimas
engine_power = st.number_input('Engine Power (KW)', min_value=0)  # Variklio galia
co2_emissions = st.number_input('CO2 Emissions (g/km)', min_value=0)  # CO2 emisijos
total_mileage = st.number_input('Mileage', min_value=0)  # Rida
fuel_out_city = st.number_input('Fuel Consumption (Out of City, l/100 km)', min_value=0.0)  # Degalų sąnaudos užmiestyje
fuel_urban = st.number_input('Fuel Consumption (Urban, l/100 km)', min_value=0.0)  # Degalų sąnaudos mieste

# LED žibintų pasirinkimas
led_headlights = st.checkbox('LED Headlights')  # Pasirinkimas, ar yra LED žibintai

# Mygtukas kainos prognozavimui
if st.button('Predict Price'):
    if model is not None:
        try:
            # Kategorialių laukų kodavimas
            encoded_make = make_mapping.get(make, -1)  # Markės kodavimas
            encoded_model = model_mapping.get(model_choice, -1)  # Modelio kodavimas
            encoded_body_type = body_type_mapping.get(body_type, -1)  # Kėbulo tipo kodavimas
            encoded_gearbox = gearbox_mapping.get(gearbox, -1)  # Pavarų dėžės kodavimas
            encoded_driven_wheels = driven_wheels_mapping.get(driven_wheels, -1)  # Varomųjų ratų kodavimas
            encoded_registration_country = registration_country_mapping.get(registration_country, -1)  # Šalies kodavimas
            encoded_color = color_mapping.get(color, -1)  # Spalvos kodavimas

            # Paruošiami duomenys prognozavimui
            input_data = pd.DataFrame([[
                year_of_make, engine_power, total_mileage, encoded_model, co2_emissions,
                encoded_body_type, encoded_gearbox, encoded_make, encoded_driven_wheels,
                encoded_registration_country, int(led_headlights), number_of_seats,
                fuel_out_city, fuel_urban, encoded_color
            ]], columns=input_features)

            # Atliekama prognozė
            prediction = model.predict(input_data)[0]  # Modelio prognozė
            st.success(f"The predicted price of the vehicle is €{prediction:.2f}")  # Rodoma prognozuota kaina

            # Rodomi 3 panašiausi skelbimai pagal kainos skirtumą
            data['price_diff'] = np.abs(data['Price'] - prediction)  # Skaičiuojamas kainos skirtumas
            correlated_ads = data.nsmallest(3, 'price_diff')[['URL', 'Price']]  # Panašiausi skelbimai
            st.write("Top 3 similar ads:")  # Lentelės antraštė
            st.table(correlated_ads)  # Lentelė su URL ir kainomis

        except Exception as e:
            st.error(f"Prediction failed: {e}")  # Klaidos pranešimas, jei prognozė nepavyksta
    else:
        st.error("Model could not be loaded. Please check the model file.")  # Klaida, jei modelio nepavyksta įkelti
