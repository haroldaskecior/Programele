import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Įkelti modelį ir duomenis
def load_model():
    try:
        # Įkeliame išsaugotą mašininio mokymo modelį
        model = joblib.load('data/tuned_gradient_boosting_model.pkl')
        if model is None:
            raise ValueError("Įkeltas modelis yra tuščias (None).")
        return model
    except Exception as e:
        st.error(f"Klaida įkeliant modelį: {e}")
        return None

model = load_model()
# Įkeliame duomenis iš CSV failo
data = pd.read_csv('data/df2.csv')

# Kodavimo žemėlapiai (pagal duomenų paruošimą treniruotės metu)
make_mapping = {value: idx for idx, value in enumerate(data['Make'].unique())}
model_mapping = {value: idx for idx, value in enumerate(data['Model'].unique())}
body_type_mapping = {value: idx for idx, value in enumerate(data['Body type'].unique())}
gearbox_mapping = {value: idx for idx, value in enumerate(data['Gearbox'].unique())}
driven_wheels_mapping = {value: idx for idx, value in enumerate(data['Driven wheels'].unique())}
registration_country_mapping = {value: idx for idx, value in enumerate(data['First registration country'].unique())}
color_mapping = {value: idx for idx, value in enumerate(data['Color'].unique())}

# Modelio funkcijų pavadinimai
input_features = [
    'First registration', 'Engine Power KW', 'Mileage', 'Model',
    'CO2 emisija, g/km', 'Body type', 'Gearbox', 'Make',
    'Driven wheels', 'First registration country', 'LED headlights',
    'Number of seats', 'Fuel consuption out-of-city l/100 km',
    'Fuel consuption Urban l/100 km', 'Color'
]

# Streamlit aplikacija
st.title("AUTOMOBILIŲ KAINOS PROGNOZAVIMO ĮRANKIS NAUDOJANTIS MAŠININĮ MOKYMĄ")
st.write("Haroldas Kečioris PS1")

# Išskleidžiami meniu pasirinkimams
make = st.selectbox('Pasirinkite markę (Make)', data['Make'].unique())
filtered_models = data[data['Make'] == make]['Model'].unique()
model_choice = st.selectbox('Pasirinkite modelį (Model)', filtered_models)
year_of_make = st.selectbox('Pirmoji registracija (metai)', sorted(data['First registration'].unique()))
body_type = st.selectbox('Kėbulo tipas (Body Type)', data['Body type'].unique())
gearbox = st.selectbox('Pavarų dėžė (Gearbox)', data['Gearbox'].unique())
driven_wheels = st.selectbox('Varantieji ratai (Driven Wheels)', data['Driven wheels'].unique())
registration_country = st.selectbox('Pirmos registracijos šalis', data['First registration country'].unique())
color = st.selectbox('Spalva (Color)', data['Color'].unique())
number_of_seats = st.selectbox('Sėdynių skaičius (Number of Seats)', sorted(data['Number of seats'].unique()))

# Skaitinės įvestys
engine_power = st.number_input('Variklio galia (KW)', min_value=0)
co2_emissions = st.number_input('CO2 emisija (g/km)', min_value=0)
mileage = st.number_input('Rida (Mileage)', min_value=0)
fuel_out_city = st.number_input('Degalų sąnaudos užmiestyje (l/100 km)', min_value=0.0)
fuel_urban = st.number_input('Degalų sąnaudos mieste (l/100 km)', min_value=0.0)

# LED žibintai (žymimasis langelis)
led_headlights = st.checkbox('LED žibintai')

# Mygtukas kainos prognozavimui
if st.button('Prognozuoti kainą'):
    if model is not None:
        try:
            # Užkoduojame kategorinius laukus
            encoded_make = make_mapping.get(make, -1)
            encoded_model = model_mapping.get(model_choice, -1)
            encoded_body_type = body_type_mapping.get(body_type, -1)
            encoded_gearbox = gearbox_mapping.get(gearbox, -1)
            encoded_driven_wheels = driven_wheels_mapping.get(driven_wheels, -1)
            encoded_registration_country = registration_country_mapping.get(registration_country, -1)
            encoded_color = color_mapping.get(color, -1)

            # Paruošiame įvesties duomenis su funkcijų pavadinimais
            input_data = pd.DataFrame([[
                year_of_make, engine_power, mileage, encoded_model, co2_emissions,
                encoded_body_type, encoded_gearbox, encoded_make, encoded_driven_wheels,
                encoded_registration_country, int(led_headlights), number_of_seats,
                fuel_out_city, fuel_urban, encoded_color
            ]], columns=input_features)

            # Atliekame kainos prognozavimą
            prediction = model.predict(input_data)[0]
            st.success(f"Prognozuojama transporto priemonės kaina: €{prediction:.2f}")

            # Randame 3 panašiausius skelbimus
            data['price_diff'] = np.abs(data['Price'] - prediction)
            correlated_ads = data.nsmallest(3, 'price_diff')[['URL', 'Price']]
            st.write("3 panašiausi skelbimai:")
            st.table(correlated_ads)

        except Exception as e:
            st.error(f"Kainos prognozavimas nepavyko: {e}")
    else:
        st.error("Modelio nepavyko įkelti. Patikrinkite modelio failą.")
