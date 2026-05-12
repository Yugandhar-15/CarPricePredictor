import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("🚗 Car Price Predictor")

st.write("Enter car details to predict price")

present_price = st.number_input("Present Price (in lakhs)", min_value=0.0)

kms_driven = st.number_input("Kilometers Driven", min_value=0)

owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])

seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])

transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

car_age = st.slider("Car Age", 0, 20)

fuel_dict = {"Petrol": 2, "Diesel": 1, "CNG": 0}
seller_dict = {"Dealer": 0, "Individual": 1}
transmission_dict = {"Manual": 1, "Automatic": 0}

fuel = fuel_dict[fuel_type]
seller = seller_dict[seller_type]
trans = transmission_dict[transmission]

if st.button("Predict Price"):

    features = np.array([[present_price,
                          kms_driven,
                          owner,
                          fuel,
                          seller,
                          trans,
                          car_age]])

    prediction = model.predict(features)

    st.success(f"Estimated Car Price: ₹ {prediction[0]:.2f} Lakhs")