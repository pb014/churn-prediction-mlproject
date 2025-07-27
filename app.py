# Gender -> male(0)   female(1)
# Churn -> yes(1)  no(0)
# scaler is exported as scaler.pkl
# model is exported as model.pkl
# order of x -> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction")

st.divider()

st.write("Please enter the values and Predict!")

st.divider()

age = st.number_input("Enter age", min_value=10, max_value=100, value=30)

tenure = st.number_input("Enter Tenure", min_value= 0, max_value=130, value=10)

monthlycharge = st.number_input("Enter Monthly charge", min_value=30, max_value=150)

gender = st.selectbox("Enter gender", ["Male", "Female"])

st.divider()

predicbutton = st.button("Predict!")

st.divider()

if predicbutton:
    gender_selected = 1 if gender == "Female" else 0
    X = [age, gender_selected, tenure, monthlycharge]

    X1 = np.array(X)

    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]

    result = "Yes" if prediction == 1 else "No"

    st.write(f"Predicted: {result}")

else:
    st.write("Please enter the values")