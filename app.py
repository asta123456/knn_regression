import streamlit as st
import numpy as np
import pickle

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Loan Prediction App")

st.write("Enter details to predict target value")

# Inputs
age = st.number_input("Age", min_value=18, max_value=70)
income = st.number_input("Income")
loan_amount = st.number_input("Loan Amount")
credit_score = st.number_input("Credit Score")

# Example encoding (IMPORTANT: match training)
city = st.selectbox("City", ["Chennai", "Hyderabad", "Bangalore", "Mumbai"])
employment = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed"])

# Convert categorical to one-hot
city_data = [0,0,0,0]
if city == "Bangalore": city_data[0]=1
elif city == "Chennai": city_data[1]=1
elif city == "Hyderabad": city_data[2]=1
elif city == "Mumbai": city_data[3]=1

emp_data = [0,0,0]
if employment == "Salaried": emp_data[0]=1
elif employment == "Self-Employed": emp_data[1]=1
elif employment == "Unemployed": emp_data[2]=1

# Combine input
input_data = np.array([[age, income, loan_amount, credit_score] + city_data + emp_data])

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    result = model.predict(input_scaled)
    st.success(f"Predicted Value: {result[0]}")