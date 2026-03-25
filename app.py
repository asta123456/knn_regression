import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Prediction App")

st.markdown("### Enter details below:")

# 👉 Use meaningful names (change based on your project)
age = st.number_input("Age")
income = st.number_input("Income")
loan = st.number_input("Loan Amount")
credit_score = st.number_input("Credit Score")
dependents = st.number_input("Dependents")
experience = st.number_input("Experience (years)")

# Predict button
if st.button("Predict"):
    try:
        # Input
        input_data = np.array([[age, income, loan, credit_score, dependents, experience]])
        
        # Scale
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)

        # 🎯 CLEAN OUTPUT (no technical stuff)
        st.markdown("## 🎯 Result")
        
        # 👉 For regression
        st.success(f"Predicted Value: {round(prediction[0], 2)}")
        
        # 👉 For classification (optional)
        # if prediction[0] == 1:
        #     st.success("Approved ✅")
        # else:
        #     st.error("Not Approved ❌")

    except:
        st.error("Something went wrong. Please check your inputs.")
