import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("KNN Regression Prediction App")

st.write("Enter all feature values below:")

# 👉 CHANGE THESE BASED ON YOUR DATASET
# Example: If your model has 6 features, keep 6 inputs
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")
feature4 = st.number_input("Feature 4")
feature5 = st.number_input("Feature 5")
feature6 = st.number_input("Feature 6")

# Prediction button
if st.button("Predict"):
    try:
        # ✅ Correct 2D format (VERY IMPORTANT)
        input_data = np.array([[feature1, feature2, feature3, feature4, feature5, feature6]])
        
        # Debug info (you can remove later)
        st.write("Input shape:", input_data.shape)
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)
        
        # Output
        st.success(f"Prediction: {prediction[0]}")
    
    except Exception as e:
        st.error(f"Error: {e}")
