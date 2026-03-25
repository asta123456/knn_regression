import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Prediction App")

# Example inputs (modify based on your dataset)
f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")
f3 = st.number_input("Feature 3")
f4 = st.number_input("Feature 4")

if st.button("Predict"):
    
    # Convert to correct format
    input_data = np.array([[f1, f2, f3, f4]])
    
    st.write("Input shape:", input_data.shape)  # debug
    
    # Scale
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    
    st.success(f"Prediction: {prediction[0]}")
