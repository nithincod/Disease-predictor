# app.py
import streamlit as st
from predictor import predict_disease

st.set_page_config(page_title="Disease Predictor", page_icon="🩺")
st.title("🩺 Symptom-based Disease Predictor")

symptoms = st.text_area("Enter symptoms (comma-separated):")

if st.button("Predict Disease"):
    if symptoms:
        prediction = predict_disease(symptoms)
        st.success(f"Predicted Disease: {prediction}")
    else:
        st.warning("Please enter symptoms.")