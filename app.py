import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load the trained model and label encoders
model = load_model('savemodel.h5')  # Use the correct path to your model
label_encoder_diagnosis = joblib.load('label_encoder_diagnosis.pkl')  # Use the correct path
label_encoder_treatment = joblib.load('label_encoder_treatment.pkl')  # Use the correct path
scaler = joblib.load('scaler.pkl')  # Use the correct path

# Function to preprocess and make predictions
def predict_diagnosis_treatment(new_patient_data):
    preprocessed_data = scaler.transform(new_patient_data)
    predictions = model.predict(preprocessed_data)
    diagnosis_predictions = np.argmax(predictions[0], axis=1)
    treatment_predictions = np.argmax(predictions[1], axis=1)
    decoded_diagnosis = label_encoder_diagnosis.inverse_transform(diagnosis_predictions)
    decoded_treatment = label_encoder_treatment.inverse_transform(treatment_predictions)
    return decoded_diagnosis, decoded_treatment

# Streamlit app
st.title('Patient Diagnosis and Treatment Prediction')

# Input form
fever = st.number_input('Fever')
cough = st.radio('Cough', ['No', 'Yes'])
fatigue = st.radio('Fatigue', ['No', 'Yes'])
age = st.number_input('Age')
difficulty_breathing = st.radio('Difficulty Breathing', ['No', 'Yes'])
blood_pressure = st.number_input('Blood Pressure')
cholesterol_level = st.number_input('Cholesterol Level')

# Convert categorical inputs to numerical representation
cough = 1 if cough == 'Yes' else 0
fatigue = 1 if fatigue == 'Yes' else 0
difficulty_breathing = 1 if difficulty_breathing == 'Yes' else 0

# Make prediction button
if st.button('Predict'):
    new_patient_data = np.array([[fever, cough, fatigue, age, difficulty_breathing, blood_pressure, cholesterol_level]])
    predicted_diagnosis, predicted_treatment = predict_diagnosis_treatment(new_patient_data)
    st.success(f'Diagnosis Prediction: {predicted_diagnosis[0]}, Treatment Prediction: {predicted_treatment[0]}')
