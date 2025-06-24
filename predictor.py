# predictor.py

from model1 import get_disease_prediction_from_symptom

def predict_disease(symptom_text):
    return get_disease_prediction_from_symptom(symptom_text)
