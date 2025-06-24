# model.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import nltk
nltk.download('stopwords')

# Initialize preprocessing tools
ss = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Paste your data_set here (keep it at the top for clarity)
data_set = [
    "I have a fever and cough cold",
"My stomach hurts and I feel nauseous digestive",
"I twisted my ankle and it's swollen injury",
"I have a constant headache headache",
"I feel shortness of breath and chest pain emergency",
"I have a sore throat and difficulty swallowing sore_throat",
"I have a runny nose and sneezing cold",
"I am experiencing joint pain and fatigue arthritis",
"I accidentally cut my finger and it's bleeding injury",
"I have a rash on my skin and it's itchy skin_condition",
"I feel lightheaded and dizzy dizziness",
"I have trouble sleeping and always feel tired insomnia",
"I have a sprained wrist and it's painful injury",
"I am experiencing memory loss and confusion memory_loss",
"I have a bee sting and it's swollen insect_sting",
"I have a constant ringing in my ears tinnitus",
"I have a toothache and it's unbearable toothache",
"I have a burning sensation during urination urinary_tract_infection",
"I have difficulty focusing and staying alert attention_deficit",
"I have a stiff neck and shoulder pain neck_pain",
"I have a sunburn and my skin is red sunburn",
"I have frequent heartburn and acid reflux heartburn",
"I have a constant urge to urinate urinary_incontinence",
"I have a mole that changed color and shape skin_cancer",
"I have a sprained ankle and it's swollen injury",
"I have a sinus infection and nasal congestion sinus_infection",
"I have difficulty swallowing and chest pain gastroesophageal_reflux",
"I have a bruise on my leg and it's painful bruise",
"I have a cut on my hand and it needs stitches injury",
"I have a constant cough and phlegm chronic_cough",
"I have a bee sting and it's painful insect_sting",
"I have a swollen gland in my neck swollen_gland",
"I have a headache and sensitivity to light migraine",
"I have a sprained knee and it's swollen injury",
"I have a broken arm and it's painful broken_bone",
"I have a burn on my finger and it's blistering burn",
"I have a lump in my breast breast_lump",
"I have a fever and body aches flu",
"I have a sprained wrist and it's swollen injury",
"I have a constant back pain back_pain",
"I have a bee sting and it's swollen insect_sting",
"I have a constant stomach pain and bloating irritable_bowel_syndrome",
"I have a sore throat and fever streptococcal_infection",
"I have a sprained ankle and it's painful injury",
"I have a rash on my face and it's spreading rash",
"I have a constant headache and dizziness migraine",
"I have a bee sting and it's swollen insect_sting",
"I have a sprained wrist and it's painful injury",
"I have a sore throat and difficulty swallowing sore_throat",
"I have a runny nose and sneezing cold",
"I am experiencing joint pain and fatigue arthritis",
"I have a constant cough and shortness of breath chronic_obstructive_pulmonary_disease",
"I have a sunburn and my skin is peeling sunburn",
"I have a toothache and it's unbearable toothache",
"I have a rash on my arms and legs dermatitis",
"I have a constant urge to urinate and pain urinary_tract_infection",
"I have a mole that changed color and shape skin_cancer",
"I have a sprained ankle and it's swollen injury",
"I have a sinus infection and facial pain sinusitis",
"I have difficulty swallowing and chest pain gastroesophageal_reflux",
"I have a bruise on my leg and it's painful bruise",
"I have a cut on my hand and it needs stitches injury",
"I have a constant cough and phlegm chronic_cough",
"I have a bee sting and it's painful insect_sting",
"I have a swollen gland in my neck swollen_gland",
"I have a headache and sensitivity to light migraine",
"I have a sprained knee and it's swollen injury",
"I have a broken arm and it's painful broken_bone",
"I have a burn on my finger and it's blistering burn",
"I have a lump in my breast breast_lump",
"I have a fever and body aches flu",
"I have a sprained wrist and it's swollen injury",
"I have a constant back pain back_pain",
"I have a bee sting and it's swollen insect_sting",
"I have a constant stomach pain and bloating irritable_bowel_syndrome",
"I have a sore throat and fever streptococcal_infection",
"I have a sprained ankle and it's painful injury",
"I have a rash on my face and it's spreading rash",
"I have a constant headache and dizziness migraine",
"I have a bee sting and it's swollen insect_sting",
"I have a sprained wrist and it's painful injury",
"I have a sore throat and difficulty swallowing sore_throat",
"I have a runny nose and sneezing cold",
"I am experiencing joint pain and fatigue arthritis",
"I have a constant cough and shortness of breath chronic_obstructive_pulmonary_disease",
"I have a sunburn and my skin is peeling sunburn",
    "I have a stabbing pain in my abdomen and blood in my stool gastrointestinal_bleeding",
    "I have a persistent, itchy rash on my palms and soles eczema",
    "I have a sudden, severe pain in my lower back and side kidney_stones",
    "I have a constant feeling of pressure behind my eyes sinus_pressure",
    "I have a sudden, intense pain in my chest and difficulty breathing heart_attack",
    "I have a persistent cough with wheezing asthma",
    "I have a sudden, sharp pain in my side and blood in my urine kidney_stones",
    "I have a constant, burning pain in my upper abdomen acid_reflux",
    "I have a sudden, severe headache and confusion stroke",
    "I have a persistent, itchy rash with blisters on my hands contact_dermatitis",
    "I have a sudden, severe pain in my abdomen and vomiting appendicitis",
    "I have a constant feeling of discomfort and bloating after eating irritable_bowel_syndrome",
    "I have a sudden, sharp pain in my chest and pain radiating down my arm heart_attack",
    "I have a persistent, itchy rash on my scalp psoriasis",
    "I have a sudden, severe pain in my side and blood in my urine kidney_stones",
    "I have a constant, dull ache in my lower back and pain radiating down my leg sciatica",
    "I have a sudden, intense pain in my abdomen and dark, tarry stools gastrointestinal_bleeding",
    "I have a persistent, itchy rash on my groin and inner thighs fungal_infection",
    "I have a sudden, severe pain in my chest and difficulty swallowing heart_attack",
    "I have a constant, burning sensation during urination urinary_tract_infection",
    "I have a sudden, sharp pain in my chest and difficulty breathing heart_attack",
    "I have a persistent, itchy rash on my feet and between toes athlete's_foot",
    "I have a sudden, severe pain in my abdomen and high fever appendicitis",
    "I have a constant, sharp pain in my chest and difficulty breathing heart_attack",
    "I have a sudden, intense pain in my abdomen and bloating gastritis",
    "I have a persistent, dry cough and chest pain pneumonia",
    "I have a sudden, sharp pain in my chest and difficulty breathing heart_attack",
    "I have a constant, burning pain in my upper abdomen and nausea gastritis",
    "I have a sudden, intense pain in my abdomen and blood in my vomit gastrointestinal_bleeding",
    "I have a persistent, itchy rash on my buttocks",
]

# Preprocess dataset
corpus = []
for text in data_set:
    text = text.lower()
    text = ''.join(e for e in text if e.isalpha() or e.isspace())
    words = [ss.stem(word) for word in text.split() if word not in stop_words]
    corpus.append(" ".join(words))

# Vectorize entire corpus once
vectorizer = CountVectorizer()
text_vectorized = vectorizer.fit_transform(corpus)

# The function to call for prediction
def get_disease_prediction_from_symptom(symptom_text):
    user_input = symptom_text.lower()
    user_input = ''.join(e for e in user_input if e.isalpha() or e.isspace())
    user_input = user_input.split()
    user_input = [ss.stem(word) for word in user_input if word not in stop_words]
    user_input = " ".join(user_input)

    user_input_vectorized = vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_input_vectorized, text_vectorized)
    most_similar_index = np.argmax(cosine_similarities)
    predicted_line = data_set[most_similar_index]
    predicted_disease = predicted_line.split()[-1]

    return predicted_disease
