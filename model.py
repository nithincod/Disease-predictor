# model.py
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_train():
    df = pd.read_csv("data/symptom_disease.csv")
    df["cleaned_text"] = df["symptom"].apply(clean_text)
    df = df[df["cleaned_text"].str.strip() != ""]

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
    X = vectorizer.fit_transform(df["cleaned_text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.2f}")

    return model, vectorizer, clean_text