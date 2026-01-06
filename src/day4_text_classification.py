import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data/cleaned_medical_transcriptions.csv"
TEXT_COL = "cleaned_transcription"
LABEL_COL = "medical_specialty"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)
print(f"✅ Cleaned data loaded: {df.shape}")

df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
df[LABEL_COL] = df[LABEL_COL].fillna("Unknown")

# -----------------------------
# BROAD CATEGORY MAPPING
# -----------------------------
category_map = {
    "Surgery": "Surgical",
    "Orthopedic": "Surgical",
    "Neurosurgery": "Surgical",
    "Obstetrics / Gynecology": "Surgical",
    "Urology": "Surgical",

    "General Medicine": "Medicine",
    "Gastroenterology": "Medicine",
    "Neurology": "Medicine",
    "Cardiovascular / Pulmonary": "Medicine",
    "Endocrinology": "Medicine",
    "Hematology - Oncology": "Medicine",
    "Nephrology": "Medicine",

    "Radiology": "Diagnostics",
    "Lab Medicine - Pathology": "Diagnostics",

    "Emergency Room Reports": "Emergency",

    "Consult - History and Phy.": "Notes",
    "SOAP / Chart / Progress Notes": "Notes",
    "Discharge Summary": "Notes",
    "Office Notes": "Notes",
}

df["broad_category"] = df[LABEL_COL].map(category_map).fillna("Other")

# -----------------------------
# SAFETY CHECK
# -----------------------------
if df["broad_category"].nunique() < 2:
    print("⚠ Broad category collapsed to single class. Using original specialties.")
    y = df[LABEL_COL]
else:
    print("✅ Broad category mapping applied successfully")
    y = df["broad_category"]

X = df[TEXT_COL]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Train-test split completed")
print("Training label distribution:")
print(y_train.value_counts())

# -----------------------------
# TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("✅ TF-IDF vectorization completed")

# -----------------------------
# LABEL ENCODING
# -----------------------------
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)

# -----------------------------
# MODEL
# -----------------------------
model = LogisticRegression(
    max_iter=300,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train_enc)
print("✅ Model training completed")

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test_enc, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy:.4f}\n")

print("📊 Classification Report:")
print(classification_report(
    y_test_enc,
    y_pred,
    target_names=encoder.classes_
))

# -----------------------------
# SAVE
# -----------------------------
joblib.dump(model, f"{MODEL_DIR}/medical_text_model.pkl")
joblib.dump(encoder, f"{MODEL_DIR}/label_encoder.pkl")
joblib.dump(vectorizer, f"{MODEL_DIR}/tfidf_vectorizer.pkl")

print("\n✅ Model, encoder & vectorizer saved successfully")
print(f"📁 Saved in: {MODEL_DIR}")
