import joblib
import os
import sys

# ================================
# Paths
# ================================
MODEL_DIR = "models"

VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "medical_text_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# ================================
# Validate artifacts
# ================================
missing_files = []

if not os.path.exists(VECTORIZER_PATH):
    missing_files.append("tfidf_vectorizer.pkl")

if not os.path.exists(MODEL_PATH):
    missing_files.append("medical_text_model.pkl")

if not os.path.exists(LABEL_ENCODER_PATH):
    missing_files.append("label_encoder.pkl")

if missing_files:
    print("❌ Required trained files are missing:")
    for f in missing_files:
        print(f"   - {f}")

    print("\n👉 Reason:")
    print("Day-4 model training failed because the dataset contained only ONE class.")
    print("Machine learning classifiers require at least 2 classes.")

    print("\n✅ Next Step:")
    print("Fix category mapping OR use multi-class labels, then retrain Day-4.")
    sys.exit(1)

# ================================
# Load trained components
# ================================
print("🔄 Loading trained artifacts...")

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

print("✅ Model, vectorizer & label encoder loaded")

# ================================
# Prediction function
# ================================
def predict_specialty(text: str):
    text_vec = vectorizer.transform([text])
    pred_encoded = model.predict(text_vec)[0]
    return label_encoder.inverse_transform([pred_encoded])[0]

# ================================
# Demo
# ================================
if __name__ == "__main__":

    print("\n🩺 MEDICAL SPECIALTY PREDICTION DEMO\n")

    sample_note = (
        "Patient presents with chest pain, ECG changes, "
        "and elevated cardiac enzymes."
    )

    prediction = predict_specialty(sample_note)
    print("📝 Medical Note:")
    print(sample_note)
    print(f"\n🔮 Predicted Specialty: {prediction}")
