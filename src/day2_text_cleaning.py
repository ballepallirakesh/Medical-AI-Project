import pandas as pd
import os
import re

# Base directory of project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input and output paths
input_path = os.path.join(BASE_DIR, "data", "medical_transcriptions.csv")
output_path = os.path.join(BASE_DIR, "data", "cleaned_medical_transcriptions.csv")

# Load dataset
df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()                      # lowercase
    text = re.sub(r'\n', ' ', text)               # remove newlines
    text = re.sub(r'[^a-z0-9., ]', ' ', text)     # keep useful characters
    text = re.sub(r'\s+', ' ', text).strip()      # remove extra spaces
    return text

# Apply cleaning
df["cleaned_transcription"] = df["transcription"].apply(clean_text)

# Save cleaned dataset
df.to_csv(output_path, index=False)

print("✅ Day-2 preprocessing completed")
print("Cleaned file saved at:", output_path)

print("\n🔹 Sample cleaned text:\n")
print(df["cleaned_transcription"].iloc[0])
