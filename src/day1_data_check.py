import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "medical_transcriptions.csv")

df = pd.read_csv(data_path)

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nSample Medical Note:\n")
print(df["transcription"].iloc[0])


