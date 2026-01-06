import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ===============================
# PATHS (FIXED)
# ===============================
DATA_PATH = "data/cleaned_medical_transcriptions.csv"
OUTPUT_DIR = "artifacts/biobert"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "biobert_embeddings.npy")

# ===============================
# CHECK FILE EXISTS
# ===============================
if not os.path.exists(DATA_PATH):
    print("❌ Dataset not found at:", DATA_PATH)
    print("📂 Available files in data/:")
    for root, _, files in os.walk("data"):
        for f in files:
            print(" -", os.path.join(root, f))
    raise FileNotFoundError("Dataset file missing. Fix path above.")

# ===============================
# LOAD DATA
# ===============================
print("🔄 Loading cleaned data...")
df = pd.read_csv(DATA_PATH)

print("📌 Available columns:", df.columns.tolist())

# ===============================
# AUTO-DETECT TEXT COLUMN
# ===============================
PREFERRED_COLS = [
    "cleaned_text",
    "text",
    "transcription",
    "note",
    "medical_text"
]

TEXT_COL = None
for col in PREFERRED_COLS:
    if col in df.columns:
        TEXT_COL = col
        break

if TEXT_COL is None:
    raise ValueError(
        f"❌ No valid text column found. Columns: {df.columns.tolist()}"
    )

print(f"✅ Using text column: {TEXT_COL}")

texts = df[TEXT_COL].fillna("").astype(str).tolist()

# ===============================
# LOAD BioBERT
# ===============================
print("🧬 Loading BioBERT model...")
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ===============================
# EMBEDDING FUNCTION
# ===============================
def get_biobert_embedding(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # CLS token embedding
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()

# ===============================
# GENERATE EMBEDDINGS
# ===============================
print("⚙ Generating BioBERT embeddings...")
embeddings = []

for text in tqdm(texts):
    embeddings.append(get_biobert_embedding(text))

embeddings = np.array(embeddings)

# ===============================
# SAVE OUTPUT
# ===============================
np.save(EMBEDDINGS_PATH, embeddings)

print("✅ BioBERT embeddings generated successfully!")
print("📐 Shape:", embeddings.shape)
print("📦 Saved to:", EMBEDDINGS_PATH)
