import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os

# ---------------- CONFIG ----------------
DATA_PATH = "data/cleaned_medical_transcriptions_balanced.csv"
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
TEXT_COL = "transcription"
LABEL_COL = "medical_specialty"
BATCH_SIZE = 4
EPOCHS = 4
LR = 2e-5
MAX_LEN = 128
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cpu")

# ---------------- DATASET ----------------
class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ---------------- MODEL ----------------
class BioBERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)

        # 🔒 Freeze ALL layers
        for param in self.bert.parameters():
            param.requires_grad = False

        # 🔓 Unfreeze LAST 2 encoder layers
        for layer in self.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

# ---------------- LOAD DATA ----------------
print("🔄 Loading dataset...")
df = pd.read_csv(DATA_PATH)
df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

le = LabelEncoder()
labels = le.fit_transform(df[LABEL_COL])

texts_train, texts_test, y_train, y_test = train_test_split(
    df[TEXT_COL].tolist(), labels, test_size=0.2, random_state=42, stratify=labels
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_ds = MedicalDataset(texts_train, y_train, tokenizer)
test_ds = MedicalDataset(texts_test, y_test, tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ---------------- CLASS WEIGHTS ----------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# ---------------- TRAIN ----------------
model = BioBERTClassifier(num_classes=len(le.classes_)).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

print("🚀 Fine-tuning BioBERT (SAFE MODE)...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels_batch = batch["label"].to(DEVICE)

        outputs = model(input_ids, mask)
        loss = criterion(outputs, labels_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# ---------------- EVALUATION ----------------
model.eval()
preds, true = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels_batch = batch["label"].to(DEVICE)

        outputs = model(input_ids, mask)
        preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        true.extend(labels_batch.cpu().numpy())

print("\n🎯 Accuracy:", accuracy_score(true, preds))
print("\n📊 Classification Report:\n")
print(classification_report(true, preds, target_names=le.classes_))

# ---------------- SAVE ----------------
torch.save(model.state_dict(), f"{MODEL_DIR}/biobert_finetuned_safe.pt")
joblib.dump(le, f"{MODEL_DIR}/biobert_finetuned_label_encoder.pkl")

print("\n✅ SAFE fine-tuned BioBERT model saved")
