import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# ---------------- CONFIG ----------------
DATA_PATH = "data/cleaned_medical_transcriptions_balanced.csv"
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
BATCH_SIZE = 8          # SAFE for 8GB RAM
EPOCHS = 4              # SAFE MODE
LR = 2e-5
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("artifacts/day9_focal", exist_ok=True)

# ---------------- FOCAL LOSS ----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none", weight=alpha)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ---------------- DATASET ----------------
class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ---------------- MODEL ----------------
class BioBERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)

        # 🔒 Freeze ALL layers first
        for param in self.bert.parameters():
            param.requires_grad = False

        # 🔓 SAFE MODE: unfreeze last 2 encoder layers
        for layer in self.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.fc(self.dropout(pooled))

# ---------------- LOAD DATA ----------------
print("🔄 Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("📌 Available columns:", df.columns.tolist())

# ✅ FIXED COLUMN (NO 'text' ERROR)
texts = df["cleaned_transcription"].astype(str).tolist()
labels = df["medical_specialty"].tolist()

le = LabelEncoder()
labels_enc = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels_enc,
    test_size=0.2,
    stratify=labels_enc,
    random_state=42
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_ds = MedicalDataset(X_train, y_train, tokenizer)
test_ds = MedicalDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ---------------- CLASS WEIGHTS ----------------
class_counts = np.bincount(y_train)
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights.to(DEVICE)

# ---------------- TRAIN ----------------
model = BioBERTClassifier(num_classes=len(le.classes_)).to(DEVICE)
criterion = FocalLoss(alpha=class_weights, gamma=2)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

print("🚀 Training BioBERT + Focal Loss (SAFE MODE)...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels_batch = batch["label"].to(DEVICE)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# ---------------- EVALUATION ----------------
model.eval()
preds, true = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels_batch = batch["label"].to(DEVICE)

        outputs = model(input_ids, attention_mask)
        preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        true.extend(labels_batch.cpu().numpy())

acc = accuracy_score(true, preds)
print(f"\n🎯 Accuracy: {acc:.3f}\n")

print("📊 Classification Report:\n")
print(classification_report(true, preds, target_names=le.classes_, zero_division=0))

# ---------------- SAVE ----------------
torch.save(model.state_dict(), "artifacts/day9_focal/biobert_focal_safe.pt")
np.save("artifacts/day9_focal/label_classes.npy", le.classes_)

print("\n✅ BioBERT + Focal Loss (SAFE MODE) model saved")
