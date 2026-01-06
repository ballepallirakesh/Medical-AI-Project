import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------
# Config
# -----------------------------
EMBEDDINGS_PATH = "artifacts/biobert/biobert_embeddings.npy"
DATA_PATH = "data/cleaned_medical_transcriptions_balanced.csv"
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load data
# -----------------------------
print("🔄 Loading data...")
X = np.load(EMBEDDINGS_PATH)
df = pd.read_csv(DATA_PATH)

y = df["medical_specialty"].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -----------------------------
# Compute class weights
# -----------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print("⚖️ Class weights applied")

# -----------------------------
# Torch datasets
# -----------------------------
train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long),
)

test_ds = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long),
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# -----------------------------
# Bi-LSTM Model
# -----------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat((h_n[0], h_n[1]), dim=1)
        return self.fc(h)

model = BiLSTMClassifier(
    input_dim=768,
    hidden_dim=256,
    num_classes=len(np.unique(y_encoded)),
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Training loop
# -----------------------------
print("🚀 Training weighted Bi-LSTM...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# -----------------------------
# Evaluation
# -----------------------------
model.eval()
preds, trues = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        outputs = model(xb)
        preds.extend(outputs.argmax(dim=1).cpu().numpy())
        trues.extend(yb.numpy())

acc = accuracy_score(trues, preds)
print(f"\n🎯 Accuracy: {acc:.3f}\n")

print("📊 Classification Report:\n")
print(classification_report(trues, preds, target_names=encoder.classes_))

# -----------------------------
# Save artifacts
# -----------------------------
torch.save(model.state_dict(), "models/biobert_bilstm_weighted.pt")
np.save("models/biobert_classes.npy", encoder.classes_)

print("✅ Weighted Bi-LSTM model saved")
