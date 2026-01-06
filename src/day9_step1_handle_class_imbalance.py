import pandas as pd
from collections import Counter

# ===============================
# CONFIG
# ===============================
DATA_PATH = "data/cleaned_medical_transcriptions.csv"
OUTPUT_PATH = "data/cleaned_medical_transcriptions_balanced.csv"
MIN_SAMPLES = 10

# ===============================
# LOAD DATA
# ===============================
print("🔄 Loading dataset...")
df = pd.read_csv(DATA_PATH)

label_col = "medical_specialty"

print(f"📊 Total samples: {len(df)}")

# ===============================
# COUNT LABELS
# ===============================
label_counts = Counter(df[label_col])

print("\n📌 Class distribution (before):")
for label, count in label_counts.items():
    print(f"{label}: {count}")

# ===============================
# IDENTIFY RARE CLASSES
# ===============================
rare_classes = [label for label, count in label_counts.items() if count < MIN_SAMPLES]

print(f"\n⚠️ Rare classes (< {MIN_SAMPLES} samples): {len(rare_classes)}")
print(rare_classes)

# ===============================
# MERGE INTO 'Other'
# ===============================
df[label_col] = df[label_col].apply(
    lambda x: "Other" if x in rare_classes else x
)

# ===============================
# RECOUNT
# ===============================
new_counts = Counter(df[label_col])

print("\n✅ Class distribution (after):")
for label, count in new_counts.items():
    print(f"{label}: {count}")

print(f"\n🏷 Total classes reduced from {len(label_counts)} → {len(new_counts)}")

# ===============================
# SAVE CLEANED DATA
# ===============================
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n💾 Balanced dataset saved to: {OUTPUT_PATH}")
