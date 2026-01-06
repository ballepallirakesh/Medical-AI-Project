import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# -------------------------------
# Load dataset
# -------------------------------
DATA_PATH = "data/cleaned_medical_transcriptions.csv"
df = pd.read_csv(DATA_PATH)

TEXT_COL = "transcription" if "transcription" in df.columns else "cleaned_transcription"
documents = df[TEXT_COL].fillna("").astype(str).tolist()

print("✅ Loaded medical documents:", len(documents))

# -------------------------------
# Build Retriever (TF-IDF)
# -------------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)
doc_vectors = vectorizer.fit_transform(documents)

print("✅ Document index created")

# Save artifacts
with open("artifacts/rag/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("artifacts/rag/document_store.pkl", "wb") as f:
    pickle.dump(documents, f)

# -------------------------------
# Load Generator (T5)
# -------------------------------
print("🔄 Loading T5 generator...")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# -------------------------------
# RAG Inference Function
# -------------------------------
def rag_answer(query, top_k=3):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, doc_vectors)[0]

    top_indices = scores.argsort()[-top_k:][::-1]
    context = " ".join([documents[i][:500] for i in top_indices])

    prompt = f"question: {query} context: {context}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=128)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer, top_indices

# -------------------------------
# Demo
# -------------------------------
if __name__ == "__main__":
    query = "What does chest pain with elevated cardiac enzymes indicate?"
    answer, evidence_ids = rag_answer(query)

    print("\n🧠 Question:", query)
    print("\n📄 Retrieved Evidence IDs:", evidence_ids)
    print("\n💡 Generated Answer:")
    print(answer)
