import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_medical_transcriptions.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "tfidf_features.csv")

def main():
    # Load cleaned data
    df = pd.read_csv(DATA_PATH)
    print("✅ Cleaned dataset loaded")
    print("Dataset shape:", df.shape)

    # Combine important text columns
    df["combined_text"] = (
        df["transcription"].fillna("")
    )

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

    print("✅ TF-IDF vectorization completed")
    print("TF-IDF shape:", tfidf_matrix.shape)

    # Convert to DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    # Save features
    tfidf_df.to_csv(OUTPUT_PATH, index=False)

    print("✅ Features saved successfully")
    print("Saved at:", OUTPUT_PATH)

    # Show sample features
    print("\n🔹 Sample feature columns:")
    print(tfidf_df.columns[:20])

if __name__ == "__main__":
    main()
