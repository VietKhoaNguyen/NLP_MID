import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from utils.data_loader import load_dataset_vn_hsd


def train_svm():

    print("Loading dataset...")
    df = load_dataset_vn_hsd()

    print("Splitting dataset...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    y_train = train_df["label"]
    y_test = test_df["label"]

    print("Building TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2)
    )

    X_train = tfidf.fit_transform(train_df["comment"])
    X_test = tfidf.transform(test_df["comment"])

    print("Training SVM model...")
    svm = LinearSVC()
    svm.fit(X_train, y_train)

    preds = svm.predict(X_test)

    print("\n=== SVM RESULTS ===")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    os.makedirs("models", exist_ok=True)

    joblib.dump(svm, "models/svm_model.pkl")
    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

    print("\nModels saved to /models")


if __name__ == "__main__":
    train_svm()