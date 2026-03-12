import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# =========================
# Paths
# =========================
root = Path(__file__).resolve().parent.parent
data_path = root / "data" / "vn_hsd_dataset.csv"
model_dir = root / "models"

model_dir.mkdir(exist_ok=True)

# =========================
# Load dataset
# =========================
df = pd.read_csv(data_path)

df["text"] = df["comment"]
df = df[["text", "label"]]

# FIX NaN
df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str)

print("Dataset size:", len(df))
print(df.head())

# =========================
# Train test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.1,
    random_state=42
)

# =========================
# Pipeline
# =========================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=30000,
        ngram_range=(1,2)
    )),
    ("svm", SVC(
        kernel="linear",
        probability=True
    ))
])

print("Training SVM...")

pipeline.fit(X_train, y_train)

# =========================
# Evaluation
# =========================
preds = pipeline.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, preds))

# =========================
# Save model
# =========================
model_path = model_dir / "svm_model.pkl"

joblib.dump(pipeline, model_path)

print("\nModel saved to:", model_path)