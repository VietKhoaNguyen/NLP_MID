import joblib
import os

# label mapping
label_names = ["CLEAN", "OFFENSIVE", "HATE"]

# load models
svm_model = joblib.load("models/svm_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


def predict_svm(text):

    vec = tfidf.transform([text])

    pred = svm_model.predict(vec)[0]

    return label_names[pred]