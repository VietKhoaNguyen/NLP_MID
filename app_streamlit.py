import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Config
st.set_page_config(
    page_title="Vietnamese Hate Speech Detection",
    layout="wide"
)

labels = ["CLEAN", "OFFENSIVE", "HATE"]

# Load Models
@st.cache_resource
def load_svm():
    model = joblib.load("models/svm_model.pkl")
    return model

@st.cache_resource
def load_phobert():
    tokenizer = AutoTokenizer.from_pretrained("models/phobert_model")
    model = AutoModelForSequenceClassification.from_pretrained("models/phobert_model")
    model.eval()
    return tokenizer, model

svm = load_svm()
tokenizer, phobert = load_phobert()

# Prediction Functions
def predict_svm(text):
    probs = svm.predict_proba([text])[0]
    pred = np.argmax(probs)
    return pred, probs

def predict_phobert(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = phobert(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).numpy()[0]
    pred = np.argmax(probs)
    return pred, probs

# Chart function
def plot_probs(probs):
    fig = plt.figure()
    plt.bar(labels, probs)
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    plt.title("Confidence")
    return fig

# UI
st.title("Vietnamese Hate Speech Detection")
st.write("Compare predictions between SVM and PhoBERT models.")
text = st.text_area("Enter Vietnamese text", height=150)
if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text.")
        st.stop()

    svm_pred, svm_probs = predict_svm(text)
    
    pho_pred, pho_probs = predict_phobert(text)

    col1, col2 = st.columns(2)

    # SVM
    with col1:
        st.header("SVM Model")
        st.success(f"Prediction: {labels[svm_pred]}")
        st.write("Confidence")

        for i, label in enumerate(labels):
            st.write(f"{label}: {svm_probs[i]:.4f}")

        fig = plot_probs(svm_probs)
        st.pyplot(fig)

    # PhoBERT
    with col2:
        st.header("PhoBERT Model")
        st.success(f"Prediction: {labels[pho_pred]}")
        st.write("Confidence")

        for i, label in enumerate(labels):
            st.write(f"{label}: {pho_probs[i]:.4f}")

        fig = plot_probs(pho_probs)
        st.pyplot(fig)

st.divider()
st.caption(
    "Models: TF-IDF + SVM vs PhoBERT Transformer for Vietnamese hate speech detection."
)