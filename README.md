# Vietnamese Hate Speech Detection (PhoBERT + SVM)

A Natural Language Processing project for **Vietnamese Hate Speech Detection** using both:

* **PhoBERT (Transformer-based model)**
* **SVM (Machine Learning baseline)**

The system provides an interactive **Streamlit web interface** for comparing predictions between models.

---

# Project Overview

This project detects **hate speech in Vietnamese social media comments** using two approaches:

| Model   | Type             | Description                              |
| ------- | ---------------- | ---------------------------------------- |
| PhoBERT | Deep Learning    | Vietnamese pre-trained transformer model |
| SVM     | Machine Learning | TF-IDF + Support Vector Machine          |

The application allows users to:

* Enter a Vietnamese sentence
* Predict hate speech category
* Compare predictions from both models
* View probability scores

---

# Dataset

Dataset used:

**VN-HSD: Vietnamese Hate Speech Detection**

Source:
https://huggingface.co/datasets/visolex/VN-HSD

Dataset size:

```
40532 comments
```

Columns:

```
dataset
type
comment
label
```

Labels:

| Label | Meaning   |
| ----- | --------- |
| 0     | CLEAN     |
| 1     | OFFENSIVE |
| 2     | HATE      |

---

# Project Structure

```
sentiment_project
│
├── app_streamlit.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data
│   └── vn_hsd_dataset.csv
│
├── models
│   ├── svm_model.pkl
│   └── phobert_model
│       ├── config.json
│       ├── model.safetensors
│       ├── vocab.txt
│       └── tokenizer_config.json
│
├── scripts
│   ├── train_phobert.py
│   ├── train_svm.py
│   └── download_dataset.py
│
└── utils
    └── data_loader.py
```

---

# Installation

Clone repository

```
git clone https://github.com/VietKhoaNguyen/NLP_MID.git
cd NLP_MID
```

Install dependencies

```
pip install -r requirements.txt
```

---

# Download Dataset

Run:

```
python scripts/download_dataset.py
```

This will download the VN-HSD dataset and save it to:

```
data/vn_hsd_dataset.csv
```

---

# Train Models

## Train PhoBERT

```
python scripts/train_phobert.py
```

Output model will be saved to:

```
models/phobert_model/
```

---

## Train SVM

```
python scripts/train_svm.py
```

Output:

```
models/svm_model.pkl
```

---

# Run the Web Application

Launch the Streamlit interface:

```
streamlit run app_streamlit.py
```

Open in browser:

```
http://localhost:8501
```

You can now input Vietnamese text and compare predictions from:

* PhoBERT
* SVM

---

# Example Prediction

Input:

```
Đúng là bọn ngu dốt
```

Output:

```
PhoBERT Prediction: HATE
SVM Prediction: OFFENSIVE
```

---

# Model Comparison

| Model   | Advantages                                | Limitations               |
| ------- | ----------------------------------------- | ------------------------- |
| PhoBERT | Higher accuracy, contextual understanding | Requires GPU for training |
| SVM     | Fast training, simple                     | Lower performance         |

PhoBERT generally performs better on complex Vietnamese linguistic patterns.

---

# Technologies Used

* Python
* PyTorch
* HuggingFace Transformers
* Scikit-learn
* Streamlit
* Pandas

---

# Notes

PhoBERT model files can be large (>500MB).
If the model is not included in the repository, download or train it locally using:

```
python scripts/train_phobert.py
```

---

# Author

Nguyen Viet Khoa

---

# License

This project is for educational and research purposes.
