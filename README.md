# Vietnamese Hate Speech Detection

A machine learning project for detecting **Vietnamese hate speech** using both traditional Machine Learning and Transformer-based models.

This project demonstrates two approaches:

* **SVM + TF-IDF** (traditional ML baseline)
* **PhoBERT** (Transformer-based deep learning model)

The system also includes a **Streamlit web demo** to interactively test predictions.

---

# Models Used

### 1. SVM + TF-IDF

A classical machine learning pipeline using:

* TF-IDF text vectorization
* Support Vector Machine classifier

Advantages:

* Fast training
* Lightweight model
* Good baseline performance

---

### 2. PhoBERT

The transformer model **PhoBERT** developed by VinAI Research.

PhoBERT is based on the architecture of RoBERTa and is pre-trained specifically for Vietnamese NLP tasks.

Advantages:

* Better contextual understanding
* Higher accuracy on complex language patterns
* State-of-the-art Vietnamese language representation

---

# Dataset

The project uses the **Vietnamese Hate Speech Dataset (VN-HSD)**.

Labels:

| Label | Meaning   |
| ----- | --------- |
| 0     | CLEAN     |
| 1     | OFFENSIVE |
| 2     | HATE      |

Example:

| Text                             | Label |
| -------------------------------- | ----- |
| Em được làm fan cứng luôn rồi nè | CLEAN |
| Đúng là bọn mắt híp              | HATE  |

---

# Project Structure

```
sentiment_project
│
├── models
│   ├── svm_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── phobert_model
│
├── scripts
│   ├── train_svm.py
│   ├── train_phobert.py
│   └── test_predict.py
│
├── utils
│   └── data_loader.py
│
├── app_streamlit.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository from GitHub

```
git clone https://github.com/yourname/sentiment_project.git
cd sentiment_project
```

Create a Python environment (recommended)

Using Conda:

```
conda create -n sentiment python=3.10
conda activate sentiment
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Training the Models

### Train SVM Model

```
python scripts/train_svm.py
```

This will generate:

```
models/svm_model.pkl
models/tfidf_vectorizer.pkl
```

---

### Train PhoBERT Model

```
python scripts/train_phobert.py
```

The script will automatically download the PhoBERT weights from Hugging Face.

After training finishes, the model will be saved in:

```
models/phobert_model
```

---

# Running the Web Demo

The project includes an interactive web interface built with Streamlit.

Run:

```
streamlit run app_streamlit.py
```

Then open your browser at:

```
http://localhost:8501
```

You can:

* Enter Vietnamese text
* Select a model (SVM or PhoBERT)
* See the predicted label

---

# Example Usage

Input text:

```
Đúng là bọn mắt híp
```

Prediction:

```
HATE
```

---

# Requirements

Main libraries used:

* Python
* PyTorch
* Transformers
* Scikit-learn
* Streamlit
* Pandas
* HuggingFace Datasets

Install them using:

```
pip install -r requirements.txt
```

---

# Notes

If you see the warning:

```
pin_memory argument is set as true but no accelerator is found
```

This simply means the program is running on **CPU instead of GPU** and does **not affect training or inference**.

---

# Authors

Your Name

---

# License

This project is for **educational and research purposes**.
