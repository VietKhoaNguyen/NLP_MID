# Vietnamese Hate Speech Detection (PhoBERT + SVM)

This project implements a **Vietnamese Hate Speech Detection system** using two approaches:

* **PhoBERT** (Transformer-based deep learning model)
* **SVM** (Machine learning baseline using TF-IDF)

The system includes a **Streamlit web application** that allows users to input Vietnamese text and compare predictions from both models.

---

# Project Features

* Hate speech classification for Vietnamese text
* Comparison between **PhoBERT** and **SVM**
* Probability scores for predictions
* Interactive web interface using Streamlit

---

# Dataset

This project uses the **VN-HSD (Vietnamese Hate Speech Detection)** dataset from Hugging Face.

Dataset link:
https://huggingface.co/datasets/visolex/VN-HSD

Dataset statistics:

* Total samples: **40,532 comments**
* Columns:

  * `dataset`
  * `type`
  * `comment`
  * `label`

Label meanings:

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
│   ├── download_dataset.py
│   ├── train_phobert.py
│   └── train_svm.py
│
└── utils
```

---

# Installation

Clone the repository:

```
git clone https://github.com/VietKhoaNguyen/NLP_MID.git
cd NLP_MID
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Quick Start (Recommended)

Run the following commands:

```
python scripts/download_dataset.py
python scripts/train_svm.py
streamlit run app_streamlit.py
```

Open the web interface:

```
http://localhost:8501
```

---

# Download Dataset

To download the VN-HSD dataset automatically:

```
python scripts/download_dataset.py
```

The dataset will be saved to:

```
data/vn_hsd_dataset.csv
```

---

# Train Models

## Train SVM Model

```
python scripts/train_svm.py
```

The trained model will be saved to:

```
models/svm_model.pkl
```

---

## Train PhoBERT Model

```
python scripts/train_phobert.py
```

The trained model will be saved to:

```
models/phobert_model/
```

Note: PhoBERT training may require GPU and can take significant time.

---

# Run the Web Application

Start the Streamlit application:

```
streamlit run app_streamlit.py
```

Then open:

```
http://localhost:8501
```

You can enter Vietnamese text and see predictions from both **PhoBERT** and **SVM**.

---

# Example

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

# Technologies Used

* Python
* PyTorch
* Transformers
* Scikit-learn
* Streamlit
* Pandas

---

# Authors

USTH-Group 1 for Lec. Natural Language Processing:
- Nguyễn Việt Khoa - 23BI14223
- Nguyễn Phạm Trường An - 23BI14004
- Phạm Tường Ngạn - 23BI14334
- Nguyễn Công Phúc - 23BI14359
- Đỗ Minh Tiến - 23BI14421
- Phùng Đàm Tiến Sĩ - 23BI14384

---

# License

This project is developed for educational and research purposes.
