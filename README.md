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
[https://huggingface.co/datasets/visolex/VN-HSD](https://huggingface.co/datasets/visolex/VN-HSD)

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
├── train_phobert_colab.ipynb
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

Since this project is distributed as a `.zip` archive, follow these steps to set it up:

1. Extract the downloaded archive (e.g., `NLP_MID.zip`) to your desired location.
2. Open your terminal or command prompt and navigate to the extracted folder:

```
cd NLP_MID
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

---

# Quick Start (Recommended)

Run the following commands to download the data, train the baseline SVM model, and start the app:

```
python scripts/download_dataset.py
python scripts/train_svm.py
streamlit run app_streamlit.py
```

Open the web interface at:
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

Training PhoBERT requires a GPU and can take significant time. We provide multiple ways to train this model depending on your environment:

### Option 1: Train via Google Colab (Recommended)

This is the fastest way if your local machine does not have a dedicated GPU. You can either use our direct link or upload the included Jupyter Notebook file.

* **Direct Link:** [Vietnamese Hate Speech Detection Notebook](https://colab.research.google.com/drive/1-EAIOTB-wxiuUh2bY1PnQ1V-EcidoHsi?usp=sharing)
* **Upload Notebook:** Alternatively, you can upload the included `train_phobert_colab.ipynb` file to your Google Drive and open it with Google Colab.

**Steps:**
1. Open the notebook via the link or your uploaded file.
2. Go to **Runtime > Change runtime type** and select **T4 GPU** (or any available GPU).
3. Run all the cells in the notebook to train the model.
4. Once training is complete, the notebook will generate the model files. Download the `phobert_model` folder.
5. Extract (if zipped) and place the downloaded `phobert_model` folder inside the `models/` directory of your local project.

### Option 2: Train via Jupyter Notebook (Local/Kaggle)

If you use VS Code, JupyterLab, or platforms like Kaggle, you can directly open and run the included `train_phobert_colab.ipynb` file. Make sure your environment has GPU support enabled for faster training.

### Option 3: Train via Python Script (Local)

If you have a local GPU setup and prefer running Python scripts, you can run the training script directly from your terminal:

```
python scripts/train_phobert.py
```

The trained model will be saved to:
```
models/phobert_model/
```

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
- Phùng Đàm Tiến Sĩ - 23BI14383

---

# License

This project is developed for educational and research purposes.
