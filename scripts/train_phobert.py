import os
import sys
# Fix OpenMP error (Windows + Anaconda)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.model_selection import train_test_split
from utils.data_loader import load_dataset_vn_hsd

# Load dataset
df = load_dataset_vn_hsd()
print("Columns:", df.columns)

# Rename column comment -> text
if "comment" in df.columns:
    df = df.rename(columns={"comment": "text"})

df = df[["text", "label"]]

print("Dataset size:", len(df))
print(df.head())

# Train / Test split
train_df, test_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42
)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load PhoBERT tokenizer
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

# Training config
training_args = TrainingArguments(
    output_dir="../models/phobert_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="../models/logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train
trainer.train()
trainer.save_model("../models/phobert_model")
tokenizer.save_pretrained("../models/phobert_model")

print("PhoBERT training finished!")