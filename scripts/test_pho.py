from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# load model
model_path = "models/phobert_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()

# label mapping
labels = ["CLEAN", "OFFENSIVE", "HATE"]

text = "Đúng là bọn mắt híp"

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
)

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
pred = torch.argmax(logits, dim=1).item()

print("Text:", text)
print("Prediction:", labels[pred])