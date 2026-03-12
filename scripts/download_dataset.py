from datasets import load_dataset
import pandas as pd
from pathlib import Path

dataset = load_dataset("visolex/VN-HSD")

df = dataset["train"].to_pandas()

print(df.head())

# project root
root = Path(__file__).resolve().parent.parent

data_path = root / "data"
data_path.mkdir(exist_ok=True)

file_path = data_path / "vn_hsd_dataset.csv"

df.to_csv(file_path, index=False)

print("Saved to:", file_path)