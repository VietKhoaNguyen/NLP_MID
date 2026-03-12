from datasets import load_dataset

def load_dataset_vn_hsd():
    # Load dataset
    dataset = load_dataset("visolex/VN-HSD")
    full_ds = dataset["train"]

    # Remove null comments
    full_ds = full_ds.filter(lambda x: x["comment"] is not None)

    # Ensure comment is string
    def ensure_string(example):
        example["comment"] = str(example["comment"])
        return example

    full_ds = full_ds.map(ensure_string)

    # Convert to pandas
    df = full_ds.to_pandas()
    return df