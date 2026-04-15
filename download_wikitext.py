from datasets import load_dataset
import os

def download_wikitext():
    print("Downloading WikiText-103 dataset...")
    print("This may take a few minutes depending on your internet speed.")
    
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "wikitext_103_train.txt")
    print(f"Saving training data to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset["train"]:
            text = example["text"]
            if text.strip():
                f.write(text + "\n")
    
    val_file = os.path.join(output_dir, "wikitext_103_val.txt")
    print(f"Saving validation data to {val_file}...")
    with open(val_file, "w", encoding="utf-8") as f:
        for example in dataset["validation"]:
            text = example["text"]
            if text.strip():
                f.write(text + "\n")
    
    train_size = os.path.getsize(output_file) / (1024 * 1024)
    val_size = os.path.getsize(val_file) / (1024 * 1024)
    print(f"\nDataset ready!")
    print(f"Training file: {output_file} ({train_size:.1f} MB)")
    print(f"Validation file: {val_file} ({val_size:.1f} MB)")
    
    with open(output_file, "r", encoding="utf-8") as f:
        text = f.read()
        words = len(text.split())
        print(f"Approximate word count in training set: {words:,}")

if __name__ == "__main__":
    download_wikitext()
