import os
import tiktoken
import numpy as np

def tokenize_and_save(input_path, output_path):
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print("Tokenizing... (This will take about 30-60 seconds for 500MB of text)")
    # GPT-2 tokenizer uses Byte-Pair Encoding (BPE)
    # It splits words into subword units: e.g., "tokenization" -> ["token", "ization"]
    # Vocab size is 50,257
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(text)
    print(f"Tokenization complete! Total tokens: {len(tokens):,}")
    
    # Save as uint16 to save space (GPT-2 vocab < 65536)
    tokens_np = np.array(tokens, dtype=np.uint16)
    tokens_np.tofile(output_path)
    print(f"Saved to {output_path}\n")

if __name__ == "__main__":
    train_input = os.path.join("data", "raw", "wikitext_103_train.txt")
    val_input = os.path.join("data", "raw", "wikitext_103_val.txt")
    train_output = os.path.join("data", "raw", "train.bin")
    val_output = os.path.join("data", "raw", "val.bin")
    
    tokenize_and_save(train_input, train_output)
    tokenize_and_save(val_input, val_output)
