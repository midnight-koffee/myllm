import os
import sys
sys.path.append(os.getcwd())  # Add current directory to path
from src.training.dataset import TextDataset, get_batch

# Path to training binary
train_bin = os.path.join("data", "raw", "train.bin")

# Create dataset with context length of 128 tokens
dataset = TextDataset(train_bin, block_size=128)
print(f"Dataset size: {len(dataset):,} possible sequences")

# Get a batch of 4 sequences
batch_size = 4
x, y = get_batch(dataset, batch_size)

print(f"\nBatch X shape: {x.shape}")  # Should be [4, 128]
print(f"Batch Y shape: {y.shape}")    # Should be [4, 128]

print("\n--- First sequence in batch ---")
print(f"Input X[0, :10]:  {x[0, :10].tolist()}")
print(f"Target Y[0, :10]: {y[0, :10].tolist()}")
print("\nNotice how Y is exactly X shifted right by one position.")
print("This is the 'next-token prediction' task.")
