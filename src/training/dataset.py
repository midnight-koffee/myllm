import torch
import numpy as np
import os

class TextDataset:
    def __init__(self, bin_path, block_size):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Input: tokens from idx to idx+block_size
        x = torch.tensor(self.data[idx : idx + self.block_size].astype(np.int64), dtype=torch.long)
        # Target: tokens from idx+1 to idx+block_size+1 (shifted by one)
        y = torch.tensor(self.data[idx + 1 : idx + self.block_size + 1].astype(np.int64), dtype=torch.long)
        return x, y

def get_batch(dataset, batch_size, device='cpu'):
    """Sample a random batch of sequences from the dataset."""
    # Random starting indices
    indices = torch.randint(len(dataset), (batch_size,))
    x_list, y_list = [], []
    for i in indices:
        x, y = dataset[i.item()]
        x_list.append(x)
        y_list.append(y)
    x_batch = torch.stack(x_list).to(device)
    y_batch = torch.stack(y_list).to(device)
    return x_batch, y_batch
