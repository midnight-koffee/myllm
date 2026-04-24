import torch
import platform
import sys

print("=" * 50)
print("SYSTEM DIAGNOSTIC FOR LLM TRAINING")
print("=" * 50)

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Processor: {platform.processor()}")

print("\n--- PyTorch Information ---")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
else:
    print("No CUDA-capable GPU detected.")

# Check for MPS (Apple Silicon)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("Apple MPS (Metal Performance Shaders) available: Yes")
else:
    print("Apple MPS available: No")

# Memory check
import psutil
mem = psutil.virtual_memory()
print(f"\n--- System Memory ---")
print(f"Total RAM: {mem.total / 1024**3:.1f} GB")
print(f"Available RAM: {mem.available / 1024**3:.1f} GB")

# Simple tensor operation test
print("\n--- Performance Test ---")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.randn(1000, 1000).to(device)
start = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
end = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None

import time
t0 = time.time()
_ = x @ x
t1 = time.time()

if device == 'cuda':
    torch.cuda.synchronize()
print(f"Matrix multiplication (1000x1000) on {device}: {t1-t0:.4f} seconds")

print("\n--- Recommendation ---")
if torch.cuda.is_available():
    print("✅ GPU available! You can train with larger models and batch sizes.")
    print("   Suggested settings: n_embd=384, n_layer=6, batch_size=64")
else:
    print("⚠️ No GPU detected. Training will run on CPU.")
    print("   This is perfectly fine for learning, but will be slower.")
    print("   Suggested settings for CPU: n_embd=128, n_layer=3, batch_size=16, max_iters=1000")
    print("   You can still complete training overnight.")
