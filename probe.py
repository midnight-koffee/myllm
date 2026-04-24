import torch
import torch.nn.functional as F
import tiktoken
import json
from train_gpt_cpu import GPT  # reuse model definition

device = 'cpu'

# Load config and model
with open('configs/model_config.json') as f:
    config = json.load(f)
model = GPT().to(device)
checkpoint = torch.load('outputs/model_controlled.pth', map_location=device)
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint
model.load_state_dict(state_dict)
model.eval()

enc = tiktoken.get_encoding("gpt2")

prompts = [
    "Machine learning is",
    "Machine learning is a",
    "Machine learning is not",
    "Machine learning is the",
    "The capital of France is",
    "The capital of France is the",
    "Elon Musk is",
    "Elon Musk is a"
]

for prompt in prompts:
    print("\n" + "="*60)
    print(f"Prompt: '{prompt}'")
    idx = torch.tensor([enc.encode(prompt)], device=device)
    with torch.no_grad():
        logits, _ = model(idx)
        probs = F.softmax(logits[0, -1], dim=-1)
        topk = torch.topk(probs, 8)
    for i in range(8):
        token = enc.decode([topk.indices[i].item()])
        prob = topk.values[i].item()
        print(f"  {token!r:15} : {prob:.4f}")
