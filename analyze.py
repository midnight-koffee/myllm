import torch
import tiktoken
import torch.nn as nn
from torch.nn import functional as F
import json
import numpy as np

# ---------- Load config and model ----------
with open('configs/model_config.json') as f:
    config = json.load(f)

# Hyperparameters must match config
n_embd = config['n_embd']
n_head = config['n_head']
n_layer = config['n_layer']
block_size = config['block_size']
dropout = config['dropout']
vocab_size = config['vocab_size']
device = 'cpu'

# ---------- Model Definition (exact copy from training) ----------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * (k.shape[-1]**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        if return_attention:
            return out, wei
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        if return_attention:
            outs, attns = [], []
            for h in self.heads:
                out, attn = h(x, return_attention=True)
                outs.append(out)
                attns.append(attn)
            out = torch.cat(outs, dim=-1)
            return self.dropout(self.proj(out)), torch.stack(attns, dim=1)
        else:
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            return self.dropout(self.proj(out))

class FeedFoward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.sa(self.ln1(x), return_attention=True)
            x = x + attn_out
        else:
            x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        if return_attention:
            return x, attn_weights
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, return_attention=False, layer_idx=-1):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        attn_weights = None
        for i, block in enumerate(self.blocks):
            if return_attention and i == layer_idx:
                x, attn_weights = block(x, return_attention=True)
            else:
                x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        if return_attention:
            return logits, loss, attn_weights
        return logits, loss

# ---------- Load checkpoint ----------
model = GPT().to(device)
checkpoint = torch.load('outputs/model_controlled.pth', map_location=device)
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print(f"Loaded model (final train loss: {checkpoint['final_loss']:.4f})")
else:
    state_dict = checkpoint
model.load_state_dict(state_dict)
model.eval()

enc = tiktoken.get_encoding("gpt2")

# ---------- 1. Top‑5 Next Token Predictions ----------
print("\n" + "="*60)
print("1. TOP-5 NEXT TOKEN PROBABILITIES")
print("="*60)

prompt = "Machine learning is"
idx = torch.tensor([enc.encode(prompt)]).to(device)

with torch.no_grad():
    logits, _ = model(idx)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    topk = torch.topk(probs, 5)

print(f"Prompt: '{prompt}'")
print("Next token predictions:")
for i in range(5):
    token_str = enc.decode([topk.indices[i].item()])
    prob = topk.values[i].item()
    print(f"  {i+1}. {token_str!r:15} : {prob:.4f}")

# ---------- 2. Attention Visualization (Layer 2, Head 0) ----------
print("\n" + "="*60)
print("2. ATTENTION WEIGHTS (Layer 2, Head 0)")
print("="*60)

prompt = "The capital of France is"
idx = torch.tensor([enc.encode(prompt)]).to(device)

with torch.no_grad():
    logits, _, attn_weights = model(idx, return_attention=True, layer_idx=2)
    # attn_weights shape: (B, n_head, T, T)  -> take head 0
    attn_head0 = attn_weights[0, 0].cpu().numpy()  # shape (T, T)
    tokens = [enc.decode([t]) for t in idx[0].tolist()]

print(f"Prompt tokens: {tokens}")
print("\nAttention matrix (last row shows what the final token attends to):")
# Show the last row (what the last token attends to)
last_row = attn_head0[-1]
for i, (tok, w) in enumerate(zip(tokens, last_row)):
    print(f"  {tok:10} : {w:.3f}")
