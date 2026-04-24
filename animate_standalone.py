import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

# -----------------------------------------------------------------------------
# Model definition (exact copy, no external imports)
device = 'cpu'

with open('configs/model_config.json') as f:
    config = json.load(f)

n_embd = config['n_embd']
n_head = config['n_head']
n_layer = config['n_layer']
block_size = config['block_size']
dropout = config['dropout']
vocab_size = config['vocab_size']

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
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

# -----------------------------------------------------------------------------
# Load checkpoint
model = GPT().to(device)
ckpt = torch.load('outputs/model_controlled.pth', map_location=device)
if 'model_state_dict' in ckpt:
    state_dict = ckpt['model_state_dict']
else:
    state_dict = ckpt
model.load_state_dict(state_dict)
model.eval()

enc = tiktoken.get_encoding("gpt2")
prompt = "The capital of France is the"
tokens = enc.encode(prompt)
token_strs = [enc.decode([t]) for t in tokens]

# Forward pass to get probs and attention
idx = torch.tensor([tokens], device=device)
with torch.no_grad():
    logits, _, attn_weights = model(idx, return_attention=True, layer_idx=2)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    topk = torch.topk(probs, 8)
    top_tokens = [enc.decode([t.item()]) for t in topk.indices]
    top_probs = topk.values.cpu().numpy()

    attn_head0 = attn_weights[0, 0].cpu().numpy()  # (T, T)
    last_row = attn_head0[-1]

# Create animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Inside a 13.5M Parameter LLM", fontsize=14, fontweight='bold')

ax1.set_title("Next Token Probabilities")
bars = ax1.barh(range(8), [0]*8, color='steelblue')
ax1.set_yticks(range(8))
ax1.set_yticklabels(['']*8)
ax1.set_xlabel("Probability")
ax1.set_xlim(0, max(top_probs)*1.1)

ax2.set_title(f"Attention Weights (Layer 2, Head 0)\nQuery: '{token_strs[-1]}'")
im = ax2.imshow(np.zeros((len(tokens), len(tokens))), cmap='Blues', vmin=0, vmax=1)
ax2.set_xticks(range(len(tokens)))
ax2.set_xticklabels(token_strs, rotation=45, ha='right')
ax2.set_yticks(range(len(tokens)))
ax2.set_yticklabels(token_strs)

for i in range(len(tokens)):
    ax2.add_patch(Rectangle((-0.5, i-0.5), len(tokens), 1, fill=False, edgecolor='red', lw=2 if i==len(tokens)-1 else 0))

def animate(frame):
    # Reveal bars gradually
    for i in range(min(frame+1, 8)):
        bars[i].set_width(top_probs[i])
    ax1.set_yticklabels([f"{top_tokens[i]!r}" if i <= frame else '' for i in range(8)])
    # Show attention weights as text after bars are fully shown
    if frame >= 8:
        for i, (tok, w) in enumerate(zip(token_strs, last_row)):
            ax2.text(i, len(tokens)-1, f"{w:.2f}", ha='center', va='center', color='white' if w>0.5 else 'black', fontsize=8)
    return bars, im

ani = animation.FuncAnimation(fig, animate, frames=12, interval=400, blit=False)
ani.save('llm_inference.gif', writer='pillow', fps=2.5)
print("Animation saved as llm_inference.gif")
plt.close()
