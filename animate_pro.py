import torch
import torch.nn.functional as F
import tiktoken
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as path_effects

# Set high-quality styling
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
rcParams['font.size'] = 12
rcParams['text.color'] = 'white'
rcParams['axes.facecolor'] = '#1e1e2f'
rcParams['figure.facecolor'] = '#1e1e2f'
rcParams['axes.edgecolor'] = '#444'
rcParams['axes.labelcolor'] = 'white'
rcParams['xtick.color'] = 'white'
rcParams['ytick.color'] = 'white'
rcParams['grid.color'] = '#444'
rcParams['figure.dpi'] = 150

# Load model and config (standalone, no training)
device = 'cpu'
with open('configs/model_config.json') as f:
    config = json.load(f)

n_embd = config['n_embd']; n_head = config['n_head']; n_layer = config['n_layer']
block_size = config['block_size']; dropout = config['dropout']; vocab_size = config['vocab_size']

# Model definition (exact copy, no import side effects)
import torch.nn as nn
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
        k = self.key(x); q = self.query(x)
        wei = q @ k.transpose(-2,-1) * (k.shape[-1]**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ self.value(x)
        return (out, wei) if return_attention else out

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
                outs.append(out); attns.append(attn)
            out = torch.cat(outs, dim=-1)
            return self.dropout(self.proj(out)), torch.stack(attns, dim=1)
        else:
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            return self.dropout(self.proj(out))

class FeedFoward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.ReLU(), nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward()
        self.ln1 = nn.LayerNorm(n_embd); self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.sa(self.ln1(x), return_attention=True)
            x = x + attn_out
        else:
            x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return (x, attn_weights) if return_attention else x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None, return_attention=False, layer_idx=-1):
        B,T = idx.shape
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
        loss = None
        if targets is not None:
            B,T,C = logits.shape
            logits = logits.view(B*T, C); targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return (logits, loss, attn_weights) if return_attention else (logits, loss)

# Load model
model = GPT().to(device)
ckpt = torch.load('outputs/model_controlled.pth', map_location=device)
state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
model.load_state_dict(state_dict)
model.eval()
enc = tiktoken.get_encoding("gpt2")

# Get real probabilities for two prompts
prompt1 = "Machine learning is"
prompt2 = "Machine learning is a"
tokens1 = enc.encode(prompt1)
tokens2 = enc.encode(prompt2)
token_strs1 = [enc.decode([t]) for t in tokens1]
token_strs2 = [enc.decode([t]) for t in tokens2]

with torch.no_grad():
    logits1, _ = model(torch.tensor([tokens1]))
    probs1 = F.softmax(logits1[0, -1], dim=-1)
    topk1 = torch.topk(probs1, 5)
    top_tokens1 = [enc.decode([t.item()]) for t in topk1.indices]
    top_probs1 = topk1.values.cpu().numpy()

    logits2, _ = model(torch.tensor([tokens2]))
    probs2 = F.softmax(logits2[0, -1], dim=-1)
    topk2 = torch.topk(probs2, 5)
    top_tokens2 = [enc.decode([t.item()]) for t in topk2.indices]
    top_probs2 = topk2.values.cpu().numpy()

# Attention for "The capital of France is the"
prompt3 = "The capital of France is the"
tokens3 = enc.encode(prompt3)
token_strs3 = [enc.decode([t]) for t in tokens3]
with torch.no_grad():
    logits3, _, attn_weights = model(torch.tensor([tokens3]), return_attention=True, layer_idx=2)
    probs3 = F.softmax(logits3[0, -1], dim=-1)
    topk3 = torch.topk(probs3, 5)
    top_tokens3 = [enc.decode([t.item()]) for t in topk3.indices]
    top_probs3 = topk3.values.cpu().numpy()
    attn_head0 = attn_weights[0, 0].cpu().numpy()
    last_row = attn_head0[-1]

# Animation
fig = plt.figure(figsize=(14, 7), facecolor='#1e1e2f')
fig.suptitle("Inside a 13.5M Parameter Language Model", fontsize=18, fontweight='bold', color='white', y=0.98)

# Layout: three panels
gs = fig.add_gridspec(2, 3, height_ratios=[0.3, 1], hspace=0.4, wspace=0.3)
ax_token1 = fig.add_subplot(gs[0, 0])
ax_token2 = fig.add_subplot(gs[0, 1])
ax_attention = fig.add_subplot(gs[0, 2])
ax_prob = fig.add_subplot(gs[1, :])

for ax in [ax_token1, ax_token2, ax_attention, ax_prob]:
    ax.set_facecolor('#1e1e2f')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

ax_token1.set_title("Prompt 1", color='white', fontweight='bold')
ax_token2.set_title("Prompt 2", color='white', fontweight='bold')
ax_attention.set_title("Attention Flow (Layer 2, Head 0)", color='white', fontweight='bold')
ax_prob.set_title("Next‑Token Probability Distribution", color='white', fontweight='bold')

# Static token displays
ax_token1.text(0.5, 0.5, prompt1, ha='center', va='center', fontsize=16, color='lightblue', transform=ax_token1.transAxes)
ax_token2.text(0.5, 0.5, prompt2, ha='center', va='center', fontsize=16, color='lightblue', transform=ax_token2.transAxes)
ax_token1.set_xticks([]); ax_token1.set_yticks([])
ax_token2.set_xticks([]); ax_token2.set_yticks([])

# Probability bars
bar_colors1 = ['#4ecdc4']*5
bar_colors2 = ['#ff6b6b']*5
y_pos = np.arange(5)
bars1 = ax_prob.barh(y_pos - 0.2, [0]*5, height=0.4, color=bar_colors1, label=prompt1)
bars2 = ax_prob.barh(y_pos + 0.2, [0]*5, height=0.4, color=bar_colors2, label=prompt2)
ax_prob.set_yticks(y_pos)
ax_prob.set_yticklabels(['']*5)
ax_prob.set_xlim(0, max(max(top_probs1), max(top_probs2)) * 1.2)
ax_prob.set_xlabel('Probability', color='white')
ax_prob.legend(loc='upper right', facecolor='#2d2d44', edgecolor='none', labelcolor='white')

# Attention panel setup
ax_attention.set_xlim(-0.5, len(token_strs3)-0.5)
ax_attention.set_ylim(-0.5, 2.5)
ax_attention.set_xticks(range(len(token_strs3)))
ax_attention.set_xticklabels(token_strs3, rotation=45, ha='right', fontsize=9)
ax_attention.set_yticks([0, 1, 2])
ax_attention.set_yticklabels(['Tokens', 'Attention\nWeights', 'Next Token'])
for y in [0, 1]:
    ax_attention.axhline(y=y+0.5, color='#444', linestyle='--', linewidth=0.5)

# Token boxes
token_patches = []
for i, tok in enumerate(token_strs3):
    rect = FancyBboxPatch((i-0.3, 0-0.25), 0.6, 0.5, boxstyle="round,pad=0.02", facecolor='#2d2d44', edgecolor='#4ecdc4', linewidth=1)
    ax_attention.add_patch(rect)
    ax_attention.text(i, 0, tok, ha='center', va='center', fontsize=9, color='white')
    token_patches.append(rect)

# Query indicator
query_circle = plt.Circle((len(token_strs3)-1, 0), 0.15, color='#ff6b6b', zorder=5)
ax_attention.add_patch(query_circle)
ax_attention.text(len(token_strs3)-1, -0.5, "Query", ha='center', fontsize=8, color='#ff6b6b')

# Attention arrows (will be updated)
arrows = []
weight_texts = []

def animate(frame):
    # Phase 1: Show tokens (frames 0-5)
    if frame < 5:
        alpha = frame / 4
        for patch in token_patches:
            patch.set_alpha(alpha)
        return token_patches

    # Phase 2: Morph probabilities (frames 5-20)
    elif frame < 20:
        t = (frame - 5) / 15  # 0 to 1
        # Interpolate probabilities
        interp_probs1 = top_probs1 * (1 - t)
        interp_probs2 = top_probs2 * t
        for i, (b1, b2) in enumerate(zip(bars1, bars2)):
            b1.set_width(interp_probs1[i])
            b2.set_width(interp_probs2[i])
        # Update y-tick labels
        if t < 0.5:
            labels = top_tokens1
        else:
            labels = top_tokens2
        ax_prob.set_yticklabels([f"{tok!r}" for tok in labels])
        # Show which prompt is active
        if t < 0.2:
            ax_prob.set_title("Next‑Token Probabilities: \"Machine learning is\"", color='white', fontweight='bold')
        elif t > 0.8:
            ax_prob.set_title("Next‑Token Probabilities: \"Machine learning is a\"", color='white', fontweight='bold')
        else:
            ax_prob.set_title("Probability Distribution Shifting...", color='white', fontweight='bold')
        return bars1 + bars2

    # Phase 3: Attention flow (frames 20-35)
    else:
        t = (frame - 20) / 15
        # Clear previous arrows
        for a in arrows:
            a.remove()
        arrows.clear()
        for txt in weight_texts:
            txt.remove()
        weight_texts.clear()
        # Draw attention arrows from query to each token
        query_x = len(token_strs3) - 1
        for i, w in enumerate(last_row):
            if w > 0.05:  # Only show significant weights
                alpha = min(1, t * 3)
                arrow = FancyArrowPatch((query_x, 0.25), (i, 0.75), 
                                        arrowstyle='-|>', mutation_scale=15, 
                                        color=plt.cm.Blues(w*2), alpha=alpha*w*2, linewidth=w*15)
                ax_attention.add_patch(arrow)
                arrows.append(arrow)
                txt = ax_attention.text((query_x + i)/2, 0.5, f"{w:.2f}", ha='center', va='bottom', 
                                        color='white', fontsize=8, alpha=alpha)
                weight_texts.append(txt)
        # Highlight the strongest connection
        if t > 0.7:
            max_idx = np.argmax(last_row)
            ax_attention.text(max_idx, 1.8, f"Max: {last_row[max_idx]:.2f}", ha='center', 
                              fontsize=10, color='#4ecdc4', fontweight='bold')
        return arrows + weight_texts

ani = animation.FuncAnimation(fig, animate, frames=35, interval=150, blit=False)
ani.save('llm_insight.gif', writer='pillow', fps=8)
print("High-quality animation saved as llm_insight.gif")
