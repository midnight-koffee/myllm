import torch
import torch.nn.functional as F
import tiktoken
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from train_gpt_cpu import GPT  # will not train now if guard is added, but okay even if not

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
prompt = "The capital of France is"
tokens = enc.encode(prompt)
token_strs = [enc.decode([t]) for t in tokens]

# Compute logits and attention for the prompt
idx = torch.tensor([tokens], device=device)
with torch.no_grad():
    logits, _, attn_weights = model(idx, return_attention=True, layer_idx=2)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    topk = torch.topk(probs, 10)
    top_tokens = [enc.decode([t.item()]) for t in topk.indices]
    top_probs = topk.values.cpu().numpy()

    attn_head0 = attn_weights[0, 0].cpu().numpy()  # (T, T)
    last_row = attn_head0[-1]

# Create animation figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Inside a 13.5M Parameter LLM", fontsize=14, fontweight='bold')

# Left: Bar chart (static for now, we'll animate building it)
ax1.set_title("Next Token Probabilities")
bars = ax1.barh(range(10), [0]*10, color='steelblue')
ax1.set_yticks(range(10))
ax1.set_yticklabels(['']*10)
ax1.set_xlabel("Probability")
ax1.set_xlim(0, max(top_probs)*1.1)

# Right: Attention heatmap
ax2.set_title(f"Attention Weights (Layer 2, Head 0)\nQuery: '{token_strs[-1]}'")
im = ax2.imshow(np.zeros((len(tokens), len(tokens))), cmap='Blues', vmin=0, vmax=1)
ax2.set_xticks(range(len(tokens)))
ax2.set_xticklabels(token_strs, rotation=45, ha='right')
ax2.set_yticks(range(len(tokens)))
ax2.set_yticklabels(token_strs)

# Highlight the last row
for i in range(len(tokens)):
    ax2.add_patch(Rectangle((-0.5, i-0.5), len(tokens), 1, fill=False, edgecolor='red', lw=2 if i==len(tokens)-1 else 0))

def animate(frame):
    # Frame 0-9: reveal bars one by one
    if frame < 10:
        for i in range(frame+1):
            bars[i].set_width(top_probs[i])
            ax1.set_yticklabels(['']*10)
            ax1.set_yticklabels([f"{top_tokens[i]!r}" if j <= frame else '' for j in range(10)])
    # Frame 10-19: highlight attention row gradually
    elif frame < 20:
        pass  # keep bars
    # Frame 20+: show attention weights as text overlay
    else:
        for i, (tok, w) in enumerate(zip(token_strs, last_row)):
            ax2.text(i, len(tokens)-1, f"{w:.2f}", ha='center', va='center', color='white' if w>0.5 else 'black', fontsize=8)
    return bars, im

ani = animation.FuncAnimation(fig, animate, frames=30, interval=300, blit=False)
ani.save('llm_inference.gif', writer='pillow', fps=3)
print("Animation saved as llm_inference.gif")
plt.close()
