import torch
import tiktoken
import torch.nn as nn
from torch.nn import functional as F
import json

# -----------------------------------------------------------------------------
# Hyperparameters EXACTLY matching the saved model_controlled.pth checkpoint
batch_size = 16
block_size = 128      # <-- matches config
device = 'cpu'
n_embd = 128
n_head = 4
n_layer = 3
dropout = 0.2
vocab_size = 50257
# -----------------------------------------------------------------------------

# ---------- Model Definition (exact copy from training) ----------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * (k.shape[-1]**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
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
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def load_model(weights_path):
    print(f"Loading model from {weights_path}...")
    model = GPT()
    checkpoint = torch.load(weights_path, map_location=device)
    # Handle wrapped checkpoint (new format)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded checkpoint from run with final loss: {checkpoint.get('final_loss', 'N/A'):.4f}")
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def generate_text(model, prompt, max_new_tokens=80, temperature=0.8):
    enc = tiktoken.get_encoding("gpt2")
    prompt_tokens = enc.encode(prompt)
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        generated_idx = model.generate(idx, max_new_tokens, temperature)
    generated_tokens = generated_idx[0].tolist()
    return enc.decode(generated_tokens)

if __name__ == "__main__":
    model = load_model("outputs/model_controlled.pth")
    print("\n🎉 Your improved LLM is ready! Type a prompt and see what it generates.")
    print("Type 'quit' to exit.\n")
    while True:
        prompt = input("👉 Prompt: ").strip()
        if prompt.lower() == 'quit':
            break
        if not prompt:
            continue
        output = generate_text(model, prompt, max_new_tokens=100)
        print("\n📝 Generated:\n")
        print(output)
        print("\n" + "-"*60 + "\n")
