import torch
import torch.nn as nn
import torch.nn.functional as F


#Hyper Parameters
batch_size = 32
context_window = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iters = 200


torch.manual_seed(69420)

with open("input.txt", "r", encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for i,c in enumerate(chars)}

encode = lambda text: [stoi[char] for char in text]
decode = lambda nums : ''.join(itos[num] for num in nums)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - context_window, (batch_size,))
    x = torch.stack([data[i:i+context_window] for i in idx])
    y = torch.stack([data[i+1:i+context_window+1] for i in idx])
    x = x.to(device)
    y = y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[i] = loss.item()

        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size : int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        logits = self.token_embedding_table(idx) #B,T -> B,T,C
        # print(logits.shape)
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_tokens):

        for _ in range(max_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # Only consider last time stamp B,T,C -> B, C
            probs = F.softmax(logits, dim=-1) #B, C -> B, C
            idx_next = torch.multinomial(probs, num_samples=1) # B, C -> B, 1

            idx = torch.concat((idx, idx_next), dim=1) #Append to input then predict more B, T+1

        return idx

model = BigramLanguageModel(vocab_size)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_tokens = 500)[0].tolist()))