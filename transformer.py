import torch
import torch.nn as nn
from torch.nn import functional as F 
#%%
device = 'cuda' 
dropout = 0.2
context_size = 32
batch_size = 16
embed_length = 64
num_heads = 4
num_blocks = 4
lr = 1e-3
num_iters = 5000
torch.manual_seed(1337)
#%%
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_length, head_size, bias=False)
        self.query = nn.Linear(embed_length, head_size, bias=False)
        self.value = nn.Linear(embed_length, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        self.out = weights @ v
        return self.out
    
class MultiHead(nn.Module):
    def __init__(self, head_size, num_heads, embed_length):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for i in range(num_heads)])
        self.proj = nn.Linear(embed_length, embed_length)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        multi_attention = [head(x) for head in self.heads]
        attention = torch.cat(multi_attention, dim=-1)
        self.out = self.proj(attention)
        self.out = self.dropout(self.out)
        return self.out
  
class FeedForward(nn.Module):
    def __init__(self, embed_length):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(embed_length, 4*embed_length), nn.ReLU(), nn.Linear(4*embed_length, embed_length), nn.Dropout(dropout))
    def forward(self, x):
        self.out = self.layer(x)
        return self.out
        
# class LayerNorm1D:
#     def __init__(self, dim, eps=1e-5, momentum=0.1):
#         self.eps = eps
#         self.momentum = momentum
#         self.training = True
#         self.gamma = torch.ones(dim)
#         self.beta = torch.zeros(dim)
#     def __call__(self, inputs):
#         mean = inputs.mean(1, keepdim=True)
#         var = inputs.var(1, keepdim=True)
#         self.out = self.gamma * ((inputs - mean) / torch.sqrt(var + self.eps)) + self.beta
#         return self.out
#     def parameters(self):
#         return [self.gamma, self.beta]

class Block(nn.Module):
    def __init__(self, embed_length, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = embed_length // num_heads
        self.norm1 = nn.LayerNorm(embed_length)
        self.attention = MultiHead(self.head_size, self.num_heads, embed_length)
        self.norm2 = nn.LayerNorm(embed_length)
        self.fwd = FeedForward(embed_length)
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.fwd(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, context_size, embed_length, num_heads, num_blocks):
        super().__init__()
        self.context_size = context_size
        self.char_to_embedding = nn.Embedding(vocab_size, embed_length)
        self.pos_to_embedding = nn.Embedding(context_size, embed_length)
        self.blocks = nn.Sequential(*[Block(embed_length, num_heads) for i in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_length)
        self.final = nn.Linear(embed_length, vocab_size)
    def forward(self, x, targets=None):
        B, T = x.shape
        char_token = self.char_to_embedding(x)
        pos_token = self.pos_to_embedding(torch.arange(T))
        x = char_token + pos_token
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.final(x)
        if targets==None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_length):
        for _ in range(max_length):
            idx_block = idx[:, -self.context_size:]
            logits, loss = self(idx_block)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            char = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, char), dim=1)
        return idx
#%%
with open('C://Users/ATMH2/Documents/Python codes/python libraries/shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for i, s in enumerate(chars)}
encode = lambda s: [stoi[char] for char in s]
decode = lambda s: ''.join([itos[index] for index in s])
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(100)
        for k in range(100):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_batch(mode):
    data = train_data if mode == 'train' else val_data
    batch_index = torch.randint(len(data) - context_size, (batch_size,)) 
    Xbatch = torch.stack([data[i:context_size+i] for i in batch_index])
    Ybatch = torch.stack([data[i+1:context_size+i+1] for i in batch_index])
    return Xbatch, Ybatch
#%%
model = Transformer(len(chars), context_size, embed_length, num_heads, num_blocks)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for i in range(num_iters):
    if i % 100 == 0 or i == num_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    Xbatch, Ybatch = get_batch('train')
    logits, loss = model(Xbatch, Ybatch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, max_length=2000)[0].tolist()))