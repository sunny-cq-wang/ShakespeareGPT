import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparamters
batch_size = 32 # how many independent sequences will be processed in parallel
block_size = 128 # what the max context length is for prediction
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 192
n_head = 3
n_layer = 3
dropout = 0.2
# ------------

torch.manual_seed(1337)

#      opening text
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#      unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

#      tokenize input text (raw text to integers)
#      mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
#      encoder: take a string, output a list of integers
encode = lambda s: [stoi[c] for c in s]
#      decoder: take a list of integers, output a string
decode = lambda l: ''.join([itos[i] for i in l])

#      encoding entire text & storing into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
#      splitting data into train & validation sets
n = int(0.9*len(data)) # first 90% for training, rest for validation
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generates a small batch of data of intputs x and targets y
    data = train_data if split == 'trains' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 4 chunks of chars
    x = torch.stack([data[i:i+block_size] for i in ix]) # context of the 4 chunks
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # targets of the 4 chunks
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # tells PyTorch that all this code won't be called backward on, helps with memory efficiency
def estimate_loss():
    out = {}
    model.eval() # eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores (affinities)
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) ---> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) ---> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer going back into the residual pathway
            nn.Dropout(dropout),
        )


    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication follwed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_head = MultiHeadAttention(4, n_embd//4) # i.e. 4 heads of 8-dimensional self-attention
        # self.ffwd = FeedForward(n_embd)
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd),
        # )
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        # x = self.sa_head(x) # apply one head of self-attention, (B,T,C)
        # x = self.ffwd(x) # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,C) where C is vocab_size, not the same as one above

        if targets is None:
            loss = None
        else:
            # reshape logits
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # measuring quality of predictions, comparing the logits to the targets
            loss = F.cross_entropy(logits, targets) # wants it (B,C,T) instead of (B,T,C)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx in the last block_size tokens since pos_embd is only as large as the block_size
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_cond) # loss is ignored
            # focus only on the last time step
            logits = logits[:, -1, :] # become (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

#      training the model
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3) # PyTorch optimizer, updates parameters

for iter in range(max_iters):

    # every one in a while evaluate loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batchof data
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True) # zeroing gradients of the prev step
    loss.backward() # getting gradients for all parameters
    optimizer.step() # using above gradients to update paramters

# generate from the model
context = torch.zeros((2, 2), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

