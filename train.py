import torch
import torch.nn as nn
from torch.nn import functional as F

#      opening text
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# print("length of dataset in characters: ", len(text))
# print(text[:1000])

#      unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)

#      tokenize input text (raw text to integers)
#      mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
#      encoder: take a string, output a list of integers
encode = lambda s: [stoi[c] for c in s]
#      decoder: take a list of integers, output a string
decode = lambda l: ''.join([itos[i] for i in l])
# print(encode('hii there'))
# print(decode(encode('hii there')))

#      encoding entire text & storing into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000]) # what the 1000 characters will look like for GPT

#      splitting data into train & validation sets
n = int(0.9*len(data)) # first 90% for training, rest for validation
train_data = data[:n]
val_data = data[n:]

#      training parameters
block_size = 8
# print(train_data[:block_size+1]) # plus one because we're using past context to predict the next char, example below
# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"when input is {context} the target: {target}")

#      batch dimensions
torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will be processed in parallel
block_size = 8 # maximum context length for predictions (same var as above)

def get_batch(split):
    # generates a small batch of data of intputs x and targets y
    data = train_data if split == 'trains' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 4 chunks of chars
    x = torch.stack([data[i:i+block_size] for i in ix]) # context of the 4 chunks
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # targets of the 4 chunks
    return x, y

xb, yb = get_batch('train') # inputs & targets to the transformer
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)
#
# print('----')
#
# for b in range(batch_size): # batch dimension
#     for t in range(block_size): # time dimension
#         context = xb[b, :t+1]
#         target = yb[b,t]
#         print(f"when input is {context.tolist()} the target: {target}")

torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx is the identities of the context
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C) where C is vocab_size
        # logits are the scores/integers for the next char preidcted in the sequence
        # model is predicting what comes next based on a single token

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
            # get predictions
            logits, loss = self(idx) # loss is ignored
            # focus only on the last time step
            logits = logits[:, -1, :] # become (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)
#
# idx = torch.zeros((1,1), dtype=torch.long) # used to kick off the generation
# print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

#      training the model
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3) # PyTorch optimizer, updates parameters

batch_size = 32
for steps in range(10000):

    # sampling a batch of data
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True) # zeroing gradients of the prev step
    loss.backward() # getting gradients for all parameters
    optimizer.step() # using above gradients to update parameters

print(loss.item())
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=300)[0].tolist()))