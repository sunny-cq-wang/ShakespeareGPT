import torch
import torch.nn as nn
from torch.nn import functional as F

# # now want x[b,t] = mean_{i<=t} x[b,i] (self-attention preview)
# B,T,C = 4,8,2 # batch, time, channels
# x = torch.randn(B,T,C)
# xbow = torch.zeros((B,T,C)) # bow for "bag of words"
# for b in range(B): # inefficient version 1
#     for t in range(T):
#         xprev = x[b,:t+1] # (t,C)
#         xbow[b,t] = torch.mean(xprev, 0)
#
# # version 2
# wei = torch.tril(torch.ones(T, T)) # wei for weights
# wei = wei / wei.sum(1, keepim=True) # averaging
# xbow2 = wei @ x # (B, T, T) @ (B, T, C) ---> (B, T, C) (@ is a batch multiplier) xbow and xbow 2 are the same
#
# # version 3: using Softmax
# tril = torch.tril(torch.ones(T, T))
# wei = torch.zeros((T,T)) # begins as all zeros
# wei = wei.masked_fill(tril == 0, float('inf')) # "for all elements where tril == 0, make them be neg inf"
# wei = F.softmax(wei, dim=-1) # softmax on every row => normalization of the matrix
# xbow3 = wei @ x # xbow3 also identical to the first and second

# version 4: self-attention
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

# single Head performing self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x) # (B, T, 16)
q = query(x) # (B, T, 16)
wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ----> (B, T, T)
wei = wei * head_size**-0.5

tril = torch.tril(torch.ones(T, T))
# wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei,  dim=-1)

v = value(x)
out = wei @ v
# out = wei @ x # x if considered "private", v is "public"


class BatchNormld:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __cal__(self, x):
        # calculate the forward pass
        xmean = x.mean(1, keepdim=True)  # batch mean
        xvar = x.var(1, keepdim=True)  # batch variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # normalize to unit variance
        self.out = self.gamma + xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


torch.manual_seed(1337)
module = BatchNormld(100)
x = torch.randn(32, 100)  # batch size 32 of 100-dimensional vectors
x = module(x)