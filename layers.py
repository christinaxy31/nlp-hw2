import torch
import torch.nn as nn
import math
from utils import clones
from torch.nn.functional import log_softmax


class LayerNorm(nn.Module):
    "Construct a layernorm module - https://arxiv.org/abs/1607.06450"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection (https://arxiv.org/abs/1512.03385) followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attention and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math


import torch
import math

def attention(query, key, value, mask=None, dropout=None):
    # query: (batch_size, h, seq_len_q, d_k)
    # key: (batch_size, h, seq_len_k, d_k)
    # value: (batch_size, h, seq_len_k, d_v)

    d_k = query.size(-1)

    # Compute scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # (batch_size, h, seq_len_q, seq_len_k)
    print(f"Scores shape: {scores.shape}")  # Debugging: Print scores shape

    # Adjust mask shape if necessary
    if mask is not None:
        print(f"Mask shape before unsqueeze: {mask.shape}")  # Debugging: Print mask shape before unsqueeze
        mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len_k)
        print(f"Mask shape after unsqueeze: {mask.shape}")  # Debugging: Print mask shape after unsqueeze
        print(f"Mask values: {mask}")  # Debugging: Print mask values

        # Ensure mask can broadcast with scores
        assert mask.size(-1) == scores.size(
            -1), f"Mask shape {mask.shape} is not compatible with scores shape {scores.shape}"
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax over the last dimension
    attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, h, seq_len_q, seq_len_k)
    print(f"Attention weights after softmax: {attn_weights}")  # Debugging: Print attention weights

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # Check that attention weights are correctly masked
    if mask is not None:
        # Locations where mask is 0 should have near-zero attention weights
        masked_positions = attn_weights.masked_select(mask == 0)
        print(f"Attention weights at masked positions: {masked_positions}")
        assert torch.allclose(masked_positions, torch.zeros_like(masked_positions)), "Attention weights are incorrectly masked"

    # Multiply attention weights with value
    output = torch.matmul(attn_weights, value)  # (batch_size, h, seq_len_q, d_v)
    return output, attn_weights



class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.h = h
        self.d_k = d_model // h
        self.d_model = d_model

        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Check the shapes of query, key, and value
        print(
            f"Query shape: {query.shape}, Key shape: {key.shape}, Value shape: {value.shape}")  # Debugging: Print input shapes

        # Ensure d_model is divisible by h
        assert self.d_model % self.h == 0, "d_model must be divisible by the number of heads (h)."

        # Linear projection and reshape to (batch_size, h, seq_len, d_k)
        query, key, value = [linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for linear, x in zip(self.linears, (query, key, value))]


class PositionwiseFeedForward(nn.Module):                                               
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())    


