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
    # query and key have same D
    # key and value have same L
    # query: [B, L_q, D]
    # key: [B, L_k, D]
    # value: [B, L_k, D]
    # scaled_score：[B, L_q, L_k]
    # mask：[B, 1, L_k]
    # attention：[B, L_q, L_k]
    # output：[B, L_q, D]
    # for multihead:
    # query：[B, L_q, num_heads, head_dim] -> [B, num_heads, L_q, head_dim]
    # key：[B, L_k, num_heads, head_dim] -> [B, num_heads, L_k, head_dim]
    # value：[B, L_k, num_heads, head_dim] -> [B, num_heads, L_k, head_dim]
    # scaled_score：[B, num_heads, L_q, L_k]
    # mask：[B, 1, 1, L_k]（broadcast in num_heads and L_q）
    # attention：[B, num_heads, L_q, L_k]
    # output：comcatenate and linear transformation to [B, L_q, D]
    

    dk = key.shape[-1]
    score = torch.matmul(query,key.transpose(-1,-2)) #BxLxD
    scaled_score = score/math.sqrt(dk)
    print(f"Scaled score shape: {scaled_score.shape}")
    #Masking (optional) 
    #Increase score to very large negative number for tokens that are masked.
    #Such large negative number will have 0 exponentiation and hence their softmax will be 0 as well. 
    if mask is not None:
        print(f"Mask shape: {mask.shape}")
        mask = mask.unsqueeze(1)  # mask is extended to [batch_size, 1, key_len]
        print(f"Mask shape after unsqueeze: {mask.shape}")
        scaled_score.masked_fill(mask==0,-1e9)
    attention = torch.softmax(scaled_score,dim=-1)
    if mask is not None:
        attention = attention * mask 
    #Optional: Dropout
    if dropout is not None:
        attention = nn.Dropout(dropout)(attention)
    #Z = enriched embedding 
    output = torch.matmul(attention,value)
    return output, attention
    

    



class MultiHeadedAttention(nn.Module):
    def __init__(self,nheads,dmodel,dropout=0.1):
        super(MultiHeadedAttention,self).__init__()
        assert dmodel % nheads ==0 
        self.dk = dmodel//nheads
        self.nheads =  nheads
        
        self.Wq = nn.Linear(dmodel,dmodel)
        self.Wk = nn.Linear(dmodel,dmodel)
        self.Wv = nn.Linear(dmodel,dmodel)
        self.Wo = nn.Linear(dmodel,dmodel)
        
        self.dropout_value = dropout
        self.dropout = nn.Dropout(p= dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all of the nheads
            mask = mask.unsqueeze(1) 

        batch_size = query.size(0)  
        seq_length = query.size(1)  
    
        # Project key, query, value using linear layers
        key, query, value = self.Wk(key), self.Wq(query), self.Wv(value)  # k, q, v = (B, L, dmodel)
        
        # Reshape to (B, L, nheads, dk), where dk = dmodel // nheads
        key = key.view(batch_size, seq_length, self.nheads, self.dk)  
        query = query.view(batch_size, seq_length, self.nheads, self.dk)  
        value = value.view(batch_size, seq_length, self.nheads, self.dk)  
        
        # Transpose to (B, nheads, L, dk) to prepare for attention calculation
        key = key.transpose(1, 2)    # (B, L, nheads, dk) --> (B, nheads, L, dk)
        query = query.transpose(1, 2)  
        value = value.transpose(1, 2)
    
        # Calculate self-attention
        z, self.attn = attention(query, key, value, mask, self.dropout_value)  # z: (B, nheads, L, dk)
    
        # Reshape z from (B, nheads, L, dk) --> (B, L, nheads * dk)
        z_concat = z.transpose(1, 2).contiguous()  # z_concat: (B, L, nheads, dk)
        z_concat = z_concat.view(batch_size, seq_length, -1)  # z_concat: (B, L, nheads * dk)
        
        # Project the concatenated output back to (B, L, dmodel)
        print("z_concat's shape:", z_concat.shape())
        z_enriched = self.Wo(z_concat)  # z_enriched: (B, L, dmodel)
    
        return z_enriched


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


