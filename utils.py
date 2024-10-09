import torch
import copy
import math
import torch.nn as nn
from torch.nn.functional import pad
import sacrebleu


## Dummy functions defined to use the same function run_epoch() during eval
class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.pe.unsqueeze(0)
        x = x + pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)



def greedy_decode(model, src, src_mask, max_len, start_symbol):
    # Step 1: 编码阶段，查看 memory
    memory = model.encode(src, src_mask)
    print(f"Memory shape: {memory.shape}")
    
    # 初始化 ys 为起始符号
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    print(f"Initial ys: {ys}")
    
    for i in range(max_len - 1):
        # Step 2: 每一步解码，查看 out 输出
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        print(f"Decoder output at step {i}: {out.shape}")
        
        # Step 3: 生成概率分布，检查 prob 和 next_word
        prob = model.generator(out[:, -1])
        print(f"Probability distribution at step {i}: {prob}")
        
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        print(f"Next word at step {i}: {next_word}")

        # Step 4: 扩展生成序列 ys
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
        print(f"Generated sequence ys at step {i}: {ys}")
    
    # 返回生成的完整序列
    return ys



def beam_search_decode(model, src, src_mask, max_len, start_symbol, beam_size, end_idx):
    """
    Implement beam search decoding with 'beam_size' width
    """
    
    # Encode the input sequence
    memory = model.encode(src, src_mask)
    # Initialize the decoder input with the start symbol
    ys = torch.full((1, 1), start_symbol, dtype=src.dtype).cuda()
    # Initialize the scores with 0
    scores = torch.zeros(1).cuda()

    for step in range(max_len - 1):
        # Decode the current sequence
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        # Get the probability distribution for the next token
        prob = model.generator(out[:, -1])
        vocab_size = prob.shape[-1]

        # Prevent EOS tokens from being expanded further
        prob[ys[:, -1] == end_idx, :] = 0
        # Update scores with the log probabilities
        scores = scores.unsqueeze(1) + prob

        # Get top-k scores and their corresponding indices
        scores, indices = torch.topk(scores.view(-1), beam_size)
        # Compute beam indices and token indices from the top-k indices
        beam_indices = torch.div(indices, vocab_size, rounding_mode='floor') # indices // vocab_size
        token_indices = torch.remainder(indices, vocab_size)

        # Prepare the next input for the decoder
        next_decoder_input = []
        for beam_idx, token_idx in zip(beam_indices, token_indices):
            prev_input = ys[beam_idx]
            if prev_input[-1] == end_idx:
                token_idx = end_idx  # Once EOS is reached, continue to use EOS
            token_idx = torch.LongTensor([token_idx]).cuda()
            next_decoder_input.append(torch.cat([prev_input, token_idx]))
        ys = torch.vstack(next_decoder_input)

        # Exit if all beams have reached EOS
        if (ys[:, -1] == end_idx).sum() == beam_size:
            break

        # Expand memory and src_mask for beam size after the first step
        if step == 0:
            memory = memory.expand(beam_size, *memory.shape[1:])
            src_mask = src_mask.expand(beam_size, *src_mask.shape[1:])

    # Select the highest-scoring sequence
    ys, _ = max(zip(ys, scores), key=lambda x: x[1])
    ys = ys.unsqueeze(0)

    return ys


def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for s in batch:
        _src = s['de']
        _tgt = s['en']
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def remove_start_end_tokens(sent):

    if sent.startswith('<s>'):
        sent = sent[3:]

    if sent.endswith('</s>'):
        sent = sent[:-4]

    return sent


def compute_corpus_level_bleu(refs, hyps):

    refs = [remove_start_end_tokens(sent) for sent in refs]
    hyps = [remove_start_end_tokens(sent) for sent in hyps]

    bleu = sacrebleu.corpus_bleu(hyps, [refs])

    return bleu.score

