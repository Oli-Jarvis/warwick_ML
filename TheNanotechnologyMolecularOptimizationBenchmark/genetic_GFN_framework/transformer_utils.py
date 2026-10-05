import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class RMSNorm(nn.Module):
    """ Root Mean Square Layer Normalization """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class FeedForward(nn.Module):
    """ SwiGLU Feed-Forward Network """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) with KV Cache support.
    Accepts start_pos to correctly apply rotation to new tokens.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        theta = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("theta", theta)

        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, self.theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex number form
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x: torch.Tensor, start_pos: int):
        T = x.shape[1]
        freqs = self.freqs_cis[start_pos: start_pos + T].view(1, T, 1, -1)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * freqs
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.flatten(start_dim=3)
        return x_out.type_as(x)


class Attention(nn.Module):
    """ Grouped-Query Attention (GQA) with KV Cache """

    def __init__(self, d_model: int, n_query_heads: int, n_kv_heads: int, max_seq_len: int, max_batch_size: int):
        super().__init__()
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads

        if n_query_heads % n_kv_heads != 0:
            raise ValueError("n_query_heads must be divisible by n_kv_heads for GQA/MQA")

        self.n_rep = n_query_heads // n_kv_heads
        self.head_dim = d_model // n_query_heads

        self.wq = nn.Linear(d_model, n_query_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.rotary_encoder = RotaryEmbedding(self.head_dim, max_seq_len)


        self.register_buffer("cache_k", torch.zeros(
            (max_batch_size, max_seq_len, n_kv_heads, self.head_dim)
        ), persistent=False)
        self.register_buffer("cache_v", torch.zeros(
            (max_batch_size, max_seq_len, n_kv_heads, self.head_dim)
        ), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor], use_cache: bool = False):
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_query_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = self.rotary_encoder(q, start_pos)
        k = self.rotary_encoder(k, start_pos)

        if use_cache:
            # --- KV Cache Logic for INFERENCE ---
            self.cache_k[:B, start_pos: start_pos + T] = k.detach()
            self.cache_v[:B, start_pos: start_pos + T] = v.detach()

            keys = self.cache_k[:B, :start_pos + T]
            values = self.cache_v[:B, :start_pos + T]
        else:
            # --- No Cache Logic for TRAINING ---
            keys = k
            values = v

        keys = keys.repeat_interleave(self.n_rep, dim=2)
        values = values.repeat_interleave(self.n_rep, dim=2)

        q = q.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask[:, :, :T, :start_pos + T]

        attn_weights = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(attn_weights, values)

        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(output)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_kv_heads: int, ffn_hidden_dim: int, max_seq_len: int, max_batch_size: int):
        super().__init__()
        self.attention = Attention(d_model, n_head, n_kv_heads, max_seq_len, max_batch_size)
        self.feed_forward = FeedForward(dim=d_model, hidden_dim=ffn_hidden_dim)
        self.attention_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor], use_cache: bool = False):
        h = x + self.attention(self.attention_norm(x), start_pos, mask, use_cache)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out