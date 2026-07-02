import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from icecream import ic


class RoPE(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.dim = d
        self.base = base
        self.cos_cached = None
        self.sin_cached = None

    # def _build_cache(self, x: torch.Tensor):
    #     if (
    #         self.cos_cached is not None
    #         and self.cos_cached.shape[1] >= x.shape[2]
    #     ):
    #         return

    #     seq_len = x.shape[2]
    #     theta = (
    #         1. / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
    #     ).to(x.device)
    #     seq_idx = torch.arange(0, seq_len).float().to(x.device)
    #     idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
    #     idx_theta = torch.cat([idx_theta, idx_theta], dim=1)
    #     self.cos_cached = idx_theta.cos()[None]
    #     self.sin_cached = idx_theta.sin()[None]

    def _neg_half(self, x: torch.Tensor):
        d_2 = self.dim // 2
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

    # def forward(self, x, cache_len=0):
    #     """
    #     x: (B, H, T, D)
    #     cache_len: position offset of the first token in x (0 for a fresh
    #                prompt, or how many tokens are already cached during
    #                incremental decoding)
    #     """
    #     self._build_cache(x)
    #     seq_len = x.shape[2]

    #     # cache_len > 0 — e.g. cache_len=5, seq_len=1 (a single new token)
    #     # gives the empty slice cos_cached[:, :, 5:1, :]. We need the
    #     # *absolute* end position, not seq_len (which is just the length of
    #     # the current chunk, not the cumulative position).
    #     ic(cache_len)
    #     ic(cache_len + seq_len)
    #     cos = self.cos_cached[:, None, cache_len:cache_len + seq_len, :]
    #     sin = self.sin_cached[:, None, cache_len:cache_len + seq_len, :]

    #     return x * cos + self._neg_half(x) * sin
    def _build_cache(self, x: torch.Tensor, cache_len: int = 0):
        seq_len = x.shape[2]
        required_len = cache_len + seq_len  # the highest absolute position we need + 1

        if (
            self.cos_cached is not None
            and self.cos_cached.shape[1] >= required_len
        ):
            return

        theta = (
            1. / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        ).to(x.device)
        seq_idx = torch.arange(0, required_len).float().to(x.device)  # build up to required_len, not just seq_len
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
        idx_theta = torch.cat([idx_theta, idx_theta], dim=1)
        self.cos_cached = idx_theta.cos()[None]
        self.sin_cached = idx_theta.sin()[None]

    def forward(self, x, cache_len=0):
        self._build_cache(x, cache_len=cache_len)
        seq_len = x.shape[2]
        cos = self.cos_cached[:, None, cache_len:cache_len + seq_len, :]
        sin = self.sin_cached[:, None, cache_len:cache_len + seq_len, :]
        return x * cos + self._neg_half(x) * sin


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.rope = RoPE(self.head_dim)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _project(self, x, cache_len):
        
        B, T, _ = x.shape
        qkv = self.qkv(x)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # ic(f"before rope {Q.shape}, {K.shape}")
        Q = self.rope(Q, cache_len=cache_len)
        K = self.rope(K, cache_len=cache_len)
        # ic(f"after rope {Q.shape}, {K.shape}")

        return Q, K, V

    def inference(self, x, kv, cache_len):

        B, T, _ = x.shape
        # ic(x.shape)

        if kv is None:
            Q, K, V = self._project(x, cache_len=0)
            causal_mask = torch.tril(
                torch.ones(T, T, dtype=torch.bool, device=x.device)
            )
            scores = (Q @ K.transpose(-2, -1)) * (self.head_dim ** -0.5)
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        else:
            K_cache, V_cache = kv
            Q, K_new, V_new = self._project(x, cache_len=cache_len)
            K = torch.cat([K_cache, K_new], dim=2)  # cat along seq dim, not head dim
            V = torch.cat([V_cache, V_new], dim=2)
            scores = (Q @ K.transpose(-2, -1)) * (self.head_dim ** -0.5)

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        # ic(Q.shape)
        # ic(K.shape)
        # ic(V.shape)
        # ic(scores.shape)
        # ic(attn.shape)
        out = attn @ V
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        out = self.W_o(out)


        return out, (K, V)  # CHANGE: return as a tuple, never concatenated

    def forward(self, x, attention_mask, inference=False, **kwargs):
        if inference:
            return self.inference(x, kv=kwargs.get('kv'), cache_len=kwargs.get('cache_len', 0))

        B, T, _ = x.shape
        Q, K, V = self._project(x, cache_len=0)

        scores = Q @ K.transpose(-2, -1) * (self.head_dim ** -0.5)
        causal_mask = torch.tril(
            torch.ones(T, T, dtype=torch.bool, device=x.device)
        ).unsqueeze(0).unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()
        mask = causal_mask & attention_mask
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = attn @ V
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        out = self.W_o(out)

        return out, None