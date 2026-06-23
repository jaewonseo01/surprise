# -*- coding: utf-8 -*-
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Building blocks (as before)
# =========================
class ContinuousValueEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, activation='tanh'):
        super().__init__()
        self.hidden_dim = int(embed_dim ** 0.5)
        self.lin1 = nn.Linear(input_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, embed_dim)
        self.activation = torch.tanh if activation == 'tanh' else nn.ReLU()
    def forward(self, x):
        x = self.lin1(x)
        x = self.activation(x)
        return self.lin2(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, padding_mask: torch.Tensor):
        y, _ = self.attn(x, x, x, key_padding_mask=padding_mask)  # True=pad
        x = self.norm1(x + self.drop(y))
        y = self.ffn(x)
        return self.norm2(x + self.drop(y))

class FusionAttention(nn.Module):
    """mask: float [B,L] with 1=keep, 0=mask"""
    def __init__(self, embed_dim, eps: float = 1e-6):
        super().__init__()
        self.W = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.b = nn.Parameter(torch.zeros(embed_dim))
        self.u = nn.Parameter(torch.empty(embed_dim, 1))
        self.eps = eps
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.u)
    def forward(self, x, mask):
        att = torch.tanh(torch.matmul(x, self.W) + self.b)
        s = torch.matmul(att, self.u).squeeze(-1)
        s = s + (1 - mask) * torch.finfo(s.dtype).min
        s = s - s.max(dim=-1, keepdim=True)[0]
        exp_s = torch.exp(s) * mask
        denom = exp_s.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        return exp_s / denom

class CLSHead(nn.Module):
    def __init__(self, embed_dim): super().__init__(); self.fc = nn.Linear(embed_dim, embed_dim); self.act = nn.Tanh()
    def forward(self, x): return self.act(self.fc(x))

class FrcstHead(nn.Module):
    def __init__(self, embed_dim, output_dim): super().__init__(); self.l1=nn.Linear(embed_dim, embed_dim); self.a=nn.ReLU(); self.l2=nn.Linear(embed_dim, output_dim)
    def forward(self, x): return self.l2(self.a(self.l1(x)))

class Time2Vec(nn.Module):
    """
    Time2Vec for scalar time input.
    Input:  t [B, L] or [B, L, 1]
    Output: [B, L, out_dim]
        - 1 linear term
        - out_dim-1 periodic terms
    """
    def __init__(self, out_dim: int, periodic: str = "sin"):
        super().__init__()
        if out_dim < 2:
            raise ValueError("Time2Vec out_dim must be at least 2 (1 linear + >=1 periodic).")

        self.out_dim = out_dim
        self.k = out_dim - 1  # num periodic components

        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))

        self.w = nn.Parameter(torch.randn(self.k))
        self.b = nn.Parameter(torch.randn(self.k))

        if periodic == "sin":
            self.periodic_fn = torch.sin
        elif periodic == "cos":
            self.periodic_fn = torch.cos
        else:
            raise ValueError(f"Unsupported periodic fn: {periodic}")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B,L] or [B,L,1]
        if t.dim() == 3 and t.size(-1) == 1:
            t = t.squeeze(-1)  # [B,L]
        elif t.dim() != 2:
            raise ValueError(f"Time2Vec expects [B,L] or [B,L,1], got {t.shape}")

        t = t.unsqueeze(-1)           # [B,L,1]
        v0 = self.w0 * t + self.b0   # [B,L,1]
        vp = self.periodic_fn(self.w * t + self.b)  # [B,L,k]
        return torch.cat([v0, vp], dim=-1)          # [B,L,out_dim]

# Components for Trajectory based 
def _max_vertical_error_to_segment(t, y, i0, i1):
    """구간 [i0,i1]의 선분에 대해, 내부 점들의 최대 수직오차와 argmax 인덱스 반환."""
    if i1 - i0 <= 1:
        return 0.0, -1  # 내부점 없음
    t0, t1 = t[i0], t[i1]
    y0, y1 = y[i0], y[i1]
    dt = t1 - t0
    if dt.abs() < 1e-12:
        # 시간축이 같으면 수직선 → 그냥 y 편차의 max
        vals = (y[i0+1:i1] - y0).abs()
        if vals.numel() == 0:
            return 0.0, -1
        e, j = vals.max(dim=0)
        return e.item(), (i0 + 1 + j.item())
    # 직선 예측값
    slope = (y1 - y0) / dt
    y_hat = y0 + slope * (t[i0+1:i1] - t0)
    err = (y[i0+1:i1] - y_hat).abs()
    if err.numel() == 0:
        return 0.0, -1
    e, j = err.max(dim=0)
    return e.item(), (i0 + 1 + j.item())

def rdp_1d_vertical(t: torch.Tensor, y: torch.Tensor, eps: float):
    """
    RDP(수직편차 기준)로 인덱스 선택. t,y: 1D, 동일 길이, t는 증가 정렬 가정.
    반환: keep_idx(sorted list), keep_mask(Bool)
    """
    L = t.numel()
    if L <= 2:
        keep_idx = list(range(L))
        keep_mask = torch.zeros(L, dtype=torch.bool)
        keep_mask[keep_idx] = True
        return keep_idx, keep_mask

    keep = {0, L-1}
    stack = [(0, L-1)]
    while stack:
        i0, i1 = stack.pop()
        e, j = _max_vertical_error_to_segment(t, y, i0, i1)
        if e > eps and j != -1:
            # 분할
            stack.append((i0, j))
            stack.append((j, i1))
            keep.add(j)
        # else: 이 구간은 하나의 선분으로 충분 (i0, i1만 유지)
    keep_idx = sorted(keep)
    keep_mask = torch.zeros(L, dtype=torch.bool)
    keep_mask[keep_idx] = True
    return keep_idx, keep_mask


# =========================
# Surprise masks (two flavors)
# =========================


@torch.no_grad()
def surprise_mask(
    triplet_base: torch.Tensor,   # [B,L,D]
    varis: torch.Tensor,          # [B,L] (int64)
    padding_mask: torch.Tensor,   # [B,L] (bool, True=pad)
    sim_threshold: float,
    *,
    extra_pad_mask: torch.Tensor | None = None,  # ✅ NEW: treat these as pad during gating
    direction: str = "future",
    window_W: int = 0,
    adaptive: bool = False,
    adapt_alpha: float = 0.25,
    min_threshold: float = 0.50,
    normalize_by: str = "L",
    detach_embeddings: bool = True,
    use_tf32: bool = True,
) -> torch.Tensor:
    """
    Returns:
        red: [B, L] bool, True = redundant (mask out).
    NOTE:
        padding_mask is used to define "valid" tokens for gating.
        extra_pad_mask (e.g., pretrain_mask) is additionally treated as pad.
    """
    assert direction in ("past", "future")
    flip = (direction == "future")

    # ✅ combine padding masks BEFORE any logic
    if extra_pad_mask is not None:
        padding_mask = padding_mask | extra_pad_mask

    if flip:
        triplet_base = torch.flip(triplet_base, dims=[1])
        varis        = torch.flip(varis,        dims=[1])
        padding_mask = torch.flip(padding_mask, dims=[1])

    tb = triplet_base.detach() if detach_embeddings else triplet_base
    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    B, L, D = tb.shape
    device = tb.device

    norm = F.normalize(tb, p=2, dim=-1)
    sim = torch.bmm(norm, norm.transpose(1,2))  # [B,L,L]

    valid = (~padding_mask)  # ✅ includes pretrain-masked positions as invalid
    same_var = varis.unsqueeze(2).eq(varis.unsqueeze(1))

    idx = torch.arange(L, device=device)
    dist = idx[None, :] - idx[:, None]
    tri = (dist > 0) if (window_W == 0 or window_W is None) else ((dist > 0) & (dist <= window_W))
    tri = tri.unsqueeze(0)

    valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)

    if adaptive:
        v = valid.float()
        past_cnt = torch.cumsum(v, dim=1) - v
        denom = float(L) if normalize_by == "L" else past_cnt.max(dim=1, keepdim=True).values.clamp_min(1.0)
        thr_vec = sim_threshold - adapt_alpha * (past_cnt / denom)
        thr_vec = torch.clamp(thr_vec, min=min_threshold, max=0.999)
    else:
        thr_vec = torch.full((B, L), float(sim_threshold), device=device)

    thr_col = thr_vec.unsqueeze(1).expand(B, L, L)

    pair_core = (sim >= thr_col) & same_var & valid_pair & tri

    keep = torch.zeros((B, L), dtype=torch.bool, device=device)
    for j in range(L):
        blocked = (pair_core[:, :j, j] & keep[:, :j]).any(dim=1) if j > 0 else torch.zeros(B, dtype=torch.bool, device=device)
        keep[:, j] = valid[:, j] & (~blocked)

    red = valid & (~keep)
    if flip:
        red = torch.flip(red, dims=[1])
    return red  # [B,L] bool
