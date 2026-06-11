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


@torch.no_grad()
def surprise_mask_vt(
    value_emb: torch.Tensor,     # [B, L, D]
    time_emb: torch.Tensor,      # [B, L, D]
    varis: torch.Tensor,         # [B, L]
    padding_mask: torch.Tensor,  # [B, L] True=pad
    sim_threshold: float,
    *,
    extra_pad_mask: torch.Tensor | None = None,  # ✅ NEW
    direction: str = "future",
    window_W: int = 0,
    adaptive: bool = False,
    adapt_alpha: float = 0.25,
    min_threshold: float = 0.50,
    normalize_by: str = "L",
    tau_v: float = 1.0,
    tau_t: float = 1.0,
    w_v: float = 1.0,
    w_t: float = 1.0,
    use_tf32: bool = True,
    gated_vars: List[int] | None = None,
    per_token_gate_mask: torch.Tensor | None = None,  # [B,L] bool
    invert: bool = False,
    padding_idx: int | None = None,
) -> torch.Tensor:
    """
    score = w_v*(cos(value)/tau_v) + w_t*(cos(time)/tau_t)
    valid tokens for gating exclude padding_mask and extra_pad_mask.
    """
    assert direction in ("past", "future")
    flip = (direction == "future")

    if extra_pad_mask is not None:
        padding_mask = padding_mask | extra_pad_mask

    if flip:
        value_emb    = torch.flip(value_emb, dims=[1])
        time_emb     = torch.flip(time_emb,  dims=[1])
        varis        = torch.flip(varis,     dims=[1])
        padding_mask = torch.flip(padding_mask, dims=[1])

    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    vnorm = F.normalize(value_emb, p=2, dim=-1)
    tnorm = F.normalize(time_emb,  p=2, dim=-1)
    sim_v = torch.bmm(vnorm, vnorm.transpose(1, 2))
    sim_t = torch.bmm(tnorm, tnorm.transpose(1, 2))
    score = (w_v * (sim_v / max(tau_v, 1e-6))) + (w_t * (sim_t / max(tau_t, 1e-6)))

    B, L = varis.shape
    device = varis.device

    valid = (~padding_mask)  # ✅ includes pretrain-masked positions
    same_var = varis.unsqueeze(2).eq(varis.unsqueeze(1))

    # gating scope (which target columns j can be gated)
    if per_token_gate_mask is not None:
        gate_col_mask = per_token_gate_mask.bool().to(device)
    elif gated_vars is not None:
        gv = torch.tensor(gated_vars, device=device, dtype=varis.dtype)
        gate_col_mask = torch.isin(varis, gv)
        if invert:
            gate_col_mask = ~gate_col_mask
    else:
        gate_col_mask = (varis != padding_idx) if padding_idx is not None else torch.ones_like(varis, dtype=torch.bool)

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
        # ✅ NOTE: score range depends on w/tau; clamp max should match your intended scale
        thr_vec = torch.clamp(thr_vec, min=min_threshold, max=1.999)
    else:
        thr_vec = torch.full((B, L), float(sim_threshold), device=device)

    thr_col = thr_vec.unsqueeze(1).expand(B, L, L)

    pair_core = (score >= thr_col) & same_var & valid_pair & tri
    pair_core = pair_core & gate_col_mask.unsqueeze(1)  # only gate target columns j in scope

    keep = torch.zeros((B, L), dtype=torch.bool, device=device)
    for j in range(L):
        blocked = (pair_core[:, :j, j] & keep[:, :j]).any(dim=1) if j > 0 else torch.zeros(B, dtype=torch.bool, device=device)
        keep[:, j] = valid[:, j] & (~blocked)

    red = valid & (~keep)
    red = red & gate_col_mask
    if flip:
        red = torch.flip(red, dims=[1])
    return red


@torch.no_grad()
def surprise_mask_vt_raw(
    value_emb: torch.Tensor,         # [B, L, D]
    time_raw: torch.Tensor,          # [B, L], normalized to [0,1]
    varis: torch.Tensor,             # [B, L]
    padding_mask: torch.Tensor,      # [B, L], True=pad
    sim_threshold: float,
    *,
    extra_pad_mask: torch.Tensor | None = None,  # ✅ NEW
    direction: str = "future",
    window_W: float = 0.0,           # in same scale as time_raw
    adaptive: bool = False,
    adapt_alpha: float = 0.25,
    min_threshold: float = 0.50,
    normalize_by: str = "L",
    tau_v: float = 1.0,
    tau_t: float = 1.0,              # unused (compat)
    w_v: float = 1.0,
    w_t: float = 1.0,                # unused (compat)
    use_tf32: bool = True,
    gated_vars: List[int] | None = None,
    per_token_gate_mask: torch.Tensor | None = None,
    invert: bool = False,
    padding_idx: int | None = None,
) -> torch.Tensor:
    """
    score(i,j) = cos(value_i, value_j)/tau_v * w_v * (1 - |dt|)
    valid tokens for gating exclude padding_mask and extra_pad_mask.
    """
    assert direction in ("past", "future")
    flip = (direction == "future")

    if extra_pad_mask is not None:
        padding_mask = padding_mask | extra_pad_mask

    if flip:
        value_emb    = torch.flip(value_emb, dims=[1])
        time_raw     = torch.flip(time_raw,  dims=[1])
        varis        = torch.flip(varis,     dims=[1])
        padding_mask = torch.flip(padding_mask, dims=[1])

    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    B, L = varis.shape
    device = varis.device

    valid = (~padding_mask)  # ✅ includes pretrain-masked positions

    # gate scope: which j can be gated
    if per_token_gate_mask is not None:
        gate_col_mask = per_token_gate_mask.bool().to(device)
    elif gated_vars is not None:
        gv = torch.tensor(gated_vars, device=device, dtype=varis.dtype)
        gate_col_mask = torch.isin(varis, gv)
        if invert:
            gate_col_mask = ~gate_col_mask
    else:
        gate_col_mask = (varis != padding_idx) if padding_idx is not None else torch.ones_like(varis, dtype=torch.bool)

    vnorm = F.normalize(value_emb, p=2, dim=-1)
    sim_v = torch.bmm(vnorm, vnorm.transpose(1, 2))

    if tau_v is not None and tau_v > 0:
        sim_v = sim_v / tau_v
    if w_v is not None:
        sim_v = sim_v * w_v

    same_var = varis.unsqueeze(2).eq(varis.unsqueeze(1))
    valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)

    time_raw = time_raw.to(device=device, dtype=torch.float32)
    dt = (time_raw.unsqueeze(1) - time_raw.unsqueeze(2)).abs()

    if window_W is not None and window_W > 0:
        within = (dt <= float(window_W))
    else:
        within = torch.ones_like(dt, dtype=torch.bool)

    time_gap = dt.clamp(0.0, 1.0)
    time_weight = 1.0 - time_gap

    score = sim_v * time_weight

    if adaptive:
        v = valid.float()
        past_cnt = torch.cumsum(v, dim=1) - v
        if normalize_by == "L":
            denom = float(L)
        else:
            denom = past_cnt.max(dim=1, keepdim=True).values.clamp_min(1.0)
        thr_vec = sim_threshold - adapt_alpha * (past_cnt / denom)
        thr_vec = torch.clamp(thr_vec, min=min_threshold, max=0.999)
    else:
        thr_vec = torch.full((B, L), float(sim_threshold), device=device)

    thr_col = thr_vec.unsqueeze(1).expand(B, L, L)

    pair_core = (score >= thr_col) & same_var & valid_pair & within
    pair_core = pair_core & gate_col_mask.unsqueeze(1)

    keep = torch.zeros((B, L), dtype=torch.bool, device=device)
    for j in range(L):
        blocked = (pair_core[:, :j, j] & keep[:, :j]).any(dim=1) if j > 0 else torch.zeros(B, dtype=torch.bool, device=device)
        keep[:, j] = valid[:, j] & (~blocked)

    red = valid & (~keep)
    red = red & gate_col_mask

    if flip:
        red = torch.flip(red, dims=[1])

    return red


# =========================
# Base backbone
# =========================

class STRaTSBase(nn.Module):
    """
    Shared backbone & call-pattern.
    Provides two encode paths:
      - encode_shared_value(): shared value embed (feature embedding table)
      - encode_per_feature_value(): per-feature value emb (ModuleList), VT masking
    All masks use True=pad. varis on pad are forced to padding_idx.
    """
    def __init__(
        self,
        num_features: int,
        embed_dim: int = 32,
        static_dim: int = 3,
        num_heads: int = 4,
        num_blocks: int = 2,
        ff_dim: int = 64,
        dropout: float = 0.2,
        time_activation: str = "relu",
        value_activation: str = "tanh",
        final_emb_weight: float = 0.5,
        use_surprise: bool = False,
        surprise_args: Optional[Dict[str, Any]] = None,
        vt_mode: bool = False,
        vt_mask_args: Optional[Dict[str, Any]] = None,
        use_timegap_surprise: bool = False,
        use_time2vec: bool = False,
        time2vec_periodic: str = "sin",
    ):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.static_dim = static_dim
        self.padding_idx = num_features
        self.final_emb_weight = float(final_emb_weight)

        self.vt_mode = bool(vt_mode)
        self.use_surprise = bool(use_surprise)
        self.use_timegap_surprise = bool(use_timegap_surprise)
        self.surprise_args = surprise_args or {}
        self.vt_mask_args = vt_mask_args or {}
        self.use_time2vec = bool(use_time2vec)

        # transformer stack
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])

        # time embedding (switchable)
        if self.use_time2vec:
            self.time_embed = Time2Vec(embed_dim, periodic=time2vec_periodic)
        else:
            self.time_embed = ContinuousValueEmbedding(1, embed_dim, activation=time_activation)
        self.ln_time = nn.LayerNorm(embed_dim)

        # value emb options
        if not self.vt_mode:
            # shared value emb
            self.value_embed = ContinuousValueEmbedding(1, embed_dim, activation=value_activation)
            self.ln_value = nn.LayerNorm(embed_dim)
            # feature id embedding (only in shared mode)
            self.feature_embed = nn.Embedding(num_features + 1, embed_dim, padding_idx=self.padding_idx)
            self.ln_feat = nn.LayerNorm(embed_dim)
        else:
            # per-feature value emb
            self.value_embeds = nn.ModuleList([
                ContinuousValueEmbedding(1, embed_dim, activation=value_activation)
                for _ in range(num_features)
            ])
            self.ln_value = nn.LayerNorm(embed_dim)

        # fusion & static/demo
        self.fusion = FusionAttention(embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.cls)
        self.demo = nn.Sequential(
            nn.Linear(static_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
        )

    # ---- internal helpers ----

    def _time(self, times):  # [B,L] -> [B,L,D]
        if self.use_time2vec:
            t_emb = self.time_embed(times)  # [B,L,D]
        else:
            t_emb = self.time_embed(times.unsqueeze(-1))  # [B,L,D]
        return self.ln_time(t_emb)

    def _value_shared(self, values):  # [B,L] -> [B,L,D]
        return self.ln_value(self.value_embed(values.unsqueeze(-1)))

    def _value_per_feature(self, values, varis):
        B, L = values.shape
        out = torch.zeros(
            B, L, self.embed_dim,
            device=values.device,
            dtype=values.dtype,
        )
        for k in range(self.num_features):
            m = (varis == k)
            if m.any():
                emb = self.value_embeds[k](values[m].unsqueeze(-1))
                emb = emb.to(out.dtype)
                out[m] = emb
        return self.ln_value(out)

    def _encode_transformer(self, tok_emb, pad_eff):  # tok_emb [B,L,D]; pad_eff [B,L]
        B = tok_emb.size(0)
        x = self.drop(tok_emb)
        x = torch.cat([self.cls.expand(B, 1, -1), x], dim=1)  # [B,L+1,D]

        cls_pad = torch.zeros((B, 1), dtype=torch.bool, device=pad_eff.device)
        att_mask = torch.cat([cls_pad, pad_eff], dim=1)       # True=pad

        for blk in self.blocks:
            x = blk(x, att_mask)

        # fusion weights (ignore pads)
        fus_keep = (~pad_eff).float()
        fus_mask = torch.cat(
            [torch.ones_like(cls_pad, dtype=torch.float32), fus_keep],
            dim=1
        )
        w = self.fusion(x, fus_mask)
        fused = (x * w.unsqueeze(-1)).sum(dim=1)
        return self.layer_norm(fused), x  # [B,D], [B,L+1,D]

    # ---- public encode entrypoints ----

    def encode_shared_value(
        self, times, varis, values, statics, padding_mask,
        *, pretrain: bool = False, pretrain_mask: Optional[torch.Tensor] = None
    ):
        pre_m = _ensure_bool_mask(pretrain_mask, values) if pretrain else None

        # pad var ids
        varis = varis.masked_fill(padding_mask, self.padding_idx)

        t = self._time(times)          # [B,L,D]
        v = self._value_shared(values) # [B,L,D]
        f = self.ln_feat(self.feature_embed(varis))

        # ✅ value만 가리기 (token은 남김)
        if pretrain:
            v = v * (~pre_m).float().unsqueeze(-1)

        triplet_base = t + v + f

        # ✅ attention pad mask는 '진짜 padding' + (optional) surprise redundancy만
        pad_eff = padding_mask

        if self.use_surprise:
            # ✅ surprise 계산에서 masked 토큰은 "비교 대상에서 제외" (valid에서 빠지게)
            # 즉, masked 위치는 padding처럼 취급하여 다른 토큰의 redundancy 판정에 끼지 않게 함
            padding_for_surprise = padding_mask | (pre_m if pretrain else torch.zeros_like(padding_mask))

            red = surprise_mask(
                triplet_base=triplet_base,
                varis=varis,
                padding_mask=padding_for_surprise,
                **self.surprise_args
            )

            # ✅ 하지만 masked 토큰 자체는 절대 redundant로 지우지 않게 강제
            if pretrain:
                red = red & (~pre_m)

            pad_eff = pad_eff | red

        # ✅ pretrain_mask는 pad_eff에 절대 넣지 않는다 (토큰 삭제 방지)
        fused, tokens = self._encode_transformer(triplet_base, pad_eff)
        demo = self.demo(statics)

        return {
            "triplet_base": triplet_base,
            "fused": fused,
            "tokens": tokens,
            "pad_eff": pad_eff,
            "demo": demo,
            "pretrain_mask": pre_m if pretrain else None,
        }

    def encode_per_feature_value(
        self, times, varis, values, statics, padding_mask,
        *, pretrain: bool = False, pretrain_mask: Optional[torch.Tensor] = None
    ):
        pre_m = _ensure_bool_mask(pretrain_mask, values) if pretrain else None

        t = self._time(times)
        v = self._value_per_feature(values, varis)

        # ✅ value만 가리기
        if pretrain:
            v = v * (~pre_m).float().unsqueeze(-1)

        pad_eff = padding_mask

        if self.use_surprise:
            vt_args = dict(self.vt_mask_args)
            vt_args.setdefault("padding_idx", self.padding_idx)

            padding_for_surprise = padding_mask | (pre_m if pretrain else torch.zeros_like(padding_mask))

            red = surprise_mask_vt(
                value_emb=v,
                time_emb=t,
                varis=varis,
                padding_mask=padding_for_surprise,
                **vt_args
            )

            if pretrain:
                red = red & (~pre_m)

            pad_eff = pad_eff | red

        if self.use_timegap_surprise:
            vt_args = dict(self.vt_mask_args)
            vt_args.setdefault("padding_idx", self.padding_idx)

            padding_for_surprise = padding_mask | (pre_m if pretrain else torch.zeros_like(padding_mask))

            red = surprise_mask_vt_raw(
                value_emb=v,
                time_raw=times,
                varis=varis,
                padding_mask=padding_for_surprise,
                **vt_args
            )

            if pretrain:
                red = red & (~pre_m)

            pad_eff = pad_eff | red

        tok_emb = t + v
        fused, tokens = self._encode_transformer(tok_emb, pad_eff)
        demo = self.demo(statics)

        return {
            "tok_emb": tok_emb,
            "fused": fused,
            "tokens": tokens,
            "pad_eff": pad_eff,
            "demo": demo,
            "pretrain_mask": pre_m if pretrain else None,
        }

def _ensure_bool_mask(mask: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.zeros_like(ref, dtype=torch.bool)
    return mask.bool()

# =========================
# Heads-wired models
# =========================
class STraTS(nn.Module):
    """Baseline (shared value emb), no surprise gating."""
    def __init__(
        self,
        num_features,
        embed_dim=32,
        static_dim=3,
        num_heads=4,
        num_blocks=2,
        ff_dim=64,
        dropout=0.2,
        time_activation='relu',
        value_activation='relu',
        final_emb_weight=0.5,
        n_output=1,
        use_time2vec: bool = False,
        time2vec_periodic: str = "sin",
    ):
        super().__init__()
        self.backbone = STRaTSBase(
            num_features=num_features,
            embed_dim=embed_dim,
            static_dim=static_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            ff_dim=ff_dim,
            dropout=dropout,
            time_activation=time_activation,
            value_activation=value_activation,
            final_emb_weight=final_emb_weight,
            use_surprise=False,
            vt_mode=False,
            use_time2vec=use_time2vec,
            time2vec_periodic=time2vec_periodic,
        )
        final_dim = embed_dim * 2
        self.forecast_head   = FrcstHead(final_dim, num_features + 1)
        self.downstream_head = FrcstHead(final_dim, n_output)

    def forward(self, times, varis, values, statics, padding_mask,
                *, pretrain=False, pretrain_mask=None):
        enc = self.backbone.encode_shared_value(
            times, varis, values, statics, padding_mask,
            pretrain=pretrain, pretrain_mask=pretrain_mask
        )
        fused, demo, pad_eff = enc["fused"], enc["demo"], enc["pad_eff"]

        if pretrain:
            base = enc["triplet_base"]
            seq = (1 - self.backbone.final_emb_weight) * base \
                  + self.backbone.final_emb_weight * fused.unsqueeze(1)
            demo_seq = demo.unsqueeze(1).expand(-1, seq.size(1), -1)
            tok = torch.cat([seq, demo_seq], dim=-1)          # [B,L,2D]
            logits = self.forecast_head(tok)                  # [B,L,F+1]
            pred_vals = torch.gather(
                logits, 2, varis.unsqueeze(-1)
            ).squeeze(-1)
            pm = _ensure_bool_mask(enc.get("pretrain_mask", None), values)
            return {
                "pred_vals": pred_vals,
                "values": values,
                "padding_mask": pad_eff,
                "pretrain_mask": pm,
            }
        else:
            final = torch.cat([fused, demo], dim=-1)          # [B,2D]
            pred = self.downstream_head(final).squeeze(-1)
            return {"pred": pred, "embs": final}


class SurpriseSTraTS(nn.Module):
    """Shared value emb + surprise gating (triplet cosine)."""
    def __init__(
        self,
        num_features,
        embed_dim=32,
        static_dim=3,
        num_heads=4,
        num_blocks=2,
        ff_dim=64,
        dropout=0.2,
        time_activation='relu',
        value_activation='relu',
        final_emb_weight=0.5,
        n_output=1,
        surprise_args: Optional[Dict[str, Any]] = None,
        use_surprise: bool = True,
        use_time2vec: bool = False,
        time2vec_periodic: str = "sin",
    ):
        super().__init__()
        self.backbone = STRaTSBase(
            num_features=num_features,
            embed_dim=embed_dim,
            static_dim=static_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            ff_dim=ff_dim,
            dropout=dropout,
            time_activation=time_activation,
            value_activation=value_activation,
            final_emb_weight=final_emb_weight,
            use_surprise=use_surprise,
            surprise_args=surprise_args or dict(sim_threshold=0.95, direction="future"),
            vt_mode=False,
            use_time2vec=use_time2vec,
            time2vec_periodic=time2vec_periodic,
        )
        final_dim = embed_dim * 2
        self.forecast_head   = FrcstHead(final_dim, num_features + 1)
        self.downstream_head = FrcstHead(final_dim, n_output)

    def forward(self, times, varis, values, statics, padding_mask,
                *, pretrain=False, pretrain_mask=None):
        enc = self.backbone.encode_shared_value(
            times, varis, values, statics, padding_mask,
            pretrain=pretrain, pretrain_mask=pretrain_mask
        )
        fused, demo, pad_eff = enc["fused"], enc["demo"], enc["pad_eff"]

        if pretrain:
            base = enc["triplet_base"]
            seq = (1 - self.backbone.final_emb_weight) * base \
                  + self.backbone.final_emb_weight * fused.unsqueeze(1)
            demo_seq = demo.unsqueeze(1).expand(-1, seq.size(1), -1)
            tok = torch.cat([seq, demo_seq], dim=-1)
            logits = self.forecast_head(tok)
            pred_vals = torch.gather(
                logits, 2, varis.unsqueeze(-1)
            ).squeeze(-1)
            pm = _ensure_bool_mask(enc.get("pretrain_mask", None), values)
            return {
                "pred_vals": pred_vals,
                "values": values,
                "padding_mask": pad_eff,
                "pretrain_mask": pm,
            }
        else:
            final = torch.cat([fused, demo], dim=-1)
            pred = self.downstream_head(final).squeeze(-1)
            return {"pred": pred, "embs": final}

    @torch.no_grad()
    def check_padding(self,
                      times: torch.Tensor,
                      varis: torch.Tensor,
                      values: torch.Tensor,
                      padding_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Debug helper for 'triplet' surprise gating.
        Returns the effective redundant mask.
        """
        # pad var ids for stability
        varis = varis.masked_fill(padding_mask, self.backbone.padding_idx)

        t = self.backbone._time(times)          # [B,L,D]
        v = self.backbone._value_shared(values) # [B,L,D]
        f = self.backbone.ln_feat(self.backbone.feature_embed(varis))
        triplet_base = t + v + f

        sargs = dict(self.backbone.surprise_args)
        redundant_mask = surprise_mask(
            triplet_base=triplet_base,
            varis=varis,
            padding_mask=padding_mask,
            **sargs
        )  # [B,L] bool

        return {
            "times": times,
            "varis": varis,
            "values": values,
            "mask": redundant_mask,
        }
    

    
class SurpriseSTraTS_VT(nn.Module):
    """Per-feature value emb + surprise gating with value+time similarity."""
    def __init__(
        self,
        num_features,
        embed_dim=32,
        static_dim=3,
        num_heads=4,
        num_blocks=2,
        ff_dim=64,
        dropout=0.2,
        time_activation='relu',
        value_activation='relu',
        final_emb_weight=0.5,
        n_output=1,
        vt_mask_args: Optional[Dict[str, Any]] = None,
        use_surprise: bool = True,
        use_timegap_surprise: bool = False,
        use_time2vec: bool = False,
        time2vec_periodic: str = "sin",
    ):
        super().__init__()
        self.backbone = STRaTSBase(
            num_features=num_features,
            embed_dim=embed_dim,
            static_dim=static_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            ff_dim=ff_dim,
            dropout=dropout,
            time_activation=time_activation,
            value_activation=value_activation,
            final_emb_weight=final_emb_weight,
            use_surprise=use_surprise,
            vt_mode=True,
            vt_mask_args=vt_mask_args or dict(
                sim_threshold=0.95,
                direction="future",
                tau_v=1.0,
                tau_t=1.0,
                w_v=1.0,
                w_t=1.0,
                window_W=0,
                adaptive=False,
                adapt_alpha=0.25,
                min_threshold=0.55,
                normalize_by="L",
                use_tf32=True,
                gated_vars=None,
                per_token_gate_mask=None,
                invert=False,
            ),
            use_timegap_surprise=use_timegap_surprise,
            use_time2vec=use_time2vec,
            time2vec_periodic=time2vec_periodic,
        )
        final_dim = embed_dim * 2
        self.forecast_head   = FrcstHead(final_dim, num_features + 1)
        self.downstream_head = FrcstHead(final_dim, n_output)

    def forward(self, times, varis, values, statics, padding_mask,
                *, pretrain=False, pretrain_mask=None):
        enc = self.backbone.encode_per_feature_value(
            times, varis, values, statics, padding_mask,
            pretrain=pretrain, pretrain_mask=pretrain_mask
        )
        fused, demo, pad_eff = enc["fused"], enc["demo"], enc["pad_eff"]

        if pretrain:
            tok = enc["tok_emb"]  # [B,L,D] = time+value
            seq = (1 - self.backbone.final_emb_weight) * tok \
                  + self.backbone.final_emb_weight * fused.unsqueeze(1)
            demo_seq = demo.unsqueeze(1).expand(-1, seq.size(1), -1)
            tok2 = torch.cat([seq, demo_seq], dim=-1)  # [B,L,2D]
            logits = self.forecast_head(tok2)
            pred_vals = torch.gather(
                logits, 2, varis.unsqueeze(-1)
            ).squeeze(-1)
            pm = _ensure_bool_mask(enc.get("pretrain_mask", None), values)
            return {
                "pred_vals": pred_vals,
                "values": values,
                "padding_mask": pad_eff,
                "pretrain_mask": pm,
            }
        else:
            final = torch.cat([fused, demo], dim=-1)
            pred = self.downstream_head(final).squeeze(-1)
            return {"pred": pred, "embs": final}

    @torch.no_grad()
    def check_padding(self,
                      times: torch.Tensor,
                      varis: torch.Tensor,
                      values: torch.Tensor,
                      padding_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Debug helper for VT surprise gating (value + time).
        """
        t = self.backbone._time(times)                      # [B,L,D]
        v = self.backbone._value_per_feature(values, varis) # [B,L,D]

        vt_args = dict(self.backbone.vt_mask_args)
        vt_args.setdefault("padding_idx", self.backbone.padding_idx)

        if self.backbone.use_surprise:
            redundant_mask = surprise_mask_vt(
                value_emb=v,
                time_emb=t,
                varis=varis,
                padding_mask=padding_mask,
                **vt_args
            )
        elif self.backbone.use_timegap_surprise:
            redundant_mask = surprise_mask_vt_raw(
                value_emb=v,
                time_raw=times,
                varis=varis,
                padding_mask=padding_mask,
                **vt_args
            )
        else:
            redundant_mask = torch.zeros_like(padding_mask, dtype=torch.bool)

        return {
            "times": times,
            "varis": varis,
            "values": values,
            "mask": redundant_mask,
        }
