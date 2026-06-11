# =========================
# STRaTS Lightning Pipeline
# =========================
from __future__ import annotations

from dataclasses import dataclass, asdict

import os, json, math, random
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Literal, Sequence, Any, Union
from tqdm import tqdm

import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics
from torch.utils.data import Dataset, DataLoader, Sampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import Callback


from models_new import STraTS, SurpriseSTraTS, SurpriseSTraTS_VT
from runner_new import prepare_dataset_samples, normalize_many, normalize_single_samples,TimeSeriesDataset, make_dataloader, make_dataloader_random, build_ett_windows_as_samples, TimeSeriesDatasetETT
from torch import Tensor

# metrics
from torchmetrics.classification import (
    BinaryAUROC, BinaryAveragePrecision, BinaryAccuracy
)

# viz
import matplotlib.pyplot as plt
try:
    from pacmap import PaCMAP
    _HAS_PACMAP = True
except Exception:
    _HAS_PACMAP = False

# wandb
import wandb
torch.set_float32_matmul_precision('high')

# =========================
# P12/P19/PAM loaders (PhysioNet-style)
# =========================
from typing import Optional, Literal
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

def _pick_metrics(d: dict, keys: list[str]) -> dict:
    out = {}
    for k in keys:
        if k in d:
            out[k] = float(d[k])
    return out

def _mean_std(rows: list[dict], keys: list[str]) -> dict:
    import numpy as np
    res = {}
    for k in keys:
        vals = [r.get(k, None) for r in rows]
        vals = [v for v in vals if v is not None]
        if len(vals) == 0:
            continue
        res[f"{k}_mean"] = float(np.mean(vals))
        res[f"{k}_std"]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return res


def get_data_split(base_path, split_path, split_type='random', reverse=False, baseline=True,
                   dataset='P12', predictive_label='mortality'):
    if dataset == 'P12':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = ''
    elif dataset == 'P19':
        Pdict_list = np.load(base_path + '/processed_data/PT_dict_list_6.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes_6.npy', allow_pickle=True)
        dataset_prefix = 'P19_'
    elif dataset == 'PAM':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = ''
    else:
        raise ValueError(f"Unknown dataset={dataset}")

    if baseline is True:
        BL_path = ''
    else:
        BL_path = 'baselines/'

    if split_type == 'random':
        idx_train, idx_val, idx_test = np.load(base_path + split_path, allow_pickle=True)
    elif split_type == 'age':
        if reverse is False:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_under_65.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_over_65.npy', allow_pickle=True)
        else:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_over_65.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_under_65.npy', allow_pickle=True)
        np.random.shuffle(idx_vt)
        idx_val = idx_vt[:round(len(idx_vt) / 2)]
        idx_test = idx_vt[round(len(idx_vt) / 2):]
    elif split_type == 'gender':
        if reverse is False:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_male.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_female.npy', allow_pickle=True)
        else:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_female.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_male.npy', allow_pickle=True)
        np.random.shuffle(idx_vt)
        idx_val = idx_vt[:round(len(idx_vt) / 2)]
        idx_test = idx_vt[round(len(idx_vt) / 2):]
    else:
        raise ValueError(f"Unknown split_type={split_type}")

    Ptrain = Pdict_list[idx_train]
    Pval   = Pdict_list[idx_val]
    Ptest  = Pdict_list[idx_test]

    if predictive_label == 'mortality':
        y = arr_outcomes[:, -1].reshape((-1, 1))
    elif predictive_label == 'LoS':
        y = arr_outcomes[:, 3].reshape((-1, 1))
        y = np.array(list(map(lambda los: 0 if los <= 3 else 1, y)))[..., np.newaxis]
    else:
        raise ValueError(f"Unknown predictive_label={predictive_label}")

    ytrain = y[idx_train]; yval = y[idx_val]; ytest = y[idx_test]
    return Ptrain, Pval, Ptest, ytrain, yval, ytest


def getStats(P_tensor):
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    mf = np.zeros((F, 1))
    stdf = np.ones((F, 1))
    eps = 1e-7
    for f in range(F):
        vals_f = Pf[f, :]
        vals_f = vals_f[vals_f > 0]
        mf[f] = np.mean(vals_f) if vals_f.size > 0 else 0.0
        stdf[f] = np.std(vals_f) if vals_f.size > 0 else 1.0
        stdf[f] = np.maximum(stdf[f], eps)
    return mf, stdf


def mask_normalize(P_tensor, mf, stdf, lengths):
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    M = (P_tensor > 0).astype(np.float32)
    M_3D = M.transpose((2, 0, 1)).reshape(F, -1)

    M_null = np.zeros((N, T, F), dtype=np.float32)
    for i in range(N):
        length = int(lengths[i])
        M_null[i, :length, :] = 1
        M_null[i, :length, :][P_tensor[i, :length, :] == 0] = np.nan

    for f in range(F):
        Pf[f] = (Pf[f] - mf[f]) / (stdf[f] + 1e-18)
    Pf = Pf * M_3D

    Pnorm_tensor = Pf.reshape((F, N, T)).transpose((1, 2, 0))
    Pfinal_tensor = np.concatenate([Pnorm_tensor, M], axis=2)  # [N,T,2F]
    return Pfinal_tensor, M_null


def getStats_static(P_tensor, dataset='P12'):
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))
    ms = np.zeros((S, 1))
    ss = np.ones((S, 1))

    if dataset == 'P12':
        bool_categorical = [0, 1, 1, 0, 1, 1, 1, 1, 0]
    elif dataset == 'P19':
        bool_categorical = [0, 1, 0, 0, 0, 0]
    else:
        # PAM static schema가 다르면 여길 수정해야 함
        bool_categorical = [0] * S

    for s in range(S):
        if bool_categorical[s] == 0:
            vals_s = Ps[s, :]
            vals_s = vals_s[vals_s > 0]
            ms[s] = np.mean(vals_s) if vals_s.size > 0 else 0.0
            ss[s] = np.std(vals_s) if vals_s.size > 0 else 1.0
    return ms, ss


def mask_normalize_static(P_tensor, ms, ss):
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))
    for s in range(S):
        Ps[s] = (Ps[s] - ms[s]) / (ss[s] + 1e-18)
    for s in range(S):
        idx_missing = np.where(Ps[s, :] <= 0)
        Ps[s, idx_missing] = 0
    return Ps.reshape((S, N)).transpose((1, 0))


def tensorize_normalize(P, y, mf, stdf, ms, ss):
    T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])

    P_tensor = np.zeros((len(P), T, F), dtype=np.float32)
    P_time = np.zeros((len(P), T, 1), dtype=np.float32)
    P_static_tensor = np.zeros((len(P), D), dtype=np.float32)
    lengths = np.zeros(len(P), dtype=int)

    for i in range(len(P)):
        P_tensor[i] = P[i]['arr']
        P_time[i] = P[i]['time']
        P_static_tensor[i] = P[i]['extended_static']
        lengths[i] = int(P[i]['length'])

    P_tensor, M_null = mask_normalize(P_tensor, mf, stdf, lengths)
    P_tensor = torch.tensor(P_tensor, dtype=torch.float32)              # [N,T,2F]
    M_null = torch.tensor(M_null, dtype=torch.float32)                  # [N,T,F]

    P_time = torch.tensor(P_time, dtype=torch.float32) / 60.0           # hours
    P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
    P_static_tensor = torch.tensor(P_static_tensor, dtype=torch.float32)

    # BCEWithLogitsLoss wants float targets; keep as float later
    y_tensor = torch.tensor(y[:, 0], dtype=torch.float32).view(-1, 1)   # [N,1]
    return P_tensor, P_static_tensor, P_time, y_tensor, M_null


def dense_with_mask_to_tokens(
    x_t2f: torch.Tensor,   # [T, 2F]
    t_t: torch.Tensor,     # [T]
    *,
    num_features: int,     # F
    time_minmax_norm: bool = False,
    time_max: int = 60,
    max_tokens: Optional[int] = None,
):
    T, twoF = x_t2f.shape
    assert twoF == 2 * num_features, f"Expected 2F={2*num_features}, got {twoF}"

    x_val = x_t2f[:, :num_features]     # [T,F]
    x_msk = x_t2f[:, num_features:]     # [T,F]

    obs = (x_msk > 0)
    idx = obs.nonzero(as_tuple=False)   # [L,2]
    if idx.numel() == 0:
        return (torch.zeros((0,), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
                torch.zeros((0,), dtype=torch.float32))

    t_idx = idx[:, 0]
    f_idx = idx[:, 1]
    times = t_t[t_idx].float()
    values = x_val[t_idx, f_idx].float()
    varis = f_idx.long()

    if time_minmax_norm:
        tmin = float(times.min()); tmax = float(times.max())
        times = (times - tmin) / (tmax - tmin) if tmax > tmin else torch.zeros_like(times)
    else:
        times = times / time_max

    sort_key = times * (num_features + 1) + varis.float() / (num_features + 1)
    order = torch.argsort(sort_key)
    times, varis, values = times[order], varis[order], values[order]

    if max_tokens is not None and times.numel() > max_tokens:
        times, varis, values = times[:max_tokens], varis[:max_tokens], values[:max_tokens]

    return times, varis, values


class P12P19_STRaTSDataset(Dataset):
    def __init__(self, X_t2f, Time_t, y, static, M_null, *,
                 num_features: int, max_tokens: int = 4096, time_minmax_norm: bool = False, time_max:int=60, pid_offset: int = 0):
        self.X = X_t2f
        self.Time = Time_t.squeeze(-1)  # [N,T]
        self.y = y                      # [N,1] float
        self.static = static            # [N,D]
        self.num_features = num_features
        self.max_tokens = max_tokens
        self.time_minmax_norm = time_minmax_norm
        self.time_max = time_max
        self.pid_offset = pid_offset

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, i: int):
        x = self.X[i]                 # [T,2F]
        t = self.Time[i]              # [T]
        y = self.y[i]                 # [1]
        pid = torch.tensor(i + self.pid_offset, dtype=torch.long)
        static = self.static[i]

        times, varis, values = dense_with_mask_to_tokens(
            x, t, num_features=self.num_features,
            time_minmax_norm=self.time_minmax_norm,
            time_max = self.time_max,
            max_tokens=self.max_tokens
        )

        pre_mask = torch.zeros_like(times, dtype=torch.bool)
        padding_mask = torch.zeros_like(times, dtype=torch.bool)
        return times, varis, values, padding_mask, pre_mask, y, pid, static




def make_strats_collate(padding_idx: int):
    def collate(batch):
        times_list, varis_list, values_list, _, pre_list, y_list, pid_list, static_list = zip(*batch)
        B = len(batch)
        lens = torch.tensor([t.numel() for t in times_list], dtype=torch.long)
        Lmax = int(lens.max().item()) if B > 0 else 0

        times = torch.zeros((B, Lmax), dtype=torch.float32)
        values = torch.zeros((B, Lmax), dtype=torch.float32)
        varis = torch.full((B, Lmax), padding_idx, dtype=torch.long)

        padding_mask = torch.ones((B, Lmax), dtype=torch.bool)  # True=pad
        pre_mask = torch.ones((B, Lmax), dtype=torch.bool)

        for i, (t, v, va, pm) in enumerate(zip(times_list, values_list, varis_list, pre_list)):
            L = t.numel()
            if L == 0:
                continue
            times[i, :L] = t
            values[i, :L] = v
            varis[i, :L] = va
            padding_mask[i, :L] = False
            pre_mask[i, :L] = pm

        # y_list: each [1] -> stack -> [B,1]
        y = torch.stack([yy.view(1) for yy in y_list], dim=0)
        pid = torch.stack(list(pid_list), dim=0)
        static = torch.stack(list(static_list), dim=0)

        return times, varis, values, padding_mask, pre_mask, y, pid, static
    return collate

def make_strats_collate_pretrain(*, padding_idx: int, pre_mask_p: float = 0.15):
    base_collate = make_strats_collate(padding_idx=padding_idx)

    def collate(batch):
        # base returns 8 items
        times, varis, values, pad_mask, base_pre_mask, y, pid, static = base_collate(batch)

        # pad 제외하고 Bernoulli로 새 pre_mask 생성
        valid = ~pad_mask
        pre_mask = (torch.rand_like(times, dtype=torch.float32) < pre_mask_p) & valid

        # ✅ PretrainLit이 기대하는 순서 그대로:
        # (times, varis, values, pad, pre, y, pid, static)
        return times, varis, values, pad_mask, pre_mask, y, pid, static

    return collate

class P12P19DataModule(pl.LightningDataModule):
    def __init__(self, *, dataset: Literal["P12","P19","PAM"], base_path: str, split_path: str,
                 split_type: str = "random", reverse: bool = False, baseline: bool = False,
                 predictive_label: str = "mortality",
                 batch_size: int = 128, num_workers: int = 4, max_tokens: int = 4096):
        super().__init__()
        self.dataset = dataset
        self.base_path = base_path
        self.split_path = split_path
        self.split_type = split_type
        self.reverse = reverse
        self.baseline = baseline
        self.predictive_label = predictive_label
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_tokens = max_tokens

    def setup(self, stage=None):
        Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(
            self.base_path, self.split_path,
            split_type=self.split_type,
            reverse=self.reverse,
            baseline=self.baseline,
            dataset=self.dataset,
            predictive_label=self.predictive_label
        )

        T, F = Ptrain[0]["arr"].shape
        D = len(Ptrain[0]["extended_static"])

        Ptrain_tensor = np.zeros((len(Ptrain), T, F), dtype=np.float32)
        Ptrain_static_tensor = np.zeros((len(Ptrain), D), dtype=np.float32)
        for i in range(len(Ptrain)):
            Ptrain_tensor[i] = Ptrain[i]["arr"]
            Ptrain_static_tensor[i] = Ptrain[i]["extended_static"]

        mf, stdf = getStats(Ptrain_tensor)
        ms, ss = getStats_static(Ptrain_static_tensor, dataset=self.dataset)

        Xtr, Str, Ttr, ytr, Mtr = tensorize_normalize(Ptrain, ytrain, mf, stdf, ms, ss)
        Xva, Sva, Tva, yva, Mva = tensorize_normalize(Pval,   yval,   mf, stdf, ms, ss)
        Xte, Ste, Tte, yte, Mte = tensorize_normalize(Ptest,  ytest,  mf, stdf, ms, ss)

        self.num_features = F  # padding_idx = num_features

        cfg.model_cfg["static_dim"] = D
        if self.dataset == "P12":
            time_max = 2880 / 60 # Tensorize_normalize already turns to hours (divide by 60)
        elif self.dataset == "P19":
            time_max = 60 / 60 
        self.train_ds = P12P19_STRaTSDataset(Xtr, Ttr, ytr, Str, Mtr, num_features=F, max_tokens=self.max_tokens,
                                             time_max=time_max)
        self.val_ds   = P12P19_STRaTSDataset(Xva, Tva, yva, Sva, Mva, num_features=F, max_tokens=self.max_tokens,
                                             time_max=time_max, pid_offset=len(self.train_ds))
        self.test_ds  = P12P19_STRaTSDataset(Xte, Tte, yte, Ste, Mte, num_features=F, max_tokens=self.max_tokens,
                                             time_max=time_max, pid_offset=len(self.train_ds)+len(self.val_ds))

        self.collate = make_strats_collate(padding_idx=F)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate)
    def pretrain_train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,   # Windows면 일단 0 추천
            pin_memory=True,
            collate_fn=make_strats_collate_pretrain(
                padding_idx=self.num_features,
                pre_mask_p=0.15,
            ),
            persistent_workers=(self.num_workers > 0),
        )

    def pretrain_val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=make_strats_collate_pretrain(
                padding_idx=self.num_features,
                pre_mask_p=0.15,),
            persistent_workers=(self.num_workers > 0),
        )

class SaveOnEpochs(Callback):
    """
    Save checkpoints at specific epochs (1-indexed by default).
    Example: epochs_to_save=[1,5,10, 15, 25]
    """
    def __init__(self, epochs_to_save, dirpath, prefix="ckpt", save_weights_only=True):
        super().__init__()
        self.epochs_to_save = set(int(e) for e in epochs_to_save)
        self.dirpath = dirpath
        self.prefix = prefix
        self.save_weights_only = save_weights_only
        os.makedirs(self.dirpath, exist_ok=True)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Lightning epoch is 0-indexed internally
        epoch_1idx = trainer.current_epoch + 1
        if epoch_1idx in self.epochs_to_save:
            fname = f"{self.prefix}_epoch{epoch_1idx}.ckpt"
            path = os.path.join(self.dirpath, fname)
            trainer.save_checkpoint(path, weights_only=self.save_weights_only)
            # 콘솔 로그 (원하면 삭제)
            print(f"[INFO] Saved checkpoint: {path}")
            
@torch.no_grad()
def _binary_pr_curve_calibrated(
    y_true: Tensor,          # [N] {0,1}
    y_pred: Tensor,          # [N] probabilities in [0,1]
    pi0: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute calibrated precision-recall points (descending thresholds).
    Returns (precision[T+1], recall[T+1]) where recall is nonincreasing (ends with 0),
    precision ends with 1, mimicking sklearn's output shape.

    Calibration (Siblini et al., 2020):
      precision = TP / (TP + ratio * FP)
      ratio = pi * (1 - pi0) / (pi0 * (1 - pi)),  where pi = empirical positive rate in y_true
    """
    # sort by score descending
    s, idx = torch.sort(y_pred.flatten(), descending=True)
    y = y_true.flatten().index_select(0, idx)

    # cumulative sums
    tps = torch.cumsum(y, dim=0)                            # TP at each threshold
    fps = torch.cumsum(1 - y, dim=0)                        # FP at each threshold

    # thresholds count (unique scores)
    # emulate sklearn behavior: collapse same-score blocks by taking last index of each unique value
    # but for integration we can keep every step; AP is invariant to collapsing ties in this construction.
    # For strict equivalence, do tie-handling:
    diff = torch.ones_like(s, dtype=torch.bool)
    diff[1:] = s[1:] != s[:-1]
    # take last index of each group -> use cumulative OR trick
    # Build mask for last occurrence in equal-score runs
    last_in_run = diff.clone()
    last_in_run[:-1] = diff[1:]  # shift
    last_in_run = torch.cat([last_in_run[1:], torch.tensor([True], device=last_in_run.device)])

    tps_u = tps[last_in_run]
    fps_u = fps[last_in_run]

    # append sentinel at end (sklearn-style)
    # Calibrated precision
    if pi0 is not None:
        pi = y.float().mean().item() if y.numel() > 0 else 0.0
        # guard rails
        eps = 1e-12
        pi = min(max(pi, eps), 1 - eps)
        pi0c = min(max(float(pi0), eps), 1 - eps)
        ratio = (pi * (1 - pi0c)) / (pi0c * (1 - pi))
        denom = tps_u + ratio * fps_u
    else:
        denom = tps_u + fps_u

    precision = torch.where(denom > 0, tps_u / denom.clamp_min(1e-12), torch.zeros_like(denom))
    # recall = TP / P
    P = tps_u[-1].clamp_min(1.0)
    recall = tps_u / P

    # reverse so recall is decreasing and append (1,0) endpoint like sklearn
    precision = torch.cat([precision.flip(0), torch.ones(1, device=precision.device)])
    recall = torch.cat([recall.flip(0), torch.zeros(1, device=recall.device)])

    return precision, recall


@torch.no_grad()
def calibrated_average_precision(
    y_true: Tensor,                 # [N] or [N,C] (binary multilabel)
    y_pred: Tensor,                 # same shape, probabilities (or logits if from_logits=True)
    *,
    pi0: Optional[Union[float, List[float], Tensor]] = 0.5,
    from_logits: bool = False,
) -> Tensor:
    """
    Functional AUPRC_c. If input is [N,C], returns [C] per-class AUPRC_c.
    """
    if from_logits:
        y_pred = torch.sigmoid(y_pred)

    y_true = y_true.float()
    if y_pred.ndim == 1:
        prec, rec = _binary_pr_curve_calibrated(y_true, y_pred, float(pi0) if pi0 is not None else None)
        # AP = -sum(diff(recall) * precision[:-1])
        return -(rec[1:] - rec[:-1]).mul(prec[:-1]).sum()
    else:
        C = y_pred.size(1)
        # broadcast pi0
        if isinstance(pi0, (list, tuple)):
            pi0_t = torch.tensor(pi0, dtype=torch.float32, device=y_pred.device)
        elif isinstance(pi0, torch.Tensor):
            pi0_t = pi0.to(y_pred.device, dtype=torch.float32)
        elif pi0 is None:
            pi0_t = torch.tensor([None] * C)  # sentinel; we’ll treat as no calibration
        else:
            pi0_t = torch.full((C,), float(pi0), device=y_pred.device)

        aps = []
        for c in range(C):
            p0 = None if (pi0 is None or (isinstance(pi0_t, torch.Tensor) and (pi0_t.dim()==1 and torch.isnan(pi0_t[c]) ))) else float(pi0_t[c].item()) if isinstance(pi0_t, torch.Tensor) else pi0
            prec, rec = _binary_pr_curve_calibrated(y_true[:, c], y_pred[:, c], p0)
            ap = -(rec[1:] - rec[:-1]).mul(prec[:-1]).sum()
            aps.append(ap)
        return torch.stack(aps, dim=0)


@torch.no_grad()
def get_surprise_score_one_pid(
    model: SurpriseSTraTS,
    *,
    times: torch.Tensor,        # [B, L]
    varis: torch.Tensor,        # [B, L]
    values: torch.Tensor,       # [B, L]
    padding_mask: torch.Tensor,# [B, L]
    pid: torch.Tensor,          # [B]
    target_pid: int,
    direction: str = "future",
):
    """
    Returns:
        sim: [Lv, Lv] cosine similarity matrix (triplet_base)
        times_v: [Lv]
        varis_v: [Lv]
        values_v: [Lv]
    """

    device = times.device

    # ---------------- pid index 찾기 ----------------
    idx = (pid == target_pid).nonzero(as_tuple=False).squeeze()
    if idx.numel() == 0:
        raise ValueError(f"pid={target_pid} not found in this batch")
    if idx.numel() > 1:
        idx = idx[0]  # 하나만 사용

    # ---------------- slice ----------------
    times_1  = times[idx:idx+1]
    varis_1  = varis[idx:idx+1]
    values_1 = values[idx:idx+1]
    pad_1    = padding_mask[idx:idx+1]

    # pad var id
    varis_1 = varis_1.masked_fill(pad_1, model.backbone.padding_idx)

    # ---------------- triplet_base ----------------
    t = model.backbone._time(times_1)             # [1,L,D]
    v = model.backbone._value_shared(values_1)    # [1,L,D]
    f = model.backbone.ln_feat(
        model.backbone.feature_embed(varis_1)
    )
    triplet_base = t + v + f                       # [1,L,D]

    # ---------------- flip handling ----------------
    flip = (direction == "future")
    if flip:
        triplet_base = torch.flip(triplet_base, dims=[1])
        pad_1 = torch.flip(pad_1, dims=[1])
        times_1 = torch.flip(times_1, dims=[1])
        varis_1 = torch.flip(varis_1, dims=[1])
        values_1 = torch.flip(values_1, dims=[1])

    # ---------------- valid tokens ----------------
    valid = (~pad_1[0])
    idx_v = torch.nonzero(valid, as_tuple=False).squeeze(1)

    tb_v = triplet_base[0, idx_v]                  # [Lv,D]

    # ---------------- cosine similarity ----------------
    tb_v = F.normalize(tb_v, p=2, dim=-1)
    sim = tb_v @ tb_v.T                            # [Lv,Lv]

    # ---------------- flip 복원 ----------------
    if flip:
        L = pad_1.size(1)
        idx_orig = (L - 1) - idx_v
        idx_orig, order = torch.sort(idx_orig)
        sim = sim[order][:, order]

        times_v  = times[idx, idx_orig]
        varis_v  = varis[idx, idx_orig]
        values_v = values[idx, idx_orig]
    else:
        times_v  = times[idx, idx_v]
        varis_v  = varis[idx, idx_v]
        values_v = values[idx, idx_v]

    return {
        "sim": sim.cpu(),           # [Lv, Lv]
        "times": times_v.cpu(),     # [Lv]
        "varis": varis_v.cpu(),     # [Lv]
        "values": values_v.cpu(),   # [Lv]
        "pid": int(target_pid),
    }

@torch.no_grad()
def _fetch_one_pid_from_loader(loader, target_pid: int, device):
    """
    Returns dict with tensors on device:
      times,varis,values,pad_mask,pre_mask,y,pid,static  (each [1,L] or [1,*])
    """
    for batch in loader:
        # batch: times, varis, values, pad_mask, pre_mask, y, pid, static
        times, varis, values, pad_mask, pre_mask, y, pid, static = batch
        # pid는 CPU일 수도 있어서 먼저 CPU에서 비교
        pid_cpu = pid.detach().cpu()
        hit = (pid_cpu == int(target_pid)).nonzero(as_tuple=False).squeeze()
        if hit.numel() == 0:
            continue
        if hit.numel() > 1:
            hit = hit[0]
        b = int(hit.item())

        # slice -> [1, L]
        def sl(x):
            return x[b:b+1].to(device) if torch.is_tensor(x) else x

        return {
            "times": sl(times),
            "varis": sl(varis),
            "values": sl(values),
            "pad_mask": sl(pad_mask),
            "pre_mask": sl(pre_mask),
            "y": sl(y),
            "pid": int(pid_cpu[b].item()),
            "static": sl(static),
        }

    raise ValueError(f"pid={target_pid} not found in loader")

@torch.no_grad()
def _surp_triplet_score_df(
    model,  # SurpriseSTraTS or compatible (has backbone._time/_value_shared/feature_embed etc.)
    one,    # dict from _fetch_one_pid_from_loader
    *,
    direction="future",
):
    times  = one["times"]      # [1,L]
    varis  = one["varis"]      # [1,L]
    values = one["values"]     # [1,L]
    pad    = one["pad_mask"]   # [1,L]
    L = times.size(1)

    # pad var ids
    varis_p = varis.masked_fill(pad, model.backbone.padding_idx)

    # triplet_base
    t = model.backbone._time(times)                  # [1,L,D]
    v = model.backbone._value_shared(values)         # [1,L,D]
    f = model.backbone.ln_feat(model.backbone.feature_embed(varis_p))
    triplet_base = t + v + f                         # [1,L,D]

    flip = (direction == "future")
    if flip:
        triplet_base = torch.flip(triplet_base, dims=[1])
        pad = torch.flip(pad, dims=[1])
        times = torch.flip(times, dims=[1])
        varis = torch.flip(varis, dims=[1])
        values = torch.flip(values, dims=[1])

    valid = (~pad[0])
    idx_v = torch.nonzero(valid, as_tuple=False).squeeze(1)  # [Lv]
    tb = triplet_base[0, idx_v]                               # [Lv,D]
    tb = F.normalize(tb, p=2, dim=-1)
    sim = tb @ tb.T                                           # [Lv,Lv]

    # restore original order indices if flipped
    if flip:
        idx_orig = (L - 1) - idx_v
        idx_orig, order = torch.sort(idx_orig)
        sim = sim[order][:, order]
        idx_use = idx_orig
    else:
        idx_use = idx_v

    # token df
    tok = pd.DataFrame({
        "t": idx_use.detach().cpu().numpy().astype(int),
        "time": times[0, idx_use].detach().cpu().numpy().astype(float),
        "varis": varis[0, idx_use].detach().cpu().numpy().astype(int),
        "values": values[0, idx_use].detach().cpu().numpy().astype(float),
    })

    # matrix -> long df
    sim_np = sim.detach().cpu().numpy()
    ii, jj = np.indices(sim_np.shape)
    df = pd.DataFrame({
        "i": ii.reshape(-1).astype(int),
        "j": jj.reshape(-1).astype(int),
        "score": sim_np.reshape(-1).astype(float),
    })
    # add token indices (original positions)
    df["t_i"] = tok["t"].values[df["i"].values]
    df["t_j"] = tok["t"].values[df["j"].values]

    # add same_var on valid indices
    var_v = tok["varis"].values
    df["same_var"] = (var_v[df["i"].values] == var_v[df["j"].values])

    return tok, df


@torch.no_grad()
def _vttg_score_df(
    model,  # SurpriseSTraTS_VT or backbone has _time/_value_per_feature (but we only need value_emb + time_raw)
    one,
    *,
    direction="future",
):
    times  = one["times"].to(torch.float32)   # [1,L] normalized [0,1]
    varis  = one["varis"]                     # [1,L]
    values = one["values"]                    # [1,L]
    pad    = one["pad_mask"]                  # [1,L]
    L = times.size(1)

    # value embedding: VT mode면 per_feature, 아니면 shared. 여기서는 "네 VTTG는 vt_mode에서 쓰는 것" 기준으로
    if getattr(model.backbone, "vt_mode", False):
        vemb = model.backbone._value_per_feature(values, varis)         # [1,L,D]
    else:
        vemb = model.backbone._value_shared(values)                      # [1,L,D]

    flip = (direction == "future")
    if flip:
        vemb = torch.flip(vemb, dims=[1])
        pad = torch.flip(pad, dims=[1])
        times = torch.flip(times, dims=[1])
        varis = torch.flip(varis, dims=[1])
        values = torch.flip(values, dims=[1])

    valid = (~pad[0])
    idx_v = torch.nonzero(valid, as_tuple=False).squeeze(1)
    v = vemb[0, idx_v]                                # [Lv,D]
    v = F.normalize(v, p=2, dim=-1)
    sim_v = v @ v.T                                   # [Lv,Lv]

    t = times[0, idx_v]                               # [Lv]
    dt = (t[:, None] - t[None, :]).abs().clamp(0.0, 1.0)  # [Lv,Lv]
    w = (1.0 - dt)                                    # [Lv,Lv]
    score = sim_v * w

    if flip:
        idx_orig = (L - 1) - idx_v
        idx_orig, order = torch.sort(idx_orig)
        score = score[order][:, order]
        dt = dt[order][:, order]
        w = w[order][:, order]
        idx_use = idx_orig
    else:
        idx_use = idx_v

    tok = pd.DataFrame({
        "t": idx_use.detach().cpu().numpy().astype(int),
        "time": times[0, idx_use].detach().cpu().numpy().astype(float),
        "varis": varis[0, idx_use].detach().cpu().numpy().astype(int),
        "values": values[0, idx_use].detach().cpu().numpy().astype(float),
    })

    score_np = score.detach().cpu().numpy()
    dt_np = dt.detach().cpu().numpy()
    w_np = w.detach().cpu().numpy()

    ii, jj = np.indices(score_np.shape)
    df = pd.DataFrame({
        "i": ii.reshape(-1).astype(int),
        "j": jj.reshape(-1).astype(int),
        "score": score_np.reshape(-1).astype(float),
        "dt": dt_np.reshape(-1).astype(float),
        "time_weight": w_np.reshape(-1).astype(float),
    })
    df["t_i"] = tok["t"].values[df["i"].values]
    df["t_j"] = tok["t"].values[df["j"].values]
    var_v = tok["varis"].values
    df["same_var"] = (var_v[df["i"].values] == var_v[df["j"].values])

    return tok, df
@torch.no_grad()
def _vt_score_df(
    model,
    one,
    *,
    direction="future",
    tau_v=1.0, tau_t=1.0, w_v=1.0, w_t=1.0,
):
    times  = one["times"].to(torch.float32)
    varis  = one["varis"]
    values = one["values"]
    pad    = one["pad_mask"]
    L = times.size(1)

    # value embedding (VT mode 가정)
    if getattr(model.backbone, "vt_mode", False):
        vemb = model.backbone._value_per_feature(values, varis)
    else:
        vemb = model.backbone._value_shared(values)

    # time embedding
    temb = model.backbone._time(times)

    flip = (direction == "future")
    if flip:
        vemb = torch.flip(vemb, dims=[1])
        temb = torch.flip(temb, dims=[1])
        pad = torch.flip(pad, dims=[1])
        times = torch.flip(times, dims=[1])
        varis = torch.flip(varis, dims=[1])
        values = torch.flip(values, dims=[1])

    valid = (~pad[0])
    idx_v = torch.nonzero(valid, as_tuple=False).squeeze(1)

    v = F.normalize(vemb[0, idx_v], p=2, dim=-1)
    t = F.normalize(temb[0, idx_v], p=2, dim=-1)

    sim_v = v @ v.T
    sim_t = t @ t.T

    score = (w_v * (sim_v / max(float(tau_v), 1e-6))) + (w_t * (sim_t / max(float(tau_t), 1e-6)))

    if flip:
        idx_orig = (L - 1) - idx_v
        idx_orig, order = torch.sort(idx_orig)
        score = score[order][:, order]
        idx_use = idx_orig
    else:
        idx_use = idx_v

    tok = pd.DataFrame({
        "t": idx_use.detach().cpu().numpy().astype(int),
        "time": times[0, idx_use].detach().cpu().numpy().astype(float),
        "varis": varis[0, idx_use].detach().cpu().numpy().astype(int),
        "values": values[0, idx_use].detach().cpu().numpy().astype(float),
    })

    score_np = score.detach().cpu().numpy()
    ii, jj = np.indices(score_np.shape)
    df = pd.DataFrame({
        "i": ii.reshape(-1).astype(int),
        "j": jj.reshape(-1).astype(int),
        "score": score_np.reshape(-1).astype(float),
    })
    df["t_i"] = tok["t"].values[df["i"].values]
    df["t_j"] = tok["t"].values[df["j"].values]
    var_v = tok["varis"].values
    df["same_var"] = (var_v[df["i"].values] == var_v[df["j"].values])

    return tok, df

@torch.no_grad()
def _token_mask_df_one_pid(lit_or_model, one):
    # check_padding은 batch 입력을 받으므로 [1,L] 그대로 넣으면 됨
    if not hasattr(lit_or_model, "check_padding"):
        return None

    cp = lit_or_model.check_padding(
        times=one["times"], varis=one["varis"], values=one["values"], padding_mask=one["pad_mask"]
    )
    pad = one["pad_mask"][0].detach().cpu().numpy()
    m = cp["mask"][0].detach().cpu().numpy()
    # valid만
    idx = np.nonzero(~pad)[0]
    return pd.DataFrame({"t": idx.astype(int), "mask": m[idx].astype(bool)})


# =========================
# TorchMetrics-style Metric
# =========================
class CalibratedAveragePrecision(torchmetrics.Metric):
    """
    TorchMetrics metric for AUPRC_c.

    Args:
        num_labels: None for binary, or C for multilabel (returns [C] at compute unless average given)
        pi0: scalar, list/1D tensor per-class, or None (no calibration)
        from_logits: apply sigmoid internally
        average: None | "macro". If "macro" and multilabel, returns scalar average of per-class scores.
    """
    full_state_update = False  # we'll store all preds/targets and compute at epoch end

    def __init__(
        self,
        num_labels: Optional[int] = None,
        pi0: Optional[Union[float, List[float], Tensor]] = 0.5,
        from_logits: bool = False,
        average: Optional[str] = None,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_labels = num_labels
        self.pi0 = pi0
        self.from_logits = from_logits
        if average not in (None, "macro"):
            raise ValueError('average must be None or "macro"')
        self.average = average

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor):
        """
        preds: [N] or [N,C] (logits or probabilities)
        target: same shape with {0,1}
        """
        preds = preds.detach()
        target = target.detach().float()
        if self.from_logits:
            preds = torch.sigmoid(preds)
        # flatten batch into first dim
        if preds.dim() == 1:
            self.preds.append(preds)
            self.targets.append(target)
        else:
            self.preds.append(preds)
            self.targets.append(target)

    def compute(self) -> Tensor:
        preds = torch.cat(self.preds, dim=0)
        targets = torch.cat(self.targets, dim=0)

        score = calibrated_average_precision(
            targets, preds, pi0=self.pi0, from_logits=False  # already handled
        )
        if score.ndim == 0:
            return score
        if self.average == "macro":
            return score.mean()
        return score  # per-class [C]

# ==== you already defined these in your env ====
# - prepare_dataset_samples, normalize_many, TimeSeriesDataset, make_dataloader
# - Models: STraTS, SurpriseSTraTS, SurpriseSTraTS_VT
# - (option) STRaTSLightning if you want; here we implement a task-flexible wrapper.


# -------------------------
# LightningModule (binary / multilabel)
# -------------------------
class STRaTSLit(pl.LightningModule):
    """
    Downstream 전용 (이진 / 멀티라벨). 
    - 손실: 
        binary: BCEWithLogitsLoss
        multilabel: BCEWithLogitsLoss(각 라벨 독립)
    - 로깅: loss, ACC, AUROC, AUPRC, AUPRC_c(정밀-재현 곡선에서 임계치별 평균 정밀도와 유사; 여기선 AP)
    """
    def __init__(
        self,
        model: nn.Module,
        *,
        task_type: Literal["binary", "multilabel"] = "binary",
        outcome_cols: Sequence[str] = ("mortality_inunit",),
        target_indices: Optional[Sequence[int]] = None,  # None이면 outcome_cols의 모든 컬럼 사용
        pos_weight: Optional[Sequence[float]] = None,    # 멀티라벨일 때 각 라벨 pos_weight
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        optimizer: Literal["adamw", "adam"] = "adamw",
        scheduler: Optional[Literal["none", "cosine"]] = "none",
        scheduler_params: Optional[Dict[str, Any]] = None,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.outcome_cols = list(outcome_cols)
        self.target_indices = list(range(len(self.outcome_cols))) if target_indices is None else list(target_indices)
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.scheduler = scheduler or "none"
        self.scheduler_params = scheduler_params or {}
        self.pretrained_path = pretrained_path

        num_targets = len(self.target_indices)
        if task_type == "binary":
            self.criterion = nn.BCEWithLogitsLoss()
            # per-epoch buffers
            self._buf = {"train": [], "val": [], "test": [], "target": []}
            self._lbl = {"train": [], "val": [], "test": [], "target": []}
            self.auroc = BinaryAUROC()
            self.auprc = BinaryAveragePrecision()
            self.acc   = BinaryAccuracy()
        elif task_type == "multilabel":
            pw = None
            if pos_weight is not None:
                assert len(pos_weight) == num_targets, "pos_weight length must match #targets"
                pw = torch.tensor(pos_weight, dtype=torch.float32)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            self._buf = {"train": [], "val": [], "test": [], "target": []}
            self._lbl = {"train": [], "val": [], "test": [], "target": []}
            # per-label metrics는 epoch_end에서 루프 돌며 계산
        else:
            raise ValueError("task_type must be 'binary' or 'multilabel'.")

        self.save_hyperparameters(ignore=["model"])

    # --- optional pretrained load ---
    def on_fit_start(self):
        if self.pretrained_path and os.path.isfile(self.pretrained_path):
            ckpt = torch.load(self.pretrained_path, map_location=self.device)
            state = ckpt.get("state_dict", ckpt)
            try:
                # smart load: if it's raw inner-model weights
                if any(k.startswith("model.") for k in state.keys()):
                    self.load_state_dict(state, strict=False)
                else:
                    self.model.load_state_dict(state, strict=False)
                print(f"[INFO] Loaded pretrained weights from {self.pretrained_path}")
            except Exception as e:
                print(f"[WARN] Failed to load pretrained: {e}")
        else:
            print(f"[WARN] No files found in {self.pretrained_path}")

    def forward(self, batch):
        times, varis, values, pad_mask, pre_mask, y, pid, static = batch
        out = self.model(times=times, varis=varis, values=values, statics=static,
                         padding_mask=pad_mask, pretrain=False)
        # out['pred']: [B] for single-task; but 여기선 다중 라벨도 B×T 필요하므로 Head를 (in, #targets)로 맞춰야 함
        return out

    def _select_targets(self, y: torch.Tensor) -> torch.Tensor:
        # y: [B, K_all] -> pick indices -> [B, K]
        return y[:, self.target_indices].float()

    def _step(self, batch, stage: str):
        out = self.forward(batch)
        logits = out["pred"]  # expect [B] (binary) or [B, K] (multilabel)

        _, _, _, _, _, y, _, _ = batch
        targets = self._select_targets(y)  # [B] or [B,K]

        if self.task_type == "binary":
            # logits: [B] or [B,1] -> flatten to [B]
            if logits.dim() == 2 and logits.size(1) == 1:
                logits = logits.view(-1)        # [B]

            # targets: [B] or [B,1] -> flatten to [B]
            if targets.dim() == 2 and targets.size(1) == 1:
                targets = targets.view(-1)      # [B]

            loss = self.criterion(logits, targets)
            self._buf[stage].append(logits.detach().cpu())
            self._lbl[stage].append(targets.detach().cpu())
            self.log(f"{stage}_loss", loss, on_step=(stage=="train"), on_epoch=True, prog_bar=True)
            return loss


        else:
            # multilabel: logits [B,K], targets [B,K]
            if logits.dim() == 1:
                logits = logits.unsqueeze(-1)
            loss = self.criterion(logits, targets)
            self._buf[stage].append(logits.detach().cpu())
            self._lbl[stage].append(targets.detach().cpu())
            self.log(f"{stage}_loss", loss, on_step=(stage=="train"), on_epoch=True, prog_bar=True)
            return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    # external evaluation for target dataloader
    @torch.no_grad()
    def evaluate_dataloader(self, dataloader, stage: str = "target"):
        device = self.device

        self.eval()
        self._buf[stage].clear(); self._lbl[stage].clear()

        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Eval [{stage}]", leave=True)

        for batch_idx, batch in pbar:
            batch = tuple(x.to(device) if torch.is_tensor(x) else x for x in batch)
            times, varis, values, pad_mask, pre_mask, y, pid, static = batch

            out = self.model(
                times=times, varis=varis, values=values, statics=static,
                padding_mask=pad_mask, pretrain=False
            )

            logits = out["pred"]
            targets = self._select_targets(y)

            if self.task_type == "binary":
                if logits.dim() == 2 and logits.size(1) == 1:
                    logits = logits.view(-1)
                if targets.dim() == 2 and targets.size(1) == 1:
                    targets = targets.view(-1)
            else:
                if logits.dim() == 1:
                    logits = logits.unsqueeze(-1)

            loss = self.criterion(logits, targets.float())
            total_loss += float(loss.detach().cpu())
            n_batches += 1

            self._buf[stage].append(logits.detach().cpu())
            self._lbl[stage].append(targets.detach().cpu())

            pbar.set_postfix({
                "loss": f"{total_loss / n_batches:.4f}",
                "batches": n_batches,
            })

        # -------- per-epoch metric 계산 --------
        logits = torch.cat(self._buf[stage], dim=0)
        targets = torch.cat(self._lbl[stage], dim=0)
        logs = {}

        if self.task_type == "binary":
            if logits.dim() == 2 and logits.size(1) == 1:
                logits = logits.view(-1)
            if targets.dim() == 2 and targets.size(1) == 1:
                targets = targets.view(-1)
            probs = torch.sigmoid(logits)
            auroc = self.auroc(probs, targets.int()).item()
            auprc = self.auprc(probs, targets.int()).item()
            acc   = self.acc((probs >= 0.5).int(), targets.int()).item()
            logs.update({
                f"{stage}_AUROC": auroc,
                f"{stage}_AUPRC": auprc,
                f"{stage}_ACC":   acc,
                f"{stage}_AUPRC_c": auprc,
            })
        else:
            probs = torch.sigmoid(logits)   # [N, K]
            K = probs.size(1)
            aurocs, auprcs, accs = [], [], []
            # 선택된 타깃 이름
            sel_names = [self.outcome_cols[i] for i in self.target_indices]

            for j in range(K):
                p = probs[:, j]; t = targets[:, j].int()
                auroc_j = BinaryAUROC().to(p.device)(p, t).item()
                auprc_j = BinaryAveragePrecision().to(p.device)(p, t).item()
                acc_j   = BinaryAccuracy().to(p.device)((p >= 0.5).int(), t).item()
                aurocs.append(auroc_j); auprcs.append(auprc_j); accs.append(acc_j)

                # ←← 여기서 per-label 키로 logs에 추가 (Lightning self.log 말고 수동 dict)
                name = sel_names[j]
                logs[f"{stage}_AUROC/{name}"] = auroc_j
                logs[f"{stage}_AUPRC/{name}"] = auprc_j
                logs[f"{stage}_ACC/{name}"]   = acc_j

            logs.update({
                f"{stage}_AUROC_macro":   float(np.mean(aurocs)),
                f"{stage}_AUPRC_macro":   float(np.mean(auprcs)),
                f"{stage}_ACC_macro":     float(np.mean(accs)),
                f"{stage}_AUPRC_c_macro": float(np.mean(auprcs)),
            })

        if n_batches > 0:
            logs[f"{stage}_loss"] = total_loss / n_batches

        # ---- W&B 수동 로그 (per-label 포함) ----
        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log({k: v for k, v in logs.items()}, step=self.global_step)

        self._buf[stage].clear(); self._lbl[stage].clear()
        return logs

    # ---- epoch-end metrics ----
    def on_train_epoch_end(self): # For debugging purposes
        self._buf["train"].clear()
        self._lbl["train"].clear()
        # print("train buf:", len(self._buf["train"]))
    def on_validation_epoch_end(self):
        self._compute_and_log_epoch_metrics("val")

    def on_test_epoch_end(self):
        self._compute_and_log_epoch_metrics("test")

    def _compute_and_log_epoch_metrics(self, stage: str, allow_log: bool = True) -> Dict[str, float]:
        if len(self._buf[stage]) == 0:
            return {}
        logits = torch.cat(self._buf[stage], dim=0)
        targets = torch.cat(self._lbl[stage], dim=0)
        logs = {}

        if self.task_type == "binary":
            # logits: [B] or [B,1] -> flatten to [B]
            if logits.dim() == 2 and logits.size(1) == 1:
                logits = logits.view(-1)        # [B]

            # targets: [B] or [B,1] -> flatten to [B]
            if targets.dim() == 2 and targets.size(1) == 1:
                targets = targets.view(-1)      # [B]

            probs = torch.sigmoid(logits)
            auroc = self.auroc(probs, targets.int()).item()
            auprc = self.auprc(probs, targets.int()).item()
            acc   = self.acc((probs >= 0.5).int(), targets.int()).item()
            logs = {
                f"{stage}_AUROC": auroc,
                f"{stage}_AUPRC": auprc,
                f"{stage}_ACC":   acc,
                f"{stage}_AUPRC_c": auprc,  # 현재는 AP와 동일 취급
            }
            if allow_log:
                for k, v in logs.items():
                    self.log(k, v, prog_bar=False, on_epoch=True)

        else:
            probs = torch.sigmoid(logits)   # [N, K]
            K = probs.size(1)
            aurocs, auprcs, accs = [], [], []
            for j in range(K):
                p = probs[:, j]; t = targets[:, j].int()
                auroc_j = BinaryAUROC().to(p.device)(p, t).item()
                auprc_j = BinaryAveragePrecision().to(p.device)(p, t).item()
                acc_j   = BinaryAccuracy().to(p.device)((p >= 0.5).int(), t).item()
                aurocs.append(auroc_j); auprcs.append(auprc_j); accs.append(acc_j)
                if allow_log:
                    name = self.outcome_cols[self.target_indices[j]]
                    self.log(f"{stage}_AUROC/{name}", auroc_j, on_epoch=True)
                    self.log(f"{stage}_AUPRC/{name}", auprc_j, on_epoch=True)
                    self.log(f"{stage}_ACC/{name}",   acc_j,   on_epoch=True)

            logs = {
                f"{stage}_AUROC_macro":   float(np.mean(aurocs)),
                f"{stage}_AUPRC_macro":   float(np.mean(auprcs)),
                f"{stage}_ACC_macro":     float(np.mean(accs)),
            }
            if allow_log:
                for k, v in logs.items():
                    self.log(k, v, on_epoch=True)

        return logs

    # ---- optim ----
    def configure_optimizers(self):
        if self.optimizer_name == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        if self.scheduler == "none":
            return opt
        elif self.scheduler == "cosine":
            T_max = self.scheduler_params.get("T_max", 100)
            eta_min = self.scheduler_params.get("eta_min", 1e-6)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")


# -------------------------
# Data utilities
# -------------------------
def _load_split(raw_dir: str, domain: str, split: str, task: str, aug=False):
    if task=='pheno':
        df = pd.read_feather(os.path.join(raw_dir, f"{domain}_data_{split}.feather"))
        st = pd.read_feather(os.path.join(raw_dir, f"{domain}_data_static_{split}.feather"))
        oc = pd.read_feather(os.path.join(raw_dir, f"{domain}_outcomes_pheno_binary_{split}.feather"))
    else:
        if aug:
            df = pd.read_feather(os.path.join(raw_dir, f"{domain}_data_aug_p_{split}.feather"))
        else:
            df = pd.read_feather(os.path.join(raw_dir, f"{domain}_data_{split}.feather"))
        st = pd.read_feather(os.path.join(raw_dir, f"{domain}_data_static_{split}.feather"))
        oc = pd.read_feather(os.path.join(raw_dir, f"{domain}_outcomes_{split}.feather"))
    return df, st, oc

def _build_samples_and_normalize(
    train_df, train_st, train_oc,
    valid_df, valid_st, valid_oc,
    test_df,  test_st,  test_oc,
    *,
    categorical_itemids: Iterable[int|float] = (),
):
    # build samples
    train_s = prepare_dataset_samples(train_df, train_st)
    valid_s = prepare_dataset_samples(valid_df, valid_st)
    test_s  = prepare_dataset_samples(test_df,  test_st)

    # normalize with train stats
    normed, stats = normalize_many(
        train_s, categorical_itemids=categorical_itemids,
        train=train_s, valid=valid_s, test=test_s
    )
    return normed["train"], normed["valid"], normed["test"], stats

def _make_loaders(
    train_samples, valid_samples, test_samples,
    train_oc, valid_oc, test_oc,
    *,
    outcome_cols: Sequence[str],
    batch_size: int = 64,
    num_workers: int = 0,
    make_pre_mask_train: bool = False,
    pre_mask_p_train: float = 0.2,
):
    ds_tr = TimeSeriesDataset(train_samples, train_oc, outcome_cols=outcome_cols)
    ds_va = TimeSeriesDataset(valid_samples, valid_oc, outcome_cols=outcome_cols)
    ds_te = TimeSeriesDataset(test_samples,  test_oc,  outcome_cols=outcome_cols)

    ld_tr = make_dataloader(ds_tr, batch_size, num_workers=num_workers, make_pre_mask=make_pre_mask_train, pre_mask_p=pre_mask_p_train)
    ld_va = make_dataloader(ds_va, batch_size, num_workers=num_workers, make_pre_mask=make_pre_mask_train, pre_mask_p=pre_mask_p_train)
    ld_te = make_dataloader(ds_te, batch_size, num_workers=num_workers, make_pre_mask=False, pre_mask_p=0.0)
    return (ds_tr, ds_va, ds_te), (ld_tr, ld_va, ld_te)

def _make_loaders_random(
    train_samples, valid_samples, test_samples,
    train_oc, valid_oc, test_oc,
    *,
    outcome_cols: Sequence[str],
    batch_size: int = 64,
    num_workers: int = 0,
):
    ds_tr = TimeSeriesDatasetETT(train_samples, train_oc, outcome_cols=outcome_cols)
    ds_va = TimeSeriesDatasetETT(valid_samples, valid_oc, outcome_cols=outcome_cols)
    ds_te = TimeSeriesDatasetETT(test_samples,  test_oc,  outcome_cols=outcome_cols)

    ld_tr = make_dataloader_random(ds_tr, batch_size, shuffle=True,  num_workers=num_workers)
    ld_va = make_dataloader_random(ds_va, batch_size, shuffle=False, num_workers=num_workers)
    ld_te = make_dataloader_random(ds_te, batch_size, shuffle=False, num_workers=num_workers)
    return (ds_tr, ds_va, ds_te), (ld_tr, ld_va, ld_te)
# -------------------------
# PacMAP and dump helpers
# -------------------------
def pacmap_plot(emb_list: List[np.ndarray], labels: List[str], title: str = "PaCMAP"):
    if not _HAS_PACMAP:
        print("[WARN] PaCMAP is not installed; skipping embedding plot.")
        return None
    X = np.vstack(emb_list)  # [N, D]
    y = np.array(labels)      # [N]
    reducer = PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
    X2 = reducer.fit_transform(X, init="pca")
    plt.figure(figsize=(6,5))
    uniq = np.unique(y)
    for u in uniq:
        m = (y == u)
        plt.scatter(X2[m,0], X2[m,1], s=10, label=str(u), alpha=0.7)
    plt.legend(title="Group", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def collect_inputs_with_masks(times, varis, values, pad_mask, pids):
    """
    Return list of {pid, observations: [N,4]} with columns [time, feature, value, mask]
    mask=True means PAD (following your convention).
    """
    out = []
    B, T = times.shape
    for i in range(B):
        L = int((~pad_mask[i]).sum().item())  # valid length
        obs = torch.stack([times[i,:L], varis[i,:L].float(), values[i,:L], pad_mask[i,:L].float()], dim=1)
        out.append({"pid": int(pids[i].item()), "observations": obs.cpu().numpy()})
    return out

# ETT
class STRaTSLitForecast(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        *,
        lr=1e-3,
        weight_decay=1e-2,
        optimizer="adamw",
        # y_mean / y_std:
        #  - S 모드: scalar
        #  - M 모드: shape [F_out] (feature dim 기준)
        y_mean: float | np.ndarray | None = None,
        y_std:  float | np.ndarray | None = None,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

        # inverse stats (torch tensor로 보관, 브로드캐스트용)
        if y_mean is None or y_std is None:
            self.y_mean = None
            self.y_std = None
        else:
            ym = torch.tensor(y_mean, dtype=torch.float32)
            ys = torch.tensor(y_std, dtype=torch.float32)
            self.register_buffer("y_mean", ym)
            self.register_buffer("y_std", ys)

    def forward(self, batch):
        times, varis, values, pad_mask, pre_mask, y, pid, static = batch
        return self.model(
            times=times, varis=varis, values=values, statics=static,
            padding_mask=pad_mask, pretrain=False
        )

    def _inverse_y(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, pred_len] 또는 [B, pred_len, F_out] 등의 텐서
        """
        if (self.y_mean is None) or (self.y_std is None):
            return x
        return x * self.y_std + self.y_mean  # 브로드캐스트로 처리

    @staticmethod
    def _flatten_for_metric(pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        metric 계산용으로 batch dim(0)만 남기고 나머지 차원은 전부 flatten.
        - pred, y shape 동일 가정.
        """
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        # 예: [B, H] -> [B, H], [B, H, F] -> [B, H*F]
        B = pred.size(0)
        return pred.view(B, -1), y.view(B, -1)

    def _step(self, batch, stage: str):
        out = self.forward(batch)
        pred = out["pred"]          # [B, pred_len] 또는 [B, pred_len, F_out]
        y = batch[5].float()        # 같은 shape 기대

        # ---- scaled space metrics ----
        pred_flat, y_flat = self._flatten_for_metric(pred, y)

        mse_scaled = F.mse_loss(pred_flat, y_flat, reduction="mean")
        mae_scaled = F.l1_loss(pred_flat, y_flat, reduction="mean")
        rmse_scaled = torch.sqrt(mse_scaled.clamp_min(1e-12))

        loss = mse_scaled  # 학습 loss


        # ---- 로그 ----
        # scaled
        self.log(f"{stage}_MSE", mse_scaled, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_MAE", mae_scaled, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_RMSE", rmse_scaled, on_step=False, on_epoch=True, prog_bar=True)

        self.log(
            f"{stage}_loss", loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def test_step(self, batch, _):
        self._step(batch, "test")

    @torch.no_grad()
    def evaluate_dataloader(self, dataloader, stage: str = "target"):
        device = self.device
        self.eval()

        preds_all, y_all = [], []

        for batch in dataloader:
            batch = tuple(x.to(device) if torch.is_tensor(x) else x for x in batch)
            out = self.forward(batch)
            pred = out["pred"]
            y = batch[5].float()

            preds_all.append(pred.detach().cpu())
            y_all.append(y.detach().cpu())

        if not preds_all:
            raise RuntimeError("No batches in dataloader.")

        pred = torch.cat(preds_all, dim=0)  # [N,...]
        y = torch.cat(y_all, dim=0)

        # ---- scaled ----
        pred_flat, y_flat = self._flatten_for_metric(pred, y)
        mse_scaled = F.mse_loss(pred_flat, y_flat, reduction="mean")
        mae_scaled = F.l1_loss(pred_flat, y_flat, reduction="mean")
        rmse_scaled = torch.sqrt(mse_scaled.clamp_min(1e-12))


        logs = {
            f"{stage}_MSE": mse_scaled,
            f"{stage}_MAE": mae_scaled,
            f"{stage}_RMSE": rmse_scaled,

        }

        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log(
                {k: float(v.detach().cpu()) for k, v in logs.items()},
                step=self.global_step,
            )

        return logs

    @torch.no_grad()
    def collect_predictions_df(self, dataloader, *, stage: str, source_tag: str) -> pd.DataFrame:
        """
        - S 모드(단일 타깃): pred/y shape [B, pred_len] → y_0, ..., y_{pred_len-1}
        - M 모드(멀티 타깃): pred/y shape [B, pred_len, F_out] →
            y_t{t}_f{f} 형식으로 저장 (t=타임스텝, f=feature index)
        """
        device = self.device
        self.eval()

        rows = []
        for batch in dataloader:
            batch = tuple(x.to(device) if torch.is_tensor(x) else x for x in batch)
            out = self.forward(batch)
            pred = out["pred"].detach().cpu().numpy()  # scaled
            y = batch[5].float().detach().cpu().numpy()  # scaled
            pid_np = batch[6].detach().cpu().numpy().astype(int)

            if pred.ndim == 2:
                # [B, pred_len]
                B, K = pred.shape
                for b in range(B):
                    row = {"pid": int(pid_np[b]), "stage": stage, "source": source_tag}
                    for k in range(K):
                        row[f"y_{k}"] = float(y[b, k])
                        row[f"pred_{k}"] = float(pred[b, k])
                    rows.append(row)

            elif pred.ndim == 3:
                # [B, pred_len, F_out]
                B, T, F_out = pred.shape
                for b in range(B):
                    row = {"pid": int(pid_np[b]), "stage": stage, "source": source_tag}
                    for t in range(T):
                        for f in range(F_out):
                            row[f"y_t{t}_f{f}"] = float(y[b, t, f])
                            row[f"pred_t{t}_f{f}"] = float(pred[b, t, f])
                    rows.append(row)

            else:
                raise ValueError(f"Unexpected pred ndim={pred.ndim}")

        return pd.DataFrame(rows) if rows else pd.DataFrame()


    def configure_optimizers(self):
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    @torch.no_grad()
    def save_first_window_plot(
        self,
        dataloader,
        *,
        save_path: str,
        title: str = "Forecast vs Ground Truth (First Test Window)",
    ):
        """
        test dataloader의 첫 batch에서 첫 sample만 시각화.
        - S 모드: OT 하나만 plot (기존과 동일)
        - M 모드: 마지막 feature(또는 원하는 feature index)만 골라서 plot 하고 싶다면
          여기서 인덱스 선택해서 쓰면 됨. 지금은 S 모드 기준 그대로 유지.
        """
        self.eval()
        device = self.device

        batch = next(iter(dataloader))
        batch = tuple(x.to(device) if torch.is_tensor(x) else x for x in batch)

        out = self.forward(batch)
        pred = out["pred"]
        y = batch[5].float()
        pid = batch[6]

        # 첫 sample만
        pred = pred[0]
        y = y[0]
        pid0 = int(pid[0].item())

        # S 모드용: [pred_len] 또는 [pred_len, F_out] 중 첫 feature만 그릴 수도 있음
        if pred.dim() == 2:
            # [pred_len, F_out] 인 경우, 마지막 dim에서 하나 골라 쓰고 싶으면 인덱스 조정
            # 여기서는 일단 첫 feature만 사용
            pred = pred[:, 0]
            y = y[:, 0]


        plt.figure(figsize=(8, 4))
        plt.plot(y, label="Ground Truth", marker="o")
        plt.plot(pred, label="Prediction", marker="x")
        plt.xlabel("Forecast Horizon")
        plt.ylabel("Value")
        plt.title(f"{title}\nPID={pid0}")
        plt.legend()
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()



# -------------------------
# Train / Evaluate routines
# -------------------------
@dataclass
class RunConfig:
    source: str
    model_name: Literal["strats", "surprise", "surprise_vt"]
    model_cfg: Dict[str, Any]
    task_type: Literal["binary", "multilabel"] = "binary"
    task_name: str = 'pheno' 
    outcome_cols: Sequence[str] = ("mortality_inunit",)
    target_indices: Optional[Sequence[int]] = None
    raw_directory: str = "./data"
    target_domain: Optional[str] = None
    batch_size: int = 64
    num_workers: int = 0
    lr: float = 5e-4
    weight_decay: float = 1e-2
    optimizer: Literal["adamw","adam"] = "adamw"
    scheduler: Literal["none","cosine"] = "none"
    scheduler_params: Dict[str,Any] = None
    max_epochs_pretrain: int = 0
    max_epochs_train: int = 20
    pre_mask_p_train: float = 0.15
    categorical_itemids: Iterable[int|float] = ()
    pretrained_path: Optional[str] = None
    save_pretrained_after: bool = False
    downstream_path: Optional[str] = None
    aug: bool=False
    project_name: str = "Surprise"
    # ---- ETT 옵션 ----
    ett_csv_path: Optional[str] = None
    ett_target_csv_path: Optional[str] = None
    ett_lookback: int = 96
    ett_pred_len: int = 96
    ett_features: Literal["S","M"] = "M"
    ett_target_col: str = "OT"
    ett_official_split: bool = True
    # ---- P12/P19/PAM 옵션 ----
    p12p19_dataset: Optional[Literal["P12","P19","PAM"]] = None
    p12p19_base_path: Optional[str] = None
    p12p19_split_path: Optional[str] = None
    p12p19_num_splits: int = 5
    p12p19_split_path_pattern: Optional[str] = None
    p12p19_split_type: str = "random"         # random/age/gender
    p12p19_reverse: bool = False
    p12p19_baseline: bool = False
    p12p19_predictive_label: str = "mortality"  # mortality/LoS
    p12p19_max_tokens: int = 4096
    # ✅ 새 옵션
    eval_only: bool = False                 # True면 훈련 스킵하고 바로 평가/저장만 수행
    log_to_wandb: bool = True               # 성능 로깅을 W&B에 남길지 여부
    output_directory: str = "./outputs"     # 결과 디렉토리 (없으면 생성)

    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    accelerator: str = "auto"
    devices: Any = "auto"
    precision: str|int = "16-mixed"

def build_model(name: str, cfg: Dict[str,Any], num_features: int) -> nn.Module:
    if name == "strats":
        return STraTS(num_features=num_features, **cfg)
    elif name == "surprise":
        return SurpriseSTraTS(num_features=num_features, **cfg)
    elif name == "surprise_vt":
        return SurpriseSTraTS_VT(num_features=num_features, **cfg)
    else:
        raise ValueError(f"Unknown model_name: {name}")

def run_pipeline(cfg: RunConfig):
    pl.seed_everything(9871, workers=True)

    if cfg.task_name == "p12p19":
        if not cfg.p12p19_dataset or not cfg.p12p19_base_path:
            raise ValueError("task_name='p12p19' requires p12p19_dataset and p12p19_base_path")

        # split path 결정 로직:
        # 1) pattern이 있으면 pattern.format(fold)
        # 2) 없으면 cfg.p12p19_split_path 그대로 쓰되, fold를 지원 못함 -> 에러
        if cfg.p12p19_split_path_pattern is None:
            if not cfg.p12p19_split_path:
                raise ValueError(
                    "Need either p12p19_split_path_pattern or p12p19_split_path for p12p19."
                )
            # split 하나만 있으면 CV 못 돌림
            raise ValueError(
                "You asked for 5-split validation. Provide --p12p19_split_path_pattern "
                'e.g. "/splits/phy19_split{}_new.npy"'
            )

        K = int(getattr(cfg, "p12p19_num_splits", 5))
        metric_keys = ["val_AUROC", "val_AUPRC", "val_ACC", "test_AUROC", "test_AUPRC", "test_ACC"]

        fold_rows = []
        wandb_run = None

        # W&B는 “한 번만” init하고 fold별로 log하는 게 정석
        if cfg.log_to_wandb:
            project = cfg.project_name
            wandb_run = wandb.init(
                project=project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name,
                config={**asdict(cfg)}
            )

        for fold in range(1, K + 1):
            split_path = cfg.p12p19_split_path_pattern.format(fold)

            dm = P12P19DataModule(
                dataset=cfg.p12p19_dataset,
                base_path=cfg.p12p19_base_path,
                split_path=split_path,
                split_type=cfg.p12p19_split_type,
                reverse=cfg.p12p19_reverse,
                baseline=cfg.p12p19_baseline,
                predictive_label=cfg.p12p19_predictive_label,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                max_tokens=cfg.p12p19_max_tokens,
            )
            dm.setup()

            num_features = dm.num_features

            # 모델 매 fold마다 새로 (같은 초기화라도 seed 고정이면 재현 가능)
            model_core = build_model(cfg.model_name, cfg.model_cfg or {}, num_features=num_features)
            # =========================
            # (A) optional pretrain per-fold
            # =========================
            if cfg.max_epochs_pretrain > 0:
                # pretrain loaders: must output (times,varis,values,pad,pre,_,_,static)
                ld_tr_pre = dm.pretrain_train_dataloader() if hasattr(dm, "pretrain_train_dataloader") else dm.train_dataloader()
                ld_va_pre = dm.pretrain_val_dataloader() if hasattr(dm, "pretrain_val_dataloader") else dm.val_dataloader()

                class PretrainLit(pl.LightningModule):
                    def __init__(self, model, lr=1e-3, wd=1e-2):
                        super().__init__()
                        self.model = model
                        self.lr = lr
                        self.wd = wd

                    def training_step(self, batch, _):
                        times, varis, values, pad, pre, _, _, static = batch
                        out = self.model(
                            times=times, varis=varis, values=values, statics=static,
                            padding_mask=pad, pretrain=True, pretrain_mask=pre
                        )
                        pred_vals = out["pred_vals"]
                        pre_m = out["pretrain_mask"].bool()
                        valid = (~pad)
                        mask = pre_m & valid

                        loss = (pred_vals.sum() * 0.0) if (not mask.any()) else ((pred_vals - values) ** 2)[mask].mean()
                        self.log("pretrain_train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
                        return loss

                    def validation_step(self, batch, _):
                        times, varis, values, pad, pre, _, _, static = batch
                        out = self.model(
                            times=times, varis=varis, values=values, statics=static,
                            padding_mask=pad, pretrain=True, pretrain_mask=pre
                        )
                        pred_vals = out["pred_vals"]
                        pre_m = out["pretrain_mask"].bool()
                        valid = (~pad)
                        mask = pre_m & valid
                        loss = (pred_vals.sum() * 0.0) if (not mask.any()) else ((pred_vals - values) ** 2)[mask].mean()
                        self.log("pretrain_val_loss", loss, on_epoch=True, prog_bar=True)

                    def configure_optimizers(self):
                        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)

                es_pre = EarlyStopping(
                    monitor="pretrain_val_loss",
                    mode="min",
                    patience=7,
                    min_delta=0.0,
                    check_on_train_epoch_end=False,
                    verbose=True,
                )

                trainer_pre = pl.Trainer(
                    max_epochs=cfg.max_epochs_pretrain,
                    accelerator=cfg.accelerator,
                    devices=cfg.devices,
                    precision=cfg.precision,
                    logger=pl.loggers.WandbLogger(experiment=wandb_run) if cfg.log_to_wandb else False,
                    callbacks=[es_pre],
                    enable_checkpointing=False,
                )

                pre_lit = PretrainLit(model_core, lr=cfg.lr, wd=cfg.weight_decay)
                trainer_pre.fit(pre_lit, ld_tr_pre, ld_va_pre)

            outcome_cols = cfg.outcome_cols if (cfg.outcome_cols and len(cfg.outcome_cols) > 0) else ("mortality",)

            lit = STRaTSLit(
                model=model_core,
                task_type="binary",      # P12/P19 y가 [B,1] 단일이라 binary 고정이 안전
                outcome_cols=outcome_cols,
                target_indices=[0],
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                optimizer=cfg.optimizer,
                scheduler=cfg.scheduler,
                scheduler_params=cfg.scheduler_params,
                pretrained_path=cfg.pretrained_path,
            )
            early_stop_cb = EarlyStopping(
                monitor="val_AUPRC",  # 네가 self.log("val_AUPRC", ...)로 찍는 키 이름이 이거여야 함
                mode="max",
                patience=7,
                min_delta=0.0,
                verbose=True,
            )
            callbacks = [early_stop_cb]

            trainer = pl.Trainer(
                max_epochs=cfg.max_epochs_train,
                callbacks=callbacks,
                accelerator=cfg.accelerator,
                devices=cfg.devices,
                precision=cfg.precision,
                logger=pl.loggers.WandbLogger(experiment=wandb_run) if cfg.log_to_wandb else False,
                enable_checkpointing=False,
            )

            # fit
            trainer.fit(lit, dm.train_dataloader(), dm.val_dataloader())

            # ✅ fold별 validation 성능 "한 번 더" 확실히 회수
            # (Lightning이 fit 중 val 로그를 남기지만, 여기선 dict로 꺼내기 위해 validate 호출이 깔끔함)
            val_out = trainer.validate(lit, dataloaders=dm.val_dataloader(), verbose=False)
            val_metrics = val_out[0] if (val_out and isinstance(val_out, list)) else {}

            test_out = trainer.test(lit, dataloaders=dm.test_dataloader(), verbose=False)
            test_metrics = test_out[0] if (test_out and isinstance(test_out, list)) else {}

            row = {"fold": fold, "split_path": split_path}
            row.update(_pick_metrics(val_metrics, ["val_AUROC", "val_AUPRC", "val_ACC"]))
            row.update(_pick_metrics(test_metrics, ["test_AUROC", "test_AUPRC", "test_ACC"]))
            fold_rows.append(row)

            # fold 단위 로그(옵션)
            if wandb_run:
                wandb.log({f"cv/fold": fold, **{f"cv/{k}": v for k, v in row.items() if k in metric_keys}})

            print(f"[CV] fold={fold} done | "
                  f"val_AUROC={row.get('val_AUROC', None)} "
                  f"val_AUPRC={row.get('val_AUPRC', None)} "
                  f"val_ACC={row.get('val_ACC', None)}")

        # ---- mean/std 리포트 ----
        summary = _mean_std(fold_rows, metric_keys)

        print("\n===== P12/P19 5-split CV Report =====")
        for k in metric_keys:
            if f"{k}_mean" in summary:
                print(f"{k}: {summary[f'{k}_mean']:.4f} ± {summary[f'{k}_std']:.4f}")
        print("=====================================\n")

        if wandb_run:
            wandb.log({f"cv_summary/{k}": v for k, v in summary.items()})
            # fold_rows를 table로도 올리고 싶으면:
            try:
                df = pd.DataFrame(fold_rows)
                wandb.log({"cv/fold_table": wandb.Table(dataframe=df)})
            except Exception:
                pass
            wandb.finish()

        return {
            "cv_folds": fold_rows,
            "cv_summary": summary,
        }


    if cfg.task_name == "ett":
        assert cfg.ett_csv_path is not None, "cfg.ett_csv_path required for ETT"
        assert cfg.ett_target_col is not None

        # 1) SOURCE train: scaler stats 같이 받기
        tr_s, tr_oc, outcome_cols, F, scaler_stats = build_ett_windows_as_samples(
            cfg.ett_csv_path, "train",
            lookback=cfg.ett_lookback, pred_len=cfg.ett_pred_len,
            features=cfg.ett_features, target_col=cfg.ett_target_col,
            scale=True, official_split=cfg.ett_official_split,
            scaler_stats=None,
            return_scaler_stats=True,
        )

        # 2) SOURCE val/test: source train stats로 transform
        va_s, va_oc, _, _, _ = build_ett_windows_as_samples(
            cfg.ett_csv_path, "val",
            lookback=cfg.ett_lookback, pred_len=cfg.ett_pred_len,
            features=cfg.ett_features, target_col=cfg.ett_target_col,
            scale=True, official_split=cfg.ett_official_split,
            scaler_stats=scaler_stats,
            return_scaler_stats=True,
        )
        te_s, te_oc, _, _, _ = build_ett_windows_as_samples(
            cfg.ett_csv_path, "test",
            lookback=cfg.ett_lookback, pred_len=cfg.ett_pred_len,
            features=cfg.ett_features, target_col=cfg.ett_target_col,
            scale=True, official_split=cfg.ett_official_split,
            scaler_stats=scaler_stats,
            return_scaler_stats=True,
        )

        num_features = F

        # 3) TARGET test만 (있으면)
        tgt_loader = None
        tgt_tag = None
        if getattr(cfg, "ett_target_csv_path", None):
            tgt_tag = os.path.basename(cfg.ett_target_csv_path).replace(".csv","")
            tgt_s, tgt_oc, _, F_t, _ = build_ett_windows_as_samples(
                cfg.ett_target_csv_path, "test",
                lookback=cfg.ett_lookback, pred_len=cfg.ett_pred_len,
                features=cfg.ett_features, target_col=cfg.ett_target_col,
                scale=True, official_split=cfg.ett_official_split,
                scaler_stats=scaler_stats,            # ✅ source train stats 고정
                return_scaler_stats=True,
            )
            if F_t != F:
                raise ValueError(f"Target feature dim mismatch: source F={F}, target F={F_t}")

        # 4) loaders
        (ds_tr, ds_va, ds_te), (ld_tr, ld_va, ld_te) = _make_loaders_random(
            tr_s, va_s, te_s,
            tr_oc, va_oc, te_oc,
            outcome_cols=outcome_cols,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

        if getattr(cfg, "ett_target_csv_path", None):
            tgt_ds = TimeSeriesDatasetETT(tgt_s, tgt_oc, outcome_cols=outcome_cols)
            tgt_loader = make_dataloader_random(
                tgt_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
            )
            wandb_run = None
        if cfg.log_to_wandb:
            project = "STraTS-ETT"
            wandb_run = wandb.init(
                project=cfg.project_name,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name,
                config={**asdict(cfg), "num_features": num_features}
            )
        if cfg.ett_features == "M":
            n_out = cfg.ett_pred_len * 7    # HUFL..OT 7개
        else:
            n_out = cfg.ett_pred_len        # S 모드

        cfg.model_cfg["n_output"] = n_out

        # 5) model (n_output = pred_len 강제)
        model_core = build_model(cfg.model_name, cfg.model_cfg or {}, num_features=num_features)

        # 6) inverse stats (OT 한 컬럼만 쓰므로 target_col의 mean/std만 뽑기)
        y_mean, y_std = None, None
        if scaler_stats is not None:
            mean, std = scaler_stats
            # target_col은 features=="M"이면 df_raw.columns[1:] 중에서 index 찾는 구조였지
            # build_ett_windows_as_samples 내부 cols_data 기준 인덱스이므로,
            # 여기서는 같은 방식으로 다시 계산하거나(간단), build 함수에서 tgt_idx도 반환하게 해도 됨.
            # ✅ 간단 버전: csv를 한번 읽어서 cols_data 만들고 인덱스 찾기
            df_tmp = pd.read_csv(cfg.ett_csv_path)
            cols_data = list(df_tmp.columns[1:]) if cfg.ett_features == "M" else [cfg.ett_target_col]
            tgt_idx = cols_data.index(cfg.ett_target_col)
            y_mean = float(mean[tgt_idx])
            y_std  = float(std[tgt_idx])

        lit = STRaTSLitForecast(
            model=model_core,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            optimizer=cfg.optimizer,
            y_mean=y_mean,
            y_std=y_std,
        )

        trainer = pl.Trainer(
            max_epochs=cfg.max_epochs_train,
            accelerator=cfg.accelerator,
            devices=cfg.devices,
            precision=cfg.precision,
            logger=pl.loggers.WandbLogger(name=cfg.wandb_run_name) if cfg.log_to_wandb else False,
            enable_checkpointing=False,
        )

        trainer.fit(lit, ld_tr, ld_va)

        # 7) source test (inverse metric 자동 로그됨)
        test_metrics = trainer.test(lit, dataloaders=ld_te, verbose=False)[0]

        # 8) target test도 로그 + 저장
        target_metrics = {}
        if tgt_loader is not None:
            target_metrics = lit.evaluate_dataloader(tgt_loader, stage="target")

        # 9) pred/y feather 저장 (source test + target test)
        os.makedirs("./rd_results", exist_ok=True)

        df_src = lit.collect_predictions_df(ld_te, stage="test", source_tag=os.path.basename(cfg.ett_csv_path).replace(".csv",""))
        if not df_src.empty:
            df_src.to_feather(f"./rd_results/{cfg.wandb_run_name}_preds_source_test.feather")

        if tgt_loader is not None:
            df_tgt = lit.collect_predictions_df(tgt_loader, stage="target", source_tag=tgt_tag)
            if not df_tgt.empty:
                df_tgt.to_feather(f"./rd_results/{cfg.wandb_run_name}_preds_target_test.feather")
        plot_path = f"./rd_results/{cfg.wandb_run_name}_test_first_window.png"
        lit.save_first_window_plot(
            ld_te,
            save_path=plot_path,
            title=f"{cfg.model_name} | {os.path.basename(cfg.ett_csv_path)}",
        )
        plot_path_tr = f"./rd_results/{cfg.wandb_run_name}_train_first_window.png"
        lit.save_first_window_plot(
            ld_tr,
            save_path=plot_path_tr,
            title=f"{cfg.model_name} | {os.path.basename(cfg.ett_csv_path)}",
        )
        print(f"[INFO] first test window plot saved: {plot_path}")

        return {"test_metrics": test_metrics, "target_metrics": target_metrics}


    # 1) Load raw splits
    train_df, train_st, train_oc = _load_split(cfg.raw_directory, cfg.source, "train", cfg.task_name, cfg.aug)
    valid_df, valid_st, valid_oc = _load_split(cfg.raw_directory, cfg.source, "valid", cfg.task_name, cfg.aug)
    test_df,  test_st,  test_oc  = _load_split(cfg.raw_directory, cfg.source, "test",  cfg.task_name,)

    num_features = int(train_df["itemid"].max()) 
    num_features = int(max(num_features, valid_df["itemid"].max(), test_df["itemid"].max())) + 1

    # 2) Build *raw* samples
    train_s_raw = prepare_dataset_samples(train_df, train_st)
    valid_s_raw = prepare_dataset_samples(valid_df, valid_st)
    test_s_raw  = prepare_dataset_samples(test_df,  test_st)

    # 3) Normalize using ONLY source train stats
    normed, stats = normalize_many(
        train_s_raw,
        categorical_itemids=cfg.categorical_itemids,
        train=train_s_raw,
        valid=valid_s_raw,
        test=test_s_raw,
    )
    train_s = normed["train"]
    valid_s = normed["valid"]
    test_s  = normed["test"]

    # 4) Datasets/loaders for source
    (ds_tr, ds_va, ds_te), (ld_tr, ld_va, ld_te) = _make_loaders(
        train_s, valid_s, test_s,
        train_oc, valid_oc, test_oc,
        outcome_cols=cfg.outcome_cols,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        make_pre_mask_train=False,
        pre_mask_p_train=cfg.pre_mask_p_train,
    )

    # 5) Target domain loader (normalize with SAME stats)
    target_loader = None
    if cfg.target_domain:
        t_df, t_st, t_oc = _load_split(cfg.raw_directory, cfg.target_domain, "test", cfg.task_name)
        t_s_raw = prepare_dataset_samples(t_df, t_st)
        target_s = normalize_single_samples(t_s_raw, stats)

        target_ds = TimeSeriesDataset(target_s, t_oc, outcome_cols=cfg.outcome_cols)
        target_loader = make_dataloader(
            target_ds,
            cfg.batch_size,
            num_workers=cfg.num_workers,
            make_pre_mask=False,
            pre_mask_p=0.0,
        )

    # 6) Build model
    model_core = build_model(cfg.model_name, cfg.model_cfg or {}, num_features=num_features)

    # 7) W&B init (옵션)
    wandb_run = None
    if cfg.log_to_wandb:
        wandb_run = wandb.init(
            project=cfg.project_name,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            config={**asdict(cfg), "num_features": num_features}
        )

    # --------- ✅ eval-only 모드: 훈련 스킵, 모델 로드 후 평가/저장만 ----------
    if cfg.eval_only:
        if not cfg.downstream_path or not os.path.exists(cfg.downstream_path):
            raise FileNotFoundError(f"[eval_only] downstream_path not found: {cfg.downstream_path}")

        # load weights
        state = torch.load(cfg.downstream_path, map_location="cpu")



        # Lightning module & trainer (logger는 선택적으로만 활성)
        # lit = STRaTSLit(
        #     model=model_core,
        #     task_type=cfg.task_type,
        #     outcome_cols=cfg.outcome_cols,
        #     target_indices=cfg.target_indices,
        #     lr=cfg.lr, weight_decay=cfg.weight_decay,
        #     optimizer=cfg.optimizer, scheduler=cfg.scheduler, scheduler_params=cfg.scheduler_params,
        #     pretrained_path=None,
        # )
        lit = STRaTSLit.load_from_checkpoint(checkpoint_path=cfg.downstream_path, model=model_core,)

        trainer = pl.Trainer(
            max_epochs=0,  # no training
            accelerator=cfg.accelerator,
            devices=cfg.devices,
            precision=cfg.precision,
            logger=(pl.loggers.WandbLogger(experiment=wandb_run) if wandb_run else False),
            enable_checkpointing=False,
        )

        # ---- Evaluate on TEST
        test_metrics = trainer.test(lit, dataloaders=ld_te, verbose=False)[0]

        # ---- Evaluate on TARGET (if any)
        target_metrics = {}
        
        if target_loader is not None:
            try:
                lit = lit.to("cuda")
                lit.model = lit.model.to("cuda")
                print("[CUDA] Model succesfully moved")
            except:
                print("[CUDA] Model failed to move, falling back to [CPU]")
            print("[DEBUG] lit.device:", lit.device)
            print("[DEBUG] model param device:", next(lit.model.parameters()).device)
            print("[DEBUG] cuda available:", torch.cuda.is_available())
            target_metrics = lit.evaluate_dataloader(target_loader, stage="target")

        # ---- Save embeddings/masks (재사용 helper)
        def _collect_embeddings_df(loader, source_tag: str) -> pd.DataFrame:
            device = lit.device
            lit.eval()
            emb_rows = []

            # outcome 이름(선택된 타깃만)
            sel_names = [lit.outcome_cols[i] for i in lit.target_indices]

            with torch.no_grad():
                for batch in loader:
                    batch = tuple(x.to(device) if torch.is_tensor(x) else x for x in batch)
                    times, varis, values, pad_mask, pre_mask, y, pid, static = batch

                    out = lit.model(
                        times=times, varis=varis, values=values, statics=static,
                        padding_mask=pad_mask, pretrain=False
                    )

                    # embeddings
                    E = out["embs"].detach().cpu().numpy()                         # [B, D]
                    P = pid.detach().cpu().numpy().astype(int)                    # [B]

                    # true labels (selected)
                    Y_all = y[:, lit.target_indices].detach().cpu().numpy()        # [B, K] or [B] depending on y
                    if Y_all.ndim == 1:
                        Y_all = Y_all[:, None]                                    # [B,1]

                    # predictions
                    pred = out["pred"].detach().cpu()
                    if pred.ndim == 1:
                        # binary or single-output: [B] -> [B,1]
                        pred_np = pred.numpy()[:, None]
                    else:
                        # multilabel: should already be [B,K]
                        pred_np = pred.numpy()

                    # sanity: align K
                    K = len(sel_names)
                    if pred_np.shape[1] != K:
                        raise ValueError(
                            f"Pred dim mismatch: pred has shape {pred_np.shape} but target_indices implies K={K}."
                        )

                    df_e = pd.DataFrame(E, columns=[f"e{i}" for i in range(E.shape[1])])

                    # add true labels
                    for j, name in enumerate(sel_names):
                        df_e[name] = Y_all[:, j]

                    # add predictions
                    for j, name in enumerate(sel_names):
                        df_e[f"pred_{name}"] = pred_np[:, j]

                    df_e.insert(0, "pid", P)
                    df_e["source"] = source_tag
                    emb_rows.append(df_e)

            return pd.concat(emb_rows, axis=0, ignore_index=True) if emb_rows else pd.DataFrame()
        # Output dir
        output_dir = getattr(cfg, "output_directory", "./outputs")
        os.makedirs(output_dir, exist_ok=True)

        # 10) Save embeddings (+ outcomes) as DataFrames
        df_test_emb = _collect_embeddings_df(ld_te, f"{cfg.source}")
        test_emb_path = f"./rd_results/{cfg.wandb_run_name}_embeddings_test.feather"
        if not df_test_emb.empty:
            df_test_emb.to_feather(test_emb_path)
            print(f"[INFO] embedding+outcomes (test) saved: {test_emb_path}")
        else:
            print("[WARN] no embeddings collected for test.")

        if target_loader is not None:
            df_tgt_emb = _collect_embeddings_df(target_loader, f"{cfg.target_domain}")
            tgt_emb_path = f"./rd_results/{cfg.wandb_run_name}_embeddings_target.feather"
            if not df_tgt_emb.empty:
                df_tgt_emb.to_feather(tgt_emb_path)
                print(f"[INFO] embedding+outcomes (target) saved: {tgt_emb_path}")
            else:
                print("[WARN] no embeddings collected for target.")

        # 11) If surprise models: dump inputs + surprise mask as DataFrames (full loader)
        def _save_check_padding_df(loader, tag: str):
            device = lit.device
            rows = []
            with torch.no_grad():
                for batch in loader:
                    times, varis, values, pad_mask, pre_mask, y, pid, static = [
                        x.to(device) if torch.is_tensor(x) else x for x in batch
                    ]
                    cp = lit.model.check_padding(
                        times=times, varis=varis, values=values, padding_mask=pad_mask
                    )
                    times_np  = cp["times"].detach().cpu().numpy()
                    varis_np  = cp["varis"].detach().cpu().numpy()
                    values_np = cp["values"].detach().cpu().numpy()
                    mask_np   = cp["mask"].detach().cpu().numpy()        # True = gated out
                    pad_np    = pad_mask.detach().cpu().numpy()
                    pid_np    = pid.detach().cpu().numpy()

                    B, L = times_np.shape
                    for b in range(B):
                        valid_idx = np.nonzero(~pad_np[b])[0]
                        if valid_idx.size == 0:
                            continue
                        p = int(pid_np[b])
                        for t in valid_idx:
                            rows.append({
                                "pid":   p,
                                "times": float(times_np[b, t]),
                                "varis": int(varis_np[b, t]),
                                "values": float(values_np[b, t]),
                                "mask":  bool(mask_np[b, t]),  # True=gated
                            })

            if rows:
                df = pd.DataFrame(rows, columns=["pid","times","varis","values","mask"])
                path = f"./rd_results/{cfg.wandb_run_name}_mask_{tag}.feather"
                df.to_feather(path)
                print(f"[INFO] check_padding DataFrame saved: {path}")
            else:
                print(f"[WARN] no valid tokens for check_padding ({tag}); nothing saved.")

        if cfg.model_name in {"surprise", "surprise_vt"}:
            _save_check_padding_df(ld_te, f"{cfg.source}")
            if target_loader is not None:
                _save_check_padding_df(target_loader, f"{cfg.target_domain}")
        else:
            print("[INFO] check_padding skip: model is not surprise/surprise_vt")

        pid_src = 26373139
        pid_tgt = 268487

        def _save_pid_scores(loader, tag, target_pid):
            one = _fetch_one_pid_from_loader(loader, target_pid=target_pid, device=lit.device)

            # 모델 타입에 따라 score 함수 선택
            # 1) SurpriseSTraTS (triplet)
            if cfg.model_name == "surprise":
                direction = (cfg.model_cfg.get("surprise_args", {}) or {}).get("direction", "future")
                tok_df, mat_df = _surp_triplet_score_df(lit.model, one, direction=direction)

            # 2) SurpriseSTraTS_VT (VT or VTTG)
            elif cfg.model_name == "surprise_vt":
                vt_args = (cfg.model_cfg.get("vt_mask_args", {}) or {})
                direction = vt_args.get("direction", "future")

                if getattr(cfg, "model_cfg", {}).get("use_timegap_surprise", False):
                    # VTTG score (value_sim * (1-dt))
                    tok_df, mat_df = _vttg_score_df(lit.model, one, direction=direction)
                else:
                    # VT score (value_sim + time_sim)
                    tok_df, mat_df = _vt_score_df(
                        lit.model, one, direction=direction,
                        tau_v=vt_args.get("tau_v", 1.0),
                        tau_t=vt_args.get("tau_t", 1.0),
                        w_v=vt_args.get("w_v", 1.0),
                        w_t=vt_args.get("w_t", 1.0),
                    )
            else:
                print(f"[INFO] pid score save skipped: model_name={cfg.model_name} not supported")
                return

            # token-level mask 붙이기(가능하면)
            mdf = _token_mask_df_one_pid(lit.model, one)
            if mdf is not None:
                tok_df = tok_df.merge(mdf, on="t", how="left")

            # 저장
            base = cfg.wandb_run_name or "run"
            tok_path = f"./rd_results/{base}_{tag}_pid{target_pid}_tokens.feather"
            mat_path = f"./rd_results/{base}_{tag}_pid{target_pid}_scores.feather"
            tok_df.to_feather(tok_path)
            mat_df.to_feather(mat_path)
            print(f"[INFO] saved token df: {tok_path}")
            print(f"[INFO] saved score df : {mat_path}")

        # source test pid
        _save_pid_scores(ld_te, f"{cfg.source}_test", pid_src)

        # target test pid (있을 때만)
        if target_loader is not None:
            _save_pid_scores(target_loader, f"{cfg.target_domain}_target", pid_tgt)

            # W&B 로깅 (옵션)
            if wandb_run:
                wandb.log({"test_metrics": test_metrics, "target_metrics": target_metrics})
                wandb.finish()

            print(f"[INFO] eval-only finished. Results saved in {output_dir}")
            return {
                "test_metrics": test_metrics,
                "target_metrics": target_metrics,
                "norm_stats": stats,
            }

    # 6) (Optional) Pretrain (self-supervised)
    if cfg.max_epochs_pretrain > 0:
        if cfg.pretrained_path is not None and os.path.exists(cfg.pretrained_path):
            print(f"[INFO] Found existing pretrained checkpoint at {cfg.pretrained_path}. "
                f"Skipping pretraining and loading *backbone* weights.")

            ckpt = torch.load(cfg.pretrained_path, map_location="cpu")
            model_state = model_core.state_dict()

            filtered = {}
            skipped = []

            for k, v in ckpt.items():
                # 1) downstream head (classifier)는 공유 안 함
                if k.startswith("downstream_head."):
                    skipped.append((k, "head"))
                    continue

                # 2) 키는 같은데 shape이 다르면 (예: n_output=1 vs n_output=25) 그냥 스킵
                if k not in model_state:
                    skipped.append((k, "missing_in_model"))
                    continue
                if model_state[k].shape != v.shape:
                    skipped.append((k, f"shape_mismatch {v.shape} != {model_state[k].shape}"))
                    continue

                filtered[k] = v

            # debug용 로그 (원하면 줄여도 됨)
            print(f"[INFO] Loading {len(filtered)} params from pretrained, "
                f"skipping {len(skipped)} (head or mismatch).")
            # for k, reason in skipped:
            #     print(f"  - skip {k}: {reason}")

            model_state.update(filtered)
            # strict=False: 일부 키는 남겨둬도 됨 (head는 현재 모델 구조 사용)
            model_core.load_state_dict(model_state, strict=False)
        else:
            # ✅ 기존 pretrain 로직 그대로 수행
            class PretrainLit(pl.LightningModule):
                def __init__(self, model, lr=1e-3, wd=1e-2):
                    super().__init__()
                    self.model = model
                    self.lr = lr
                    self.wd = wd

                def training_step(self, batch, _):
                    times, varis, values, pad, pre, _, _, static = batch

                    out = self.model(
                        times=times, varis=varis, values=values, statics=static,
                        padding_mask=pad, pretrain=True, pretrain_mask=pre
                    )

                    pred_vals = out["pred_vals"]
                    pre_m = out["pretrain_mask"].bool()
                    valid = (~pad)  # pad only
                    mask = pre_m & valid

                    # ✅ 반드시 masked 위치에서만
                    if not mask.any():
                        # 마스크가 비어있으면 이 배치는 스킵 (loss 0으로)
                        loss = (pred_vals.sum() * 0.0)
                    else:
                        loss = ((pred_vals - values) ** 2)[mask].mean()

                    self.log("pretrain_train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
                    self.log("pretrain_mask_ratio", mask.float().mean(), on_step=True, on_epoch=False, prog_bar=False)
                    return loss

                def validation_step(self, batch, _):
                    times, varis, values, pad, pre, _, _, static = batch
                    out = self.model(
                        times=times, varis=varis, values=values, statics=static,
                        padding_mask=pad, pretrain=True, pretrain_mask=pre
                    )
                    pred_vals = out["pred_vals"]
                    pre_m = out["pretrain_mask"].bool()
                    valid = (~pad)
                    mask = pre_m & valid
                    self.log("val_pad_ratio", pad.float().mean(), on_epoch=True)
                    self.log("val_valid_ratio", valid.float().mean(), on_epoch=True)
                    self.log("val_pre_mask_ratio", pre_m.float().mean(), on_epoch=True)
                    self.log("val_mask_used_ratio", mask.float().mean(), on_epoch=True)


                    if not mask.any():
                        loss = (pred_vals.sum() * 0.0)
                    else:
                        loss = ((pred_vals - values) ** 2)[mask].mean()

                    self.log("pretrain_val_loss", loss, on_epoch=True, prog_bar=True)

                def configure_optimizers(self):
                    return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)

            # pretrain용 DataLoader (train은 mask ON, val은 mask OFF 권장)
            ld_tr_pre = make_dataloader(
                ds_tr, cfg.batch_size, num_workers=cfg.num_workers,
                make_pre_mask=True, pre_mask_p=cfg.pre_mask_p_train
            )
            ld_va_pre = make_dataloader(
                ds_va, cfg.batch_size, num_workers=cfg.num_workers,
                make_pre_mask=True, pre_mask_p=cfg.pre_mask_p_train
            )

            pre_lit = PretrainLit(model_core, lr=cfg.lr, wd=cfg.weight_decay)

            # ✅ EarlyStopping for pretrain
            es_pre = EarlyStopping(
                monitor="pretrain_val_loss",
                mode="min",
                patience=5,
                min_delta=0.0,
                check_on_train_epoch_end=False,
                verbose=True,
            )

            pre_ckpt_cb = SaveOnEpochs(
                epochs_to_save=[1, 5, 10, 15, 25, 40],
                dirpath=f"./rd_models/pre_ckpts/{cfg.wandb_run_name}",
                prefix=f"{cfg.wandb_run_name}_pretrain",
                save_weights_only=True,
            )

            trainer_pre = pl.Trainer(
                max_epochs=cfg.max_epochs_pretrain,
                accelerator=cfg.accelerator,
                devices=cfg.devices,
                precision=cfg.precision,
                logger=pl.loggers.WandbLogger(experiment=wandb_run) if cfg.log_to_wandb else False,
                callbacks=[es_pre, pre_ckpt_cb],
                enable_checkpointing=False,
            )
            trainer_pre.fit(pre_lit, ld_tr_pre, ld_va_pre)

            # 새로 pretrain 돌렸을 때만 저장
            if cfg.save_pretrained_after and cfg.pretrained_path:
                os.makedirs(os.path.dirname(cfg.pretrained_path), exist_ok=True)
                torch.save(model_core.state_dict(), cfg.pretrained_path)
                print(f"[INFO] Saved pretrained model to: {cfg.pretrained_path}")

    # 7) Downstream training (binary or multilabel)
    es_monitor = 'val_AUPRC' if cfg.task_type == 'binary' else 'val_AUPRC_macro'
    
    early_stop_cb = EarlyStopping(
        monitor=es_monitor,
        mode='max',
        patience=7,
        check_on_train_epoch_end=False,  # 매 epoch의 validation 후 체크
        verbose=True,
    )
    lit = STRaTSLit(
        model=model_core,
        task_type=cfg.task_type,
        outcome_cols=cfg.outcome_cols,
        target_indices=cfg.target_indices,
        lr=cfg.lr, weight_decay=cfg.weight_decay,
        optimizer=cfg.optimizer, scheduler=cfg.scheduler, scheduler_params=cfg.scheduler_params,
        pretrained_path=cfg.pretrained_path,
    )
    # Load pretrained model if pretrained_path true
    ckpt_cb = SaveOnEpochs(
        epochs_to_save=[1, 5, 10, 15, 25],
        dirpath=f"./rd_models/ckpts/{cfg.wandb_run_name}",
        prefix=f"{cfg.wandb_run_name}",
        save_weights_only=True,   # True면 state_dict만 저장 (가벼움)
    )

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs_train,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        logger=pl.loggers.WandbLogger(experiment=wandb_run),
        callbacks=[early_stop_cb, ckpt_cb],
        enable_checkpointing=False,   # ✅ 켜야 trainer.save_checkpoint가 정상 동작 / 260105 이거 키면 epoch 지나고 느려짐
    )# 정확히는 enable_checkpointing이랑 checkpoint callback이 둘 다 있으면 뭔가 누수가 생기는듯?
    trainer.fit(lit, ld_tr, ld_va)

    if cfg.downstream_path is not None:
        torch.save(lit.model.state_dict(), cfg.downstream_path)

        print(f"[INFO] Saved downstream model to: {cfg.downstream_path}")

    # 8) Evaluate on TEST
    test_metrics = trainer.test(lit, dataloaders=ld_te, verbose=False)[0]

    # 9) Evaluate on TARGET if any
    target_metrics = {}
    if target_loader is not None:
        try:
            lit = lit.to("cuda")
            lit.model = lit.model.to("cuda")
            print("[CUDA] Model succesfully moved")
        except:
            print("[CUDA] Model failed to move, falling back to [CPU]")
        print("[DEBUG] lit.device:", lit.device)
        print("[DEBUG] model param device:", next(lit.model.parameters()).device)
        print("[DEBUG] cuda available:", torch.cuda.is_available())
        target_metrics = lit.evaluate_dataloader(target_loader, stage="target")

    def _collect_embeddings_df(loader, source_tag: str) -> pd.DataFrame:
        device = lit.device
        lit.eval()
        emb_rows = []

        # outcome 이름(선택된 타깃만)
        sel_names = [lit.outcome_cols[i] for i in lit.target_indices]

        with torch.no_grad():
            for batch in loader:
                batch = tuple(x.to(device) if torch.is_tensor(x) else x for x in batch)
                times, varis, values, pad_mask, pre_mask, y, pid, static = batch

                out = lit.model(
                    times=times, varis=varis, values=values, statics=static,
                    padding_mask=pad_mask, pretrain=False
                )

                # embeddings
                E = out["embs"].detach().cpu().numpy()                         # [B, D]
                P = pid.detach().cpu().numpy().astype(int)                    # [B]

                # true labels (selected)
                Y_all = y[:, lit.target_indices].detach().cpu().numpy()        # [B, K] or [B] depending on y
                if Y_all.ndim == 1:
                    Y_all = Y_all[:, None]                                    # [B,1]

                # predictions
                pred = out["pred"].detach().cpu()
                if pred.ndim == 1:
                    # binary or single-output: [B] -> [B,1]
                    pred_np = pred.numpy()[:, None]
                else:
                    # multilabel: should already be [B,K]
                    pred_np = pred.numpy()

                # sanity: align K
                K = len(sel_names)
                if pred_np.shape[1] != K:
                    raise ValueError(
                        f"Pred dim mismatch: pred has shape {pred_np.shape} but target_indices implies K={K}."
                    )

                df_e = pd.DataFrame(E, columns=[f"e{i}" for i in range(E.shape[1])])

                # add true labels
                for j, name in enumerate(sel_names):
                    df_e[name] = Y_all[:, j]

                # add predictions
                for j, name in enumerate(sel_names):
                    df_e[f"pred_{name}"] = pred_np[:, j]

                df_e.insert(0, "pid", P)
                df_e["source"] = source_tag
                emb_rows.append(df_e)

        return pd.concat(emb_rows, axis=0, ignore_index=True) if emb_rows else pd.DataFrame()
    # Output dir
    output_dir = getattr(cfg, "output_directory", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    # 10) Save embeddings (+ outcomes) as DataFrames
    df_test_emb = _collect_embeddings_df(ld_te, f"{cfg.source}")
    test_emb_path = f"./rd_results/{cfg.wandb_run_name}_embeddings_test.feather"
    if not df_test_emb.empty:
        df_test_emb.to_feather(test_emb_path)
        print(f"[INFO] embedding+outcomes (test) saved: {test_emb_path}")
    else:
        print("[WARN] no embeddings collected for test.")

    if target_loader is not None:
        df_tgt_emb = _collect_embeddings_df(target_loader, f"{cfg.target_domain}")
        tgt_emb_path = f"./rd_results/{cfg.wandb_run_name}_embeddings_target.feather"
        if not df_tgt_emb.empty:
            df_tgt_emb.to_feather(tgt_emb_path)
            print(f"[INFO] embedding+outcomes (target) saved: {tgt_emb_path}")
        else:
            print("[WARN] no embeddings collected for target.")

    # 11) If surprise models: dump inputs + surprise mask as DataFrames (full loader)
    def _save_check_padding_df(loader, tag: str):
        device = lit.device
        rows = []
        with torch.no_grad():
            for batch in loader:
                times, varis, values, pad_mask, pre_mask, y, pid, static = [
                    x.to(device) if torch.is_tensor(x) else x for x in batch
                ]
                cp = lit.model.check_padding(
                    times=times, varis=varis, values=values, padding_mask=pad_mask
                )
                times_np  = cp["times"].detach().cpu().numpy()
                varis_np  = cp["varis"].detach().cpu().numpy()
                values_np = cp["values"].detach().cpu().numpy()
                mask_np   = cp["mask"].detach().cpu().numpy()        # True = gated out
                pad_np    = pad_mask.detach().cpu().numpy()
                pid_np    = pid.detach().cpu().numpy()

                B, L = times_np.shape
                for b in range(B):
                    valid_idx = np.nonzero(~pad_np[b])[0]
                    if valid_idx.size == 0:
                        continue
                    p = int(pid_np[b])
                    for t in valid_idx:
                        rows.append({
                            "pid":   p,
                            "times": float(times_np[b, t]),
                            "varis": int(varis_np[b, t]),
                            "values": float(values_np[b, t]),
                            "mask":  bool(mask_np[b, t]),  # True=gated
                        })

        if rows:
            df = pd.DataFrame(rows, columns=["pid","times","varis","values","mask"])
            path = f"./rd_results/{cfg.wandb_run_name}_mask_{tag}.feather"
            df.to_feather(path)
            print(f"[INFO] check_padding DataFrame saved: {path}")
        else:
            print(f"[WARN] no valid tokens for check_padding ({tag}); nothing saved.")

    if cfg.model_name in {"surprise", "surprise_vt"}:
        _save_check_padding_df(ld_te, f"{cfg.source}")
        if target_loader is not None:
            _save_check_padding_df(target_loader, f"{cfg.target_domain}")
    else:
        print("[INFO] check_padding skip: model is not surprise/surprise_vt")
    if cfg.source=="mimic": # Hardcoded ids. 
        pid_src = 26373139
        pid_tgt = 268487
    else:
        pid_tgt = 26373139
        pid_src = 268487 

    def _save_pid_scores(loader, tag, target_pid):
        one = _fetch_one_pid_from_loader(loader, target_pid=target_pid, device=lit.device)

        # 모델 타입에 따라 score 함수 선택
        # 1) SurpriseSTraTS (triplet)
        if cfg.model_name == "surprise":
            direction = (cfg.model_cfg.get("surprise_args", {}) or {}).get("direction", "future")
            tok_df, mat_df = _surp_triplet_score_df(lit.model, one, direction=direction)

        # 2) SurpriseSTraTS_VT (VT or VTTG)
        elif cfg.model_name == "surprise_vt":
            vt_args = (cfg.model_cfg.get("vt_mask_args", {}) or {})
            direction = vt_args.get("direction", "future")

            if getattr(cfg, "model_cfg", {}).get("use_timegap_surprise", False):
                # VTTG score (value_sim * (1-dt))
                tok_df, mat_df = _vttg_score_df(lit.model, one, direction=direction)
            else:
                # VT score (value_sim + time_sim)
                tok_df, mat_df = _vt_score_df(
                    lit.model, one, direction=direction,
                    tau_v=vt_args.get("tau_v", 1.0),
                    tau_t=vt_args.get("tau_t", 1.0),
                    w_v=vt_args.get("w_v", 1.0),
                    w_t=vt_args.get("w_t", 1.0),
                )
        else:
            print(f"[INFO] pid score save skipped: model_name={cfg.model_name} not supported")
            return

        # token-level mask 붙이기(가능하면)
        mdf = _token_mask_df_one_pid(lit.model, one)
        if mdf is not None:
            tok_df = tok_df.merge(mdf, on="t", how="left")

        # 저장
        base = cfg.wandb_run_name or "run"
        tok_path = f"./rd_results/{base}_{tag}_pid{target_pid}_tokens.feather"
        mat_path = f"./rd_results/{base}_{tag}_pid{target_pid}_scores.feather"
        tok_df.to_feather(tok_path)
        mat_df.to_feather(mat_path)
        print(f"[INFO] saved token df: {tok_path}")
        print(f"[INFO] saved score df : {mat_path}")

    # source test pid
    try:
        _save_pid_scores(ld_te, f"{cfg.source}_test", pid_src)

        # target test pid (있을 때만)
        if target_loader is not None:
            _save_pid_scores(target_loader, f"{cfg.target_domain}_target", pid_tgt)
    except:
        print(f"Saving pid score failed. Check if pid source {pid_src} and target {pid_tgt} is correctly in test / target")

    # 12) wrap up (no PaCMAP logs)
    wandb.log({"test_metrics": test_metrics, "target_metrics": target_metrics})
    log_dict = {
            "test/source_auroc": test_metrics['test_AUROC_macro'],
            "test/source_auprc": test_metrics['test_AUPRC_macro'],
            "test/target_auroc": target_metrics['target_AUROC_macro'],
            "test/target_auprc": target_metrics['target_AUPRC_macro'],
        }
    wandb.log(log_dict)
    wandb.finish()
    print(f"[INFO] run finished. Results saved in {output_dir}")

    return {
        "test_metrics": test_metrics,
        "target_metrics": target_metrics,
        "norm_stats": stats,
    }


def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ("yes","true","t","y","1"):   return True
    if v in ("no","false","f","n","0"):   return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def build_cfg_from_args(args: argparse.Namespace) -> RunConfig:
    # model_cfg_json -> dict
    model_cfg = json.loads(args.model_cfg_json)

    # outcome_cols: ICU면 사용, ETT면 무시해도 됨(ETT는 내부에서 y_0.. 생성)
    outcome_cols_tuple = tuple([c for c in args.outcome_cols.split(",") if c != ""]) if hasattr(args, "outcome_cols") else tuple()

    # bool 파싱
    eval_only = str2bool(args.eval_only) if hasattr(args, "eval_only") else False
    log_to_wandb = str2bool(args.log_to_wandb) if hasattr(args, "log_to_wandb") else True
    ett_official_split = str2bool(args.ett_official_split) if hasattr(args, "ett_official_split") else True
    aug = str2bool(args.aug) if hasattr(args, "aug") else False

    cfg = RunConfig(
        source=args.source,
        target_domain=args.target_domain,

        model_name=args.model_name,
        model_cfg=model_cfg,

        task_type=args.task_type,
        task_name=args.task_name,
        outcome_cols=outcome_cols_tuple,
        target_indices=None,

        raw_directory="./data",

        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),

        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        optimizer="adamw",
        scheduler="none",
        scheduler_params=None,

        max_epochs_pretrain=int(args.max_epochs_pretrain),
        max_epochs_train=int(args.max_epochs_train),

        pre_mask_p_train=0.15,
        categorical_itemids=[i for i in range(19, 35)],

        pretrained_path=args.pretrained_path,
        save_pretrained_after=True,
        downstream_path=args.downstream_path,

        project_name=args.project_name,
        wandb_entity=None,
        wandb_run_name=args.wandb_run_name,
        aug = aug,

        accelerator="auto",
        devices="auto",
        precision="16-mixed",

        # ✅ new runtime flags
        eval_only=eval_only,
        log_to_wandb=log_to_wandb,
        output_directory=args.output_directory,

        # ✅ ETT fields
        ett_csv_path=(args.ett_csv_path if args.ett_csv_path else None),
        ett_lookback=int(args.ett_lookback),
        ett_pred_len=int(args.ett_pred_len),
        ett_features=args.ett_features,
        ett_target_col=args.ett_target_col,
        ett_official_split=ett_official_split,

        # ✅ physionet fields
        p12p19_dataset=(args.p12p19_dataset if args.p12p19_dataset else None),
        p12p19_base_path=(args.p12p19_base_path if args.p12p19_base_path else None),
        p12p19_split_path=(args.p12p19_split_path if args.p12p19_split_path else None),
        p12p19_num_splits=int(args.p12p19_num_splits),
        p12p19_split_path_pattern=(args.p12p19_split_path_pattern if args.p12p19_split_path_pattern else None),
        p12p19_split_type=args.p12p19_split_type,
        p12p19_reverse=str2bool(args.p12p19_reverse),
        p12p19_baseline=str2bool(args.p12p19_baseline),
        p12p19_predictive_label=args.p12p19_predictive_label,
        p12p19_max_tokens=int(args.p12p19_max_tokens),        
    )
        

    # ETT 타겟 CSV는 target_domain에 안 싣고 별도로 운반하고 싶으면:
    # - RunConfig에 필드 추가하는 게 정석이지만
    # - 지금은 최소수정으로 target_domain에 넣어도 됨
    # 여기서는 args.ett_target_csv_path를 cfg.target_domain에 강제로 태움 (ETT에서만 의미 있음)
    if cfg.task_name == "ett":
        if args.ett_target_csv_path:
            cfg.target_domain = args.ett_target_csv_path
        # ETT는 downstream 분류 outcome_cols 사실상 안 씀
        cfg.outcome_cols = tuple()

    return cfg


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--source", required=True)
    p.add_argument("--target_domain", required=True)

    p.add_argument("--model_name", required=True)
    p.add_argument("--model_cfg_json", required=True)

    p.add_argument("--task_type", required=True, choices=["binary", "multilabel", "forecast"])
    p.add_argument("--task_name", required=True)

    # comma-separated, e.g. "mor_label" or "mor_label,readm_label"
    p.add_argument("--outcome_cols", required=True)

    p.add_argument("--batch_size", default="8")
    p.add_argument("--max_epochs_pretrain", default="0")
    p.add_argument("--max_epochs_train", default="30")

    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--downstream_path", required=True)
    p.add_argument("--aug", default=False)
    p.add_argument("--wandb_run_name", required=True)
    p.add_argument("--project_name", required=True, help="wandb project name (e.g., Surprise-P12, Surprise-P19)")
    # ---------- common runtime options ----------
    p.add_argument("--num_workers", default="0")
    p.add_argument("--lr", default="1e-3")
    p.add_argument("--weight_decay", default="1e-2")
    p.add_argument("--eval_only", default="False")
    p.add_argument("--log_to_wandb", default="True")
    p.add_argument("--output_directory", default="./outputs")

    # ---------- ETT options ----------
    p.add_argument("--ett_csv_path", default="")
    p.add_argument("--ett_target_csv_path", default="")  # target test용
    p.add_argument("--ett_lookback", default="96")
    p.add_argument("--ett_pred_len", default="96")
    p.add_argument("--ett_features", default="M", choices=["S","M"])
    p.add_argument("--ett_target_col", default="OT")
    p.add_argument("--ett_official_split", default="True")

    # ---------- P12/P19/PAM options ----------
    p.add_argument("--p12p19_dataset", default="", choices=["", "P12", "P19", "PAM"])
    p.add_argument("--p12p19_base_path", default="")   # e.g., ./processed_data/P19data  (NOT .../processed_data)
    p.add_argument("--p12p19_split_path", default="")  # e.g., /splits/phy19_split1_new.npy
    p.add_argument("--p12p19_num_splits", default="5")
    p.add_argument("--p12p19_split_path_pattern", default="")
    p.add_argument("--p12p19_split_type", default="random", choices=["random","age","gender"])
    p.add_argument("--p12p19_reverse", default="False")
    p.add_argument("--p12p19_baseline", default="False")
    p.add_argument("--p12p19_predictive_label", default="mortality", choices=["mortality","LoS"])
    p.add_argument("--p12p19_max_tokens", default="4096")
    # seed은 Lightning seed_everything에 넣는다
    p.add_argument("--seed", default="42")

    args = p.parse_args()

    # reproducibility
    pl.seed_everything(int(args.seed), workers=True)

    cfg = build_cfg_from_args(args)

    # run
    result = run_pipeline(cfg)

    # useful prints at the end
    print("=== DONE ===")
    print("source:", cfg.source, "task:", cfg.task_name, "model:", cfg.model_name)
    if "test_metrics" in result:
        print("test_metrics:", result["test_metrics"])
    if "target_metrics" in result:
        print("target_metrics:", result["target_metrics"])