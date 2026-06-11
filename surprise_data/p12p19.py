from __future__ import annotations

import numpy as np
import torch
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Iterable, Optional, Literal, Sequence, Any, Union
from torch.utils.data import Dataset, DataLoader
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
                 batch_size: int = 128, num_workers: int = 4, max_tokens: int = 4096,
                 pre_mask_p_train: float = 0.15):
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
        self.pre_mask_p_train = float(pre_mask_p_train)

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
        self.static_dim = D

        if self.dataset == "P12":
            time_max = 2880 / 60 # Tensorize_normalize already turns to hours (divide by 60)
        elif self.dataset == "P19":
            time_max = 60 / 60
        else:
            time_max = 1.0
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
                pre_mask_p=self.pre_mask_p_train,
            ),
            persistent_workers=(self.num_workers > 0),
        )

    def pretrain_val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=make_strats_collate_pretrain(
                padding_idx=self.num_features,
                pre_mask_p=self.pre_mask_p_train,),
            persistent_workers=(self.num_workers > 0),
        )
