from __future__ import annotations

import os
import math
from functools import partial
from typing import Dict, List, Tuple, Iterable, Optional, Sequence, Literal

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, Sampler


# =========================
# Sample prep / normalization
# =========================

Sample = Tuple[int, np.ndarray, np.ndarray]  # (pid, obs[N,3], static[3])


def prepare_dataset_samples(
    data: pd.DataFrame,
    static: pd.DataFrame,
    *,
    sort_obs_by: Iterable[str] = ("offset", "itemid"),
    dropna_in_obs_value: bool = True,
    cast_itemid_to_int: bool = True,
    dtypes: Tuple[str, str] = ("float32", "float32"),  # (obs dtype, static dtype)
) -> List[Sample]:
    """
    Returns list of (pid, obs_matrix, static_vec):
      - obs_matrix: (Ni,3) rows [itemid, offset, value]
      - static_vec: (3,) [age, gender, height]
    """
    obs_dtype, static_dtype = dtypes

    req_data = {"pid", "itemid", "offset", "value"}
    req_static = {"pid", "age", "gender", "height"}
    md = req_data - set(data.columns)
    ms = req_static - set(static.columns)
    if md:
        raise ValueError(f"`data` missing columns: {sorted(md)}")
    if ms:
        raise ValueError(f"`static` missing columns: {sorted(ms)}")

    static_slim = static.loc[:, ["pid", "age", "gender", "height"]].copy()
    for col in ["age", "gender", "height"]:
        static_slim[col] = pd.to_numeric(static_slim[col], errors="coerce")
    static_slim["gender"] = static_slim["gender"].clip(0, 1)

    static_slim = static_slim.dropna(subset=["age", "gender", "height"])
    static_slim = static_slim.drop_duplicates(subset=["pid"], keep="first")

    static_vec_lookup = {
        int(pid): np.array([row.age, row.gender, row.height], dtype=static_dtype)
        for pid, row in static_slim.set_index("pid").iterrows()
    }

    data_clean = data.loc[:, ["pid", "itemid", "offset", "value"]].copy()
    if cast_itemid_to_int:
        data_clean["itemid"] = pd.to_numeric(data_clean["itemid"], errors="coerce")
    data_clean["offset"] = pd.to_numeric(data_clean["offset"], errors="coerce")
    data_clean["value"]  = pd.to_numeric(data_clean["value"],  errors="coerce")
    data_clean["pid"]    = pd.to_numeric(data_clean["pid"],    errors="coerce")

    if dropna_in_obs_value:
        data_clean = data_clean.dropna(subset=["pid", "itemid", "offset", "value"])

    data_clean["pid"] = data_clean["pid"].astype(int)
    sort_cols = ["pid"] + list(sort_obs_by)
    data_clean = data_clean.sort_values(sort_cols, kind="mergesort")

    samples: List[Sample] = []
    for pid, df_pid in data_clean.groupby("pid", sort=False):
        static_vec = static_vec_lookup.get(pid, None)
        if static_vec is None:
            continue

        obs_mat = df_pid.loc[:, ["itemid", "offset", "value"]].to_numpy(dtype=obs_dtype)
        if obs_mat.shape[0] == 0:
            continue

        samples.append((int(pid), obs_mat, static_vec))

    return samples


def compute_norm_stats(
    reference_samples: List[Sample],
    categorical_itemids: Iterable[float | int] = ()
) -> Dict:
    cat_set = {float(x) for x in categorical_itemids}

    per_item_sums, per_item_sq, per_item_cnt = {}, {}, {}
    global_sum = 0.0
    global_sq  = 0.0
    global_cnt = 0

    ages, heights = [], []

    for _, obs, st in reference_samples:
        ages.append(float(st[0]))
        heights.append(float(st[2]))

        if obs.size == 0:
            continue

        itemids = obs[:, 0]
        values  = obs[:, 2]
        for it, v in zip(itemids, values):
            it = float(it); v = float(v)
            if it in cat_set:
                continue
            per_item_sums[it] = per_item_sums.get(it, 0.0) + v
            per_item_sq[it]   = per_item_sq.get(it, 0.0)   + v*v
            per_item_cnt[it]  = per_item_cnt.get(it, 0)    + 1
            global_sum += v; global_sq += v*v; global_cnt += 1

    def mean_std(sum_, sq_, cnt_):
        if cnt_ <= 1:
            m = (sum_/cnt_) if cnt_ > 0 else 0.0
            return m, 1.0
        m = sum_ / cnt_
        var = max(sq_/cnt_ - m*m, 0.0)
        s = math.sqrt(var) if var > 1e-12 else 1.0
        return m, s

    per_item_stats = {it: mean_std(per_item_sums[it], per_item_sq[it], per_item_cnt[it])
                      for it in per_item_cnt.keys()}

    g_mean, g_std = mean_std(global_sum, global_sq, global_cnt)

    def arr_mean_std(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size <= 1:
            return (float(arr.mean()) if arr.size > 0 else 0.0, 1.0)
        m = float(arr.mean()); s = float(arr.std(ddof=0))
        return (m, 1.0) if s < 1e-12 else (m, s)

    age_stats    = arr_mean_std(ages)
    height_stats = arr_mean_std(heights)

    return {
        "value": {
            "per_item": per_item_stats,
            "global": (g_mean, g_std),
            "categorical": list(cat_set),
        },
        "static": {
            "age": age_stats,
            "height": height_stats,
        }
    }


def normalize_single_samples(
    samples: List[Sample],
    stats: Dict,
    *,
    clamp_std_min: float = 1e-12
) -> List[Sample]:
    per_item = stats["value"]["per_item"]
    g_mean, g_std = stats["value"]["global"]
    cat_set = set(stats["value"].get("categorical", []))

    if g_std < clamp_std_min: g_std = 1.0
    age_m, age_s       = stats["static"]["age"]
    height_m, height_s = stats["static"]["height"]
    if age_s   < clamp_std_min:   age_s   = 1.0
    if height_s < clamp_std_min:  height_s = 1.0

    out: List[Sample] = []
    for pid, obs, st in samples:
        st2 = st.astype(np.float32, copy=True)
        st2[0] = (st2[0] - age_m)    / age_s
        st2[2] = (st2[2] - height_m) / height_s

        if obs.size > 0:
            obs2 = obs.astype(np.float32, copy=True)
            for i in range(obs2.shape[0]):
                it = float(obs2[i, 0])
                if it in cat_set:
                    continue
                m, s = per_item.get(it, (g_mean, g_std))
                if s < clamp_std_min: s = 1.0
                obs2[i, 2] = (obs2[i, 2] - m) / s
        else:
            obs2 = obs

        out.append((pid, obs2, st2))
    return out


def normalize_many(
    reference_samples: List[Sample],
    * ,
    categorical_itemids: Iterable[float | int] = (),
    **datasets: Dict[str, List[Sample]]
) -> Tuple[Dict[str, List[Sample]], Dict]:
    stats = compute_norm_stats(reference_samples, categorical_itemids=categorical_itemids)
    normalized = {name: normalize_single_samples(samples, stats)
                  for name, samples in datasets.items()}
    return normalized, stats


# =========================
# Feather split loading (NEW)
# =========================

def _feather_paths_for(domain: str, split_name: str, split_id: int, split_dir: str) -> Tuple[str, str, str]:
    """
    Your naming rule:
      {domain}_data_{train}_{split}.feather
      {domain}_data_static_{train}_{split}.feather
      {domain}_outcomes_{train}_{split}.feather
    (here `split_name` is one of: train/valid/test)
    """
    df_path = os.path.join(split_dir, f"{domain}_data_{split_name}_{split_id}.feather")
    st_path = os.path.join(split_dir, f"{domain}_data_static_{split_name}_{split_id}.feather")
    oc_path = os.path.join(split_dir, f"{domain}_outcomes_{split_name}_{split_id}.feather")
    return df_path, st_path, oc_path


def load_split_feather(
    data_root: str,
    *,
    split_id: int,
    domain: str,
    split_name: str,  # train/valid/test
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_dir = os.path.join(data_root, f"split{int(split_id)}")
    df_path, st_path, oc_path = _feather_paths_for(domain, split_name, int(split_id), split_dir)

    for p in (df_path, st_path, oc_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing feather file: {p}")

    df = pd.read_feather(df_path)
    st = pd.read_feather(st_path)
    oc = pd.read_feather(oc_path)
    return df, st, oc


def infer_num_features(*dfs: pd.DataFrame) -> int:
    mx = None
    for df in dfs:
        if "itemid" not in df.columns:
            raise KeyError("Expected column 'itemid' in data feather.")
        v = int(df["itemid"].max())
        mx = v if mx is None else max(mx, v)
    return int(mx) + 1


# =========================
# Dataset / loader (your existing code)
# =========================

class TimeSeriesDataset(Dataset):
    """
    Each item returns a dict with:
      pid       : int
      times     : FloatTensor [Ni]       (offset)
      varis     : LongTensor  [Ni]       (itemid)
      values    : FloatTensor [Ni]       (value)
      static    : FloatTensor [3]        (age, gender, height)
      y         : FloatTensor [K]        (outcomes ordered by outcome_cols)
      length    : int (Ni)
    """
    def __init__(
        self,
        samples: List[Sample],
        df_outcome,
        outcome_cols: Sequence[str],
        dtype_obs: torch.dtype = torch.float32,
        dtype_static: torch.dtype = torch.float32,
        dtype_y: torch.dtype = torch.float32,
    ):
        self.samples = samples
        self.outcome_cols = list(outcome_cols)

        out = df_outcome.set_index('pid')[self.outcome_cols].astype(float)
        self.pid_to_y = {int(pid): torch.tensor(out.loc[pid].values, dtype=dtype_y)
                         for pid in out.index}

        self.lengths = [int(s[1].shape[0]) for s in samples]

        self.dtype_obs = dtype_obs
        self.dtype_static = dtype_static

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, obs, static_vec = self.samples[idx]
        itemid = torch.from_numpy(obs[:, 0]).to(self.dtype_obs)
        offset = torch.from_numpy(obs[:, 1]).to(self.dtype_obs) / 2880
        value  = torch.from_numpy(obs[:, 2]).to(self.dtype_obs)

        static = torch.from_numpy(static_vec).to(self.dtype_static)

        y = self.pid_to_y.get(int(pid), None)
        if y is None:
            y = torch.zeros(len(self.outcome_cols), dtype=torch.float32)

        return {
            "pid": int(pid),
            "times": offset,
            "varis": itemid,
            "values": value,
            "static": static,
            "y": y,
            "length": len(offset),
        }


def collate_pad(
    batch: List[Dict],
    *,
    make_pre_mask: bool = False,
    pre_mask_p: float = 0.2,
):
    B = len(batch)
    lengths = [b["length"] for b in batch]
    max_len = max(lengths)

    times  = torch.zeros(B, max_len, dtype=torch.float32)
    varis  = torch.zeros(B, max_len, dtype=torch.long)
    values = torch.zeros(B, max_len, dtype=torch.float32)

    pad_mask = torch.ones(B, max_len, dtype=torch.bool)
    pre_mask = torch.zeros(B, max_len, dtype=torch.bool)

    y = torch.stack([b["y"] for b in batch], dim=0)
    pid = torch.tensor([b["pid"] for b in batch], dtype=torch.long)
    static = torch.stack([b["static"] for b in batch], dim=0)

    for i, b in enumerate(batch):
        L = b["length"]
        times[i, :L]  = b["times"]
        varis[i, :L]  = b["varis"].to(torch.long)
        values[i, :L] = b["values"]
        pad_mask[i, :L] = False

        if make_pre_mask and L > 0:
            m = torch.rand(L) < float(pre_mask_p)
            if not m.any():
                m[torch.randint(0, L, (1,))] = True
            pre_mask[i, :L] = m

    return (times, varis, values, pad_mask, pre_mask, y, pid, static)


class BucketBatchSampler(Sampler[List[int]]):
    def __init__(self, lengths: List[int], batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.indices = np.argsort(np.array(lengths)).tolist()
        self.batches = [self.indices[i:i+batch_size] for i in range(0, len(self.indices), batch_size)]
        if self.drop_last and self.batches and len(self.batches[-1]) < batch_size:
            self.batches = self.batches[:-1]

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(len(self.batches)).tolist()
            for bi in perm:
                yield self.batches[bi]
        else:
            for b in self.batches:
                yield b

    def __len__(self):
        return len(self.batches)


def collate_pad_with_args(batch, *, make_pre_mask=False, pre_mask_p=0.15):
    return collate_pad(batch, make_pre_mask=make_pre_mask, pre_mask_p=pre_mask_p)

def _resolve_data_filename(
    domain: str,
    split: str,
    split_id: int,
    *,
    use_aug: bool = False,
    aug_suffix: str = "aug",
) -> str:
    """
    Returns data feather filename only.

    Original:
      {domain}_data_{split}_{split_id}.feather

    Augmented:
      {domain}_data_{aug_suffix}_{split}_{split_id}.feather
      e.g. eicu_data_aug_train_1.feather
    """
    if use_aug:
        return f"{domain}_data_{aug_suffix}_{split}_{split_id}.feather"
    return f"{domain}_data_{split}_{split_id}.feather"

def make_dataloader(
    dataset: TimeSeriesDataset,
    batch_size: int,
    *,
    shuffle_batches: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    make_pre_mask: bool = False,
    pre_mask_p: float = 0.15,
    persistent_workers: bool = False,
):
    batch_sampler = BucketBatchSampler(
        dataset.lengths, batch_size,
        shuffle=shuffle_batches, drop_last=drop_last
    )

    collate_fn = partial(
        collate_pad_with_args,
        make_pre_mask=make_pre_mask,
        pre_mask_p=pre_mask_p,
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )


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


# =========================
# Split-CV loader builder (NEW)
# =========================

def _split_dir(data_root: str, split_id: int) -> str:
    return os.path.join(data_root, f"split{int(split_id)}")


def _load_split_from_dir(
    data_root: str,
    split_id: int,
    domain: str,
    split: Literal["train", "valid", "test"],
    *,
    use_aug: bool = False,
    aug_suffix: str = "aug",
):
    """
    Files:

    Original:
      {domain}_data_{split}_{split_id}.feather
      {domain}_data_static_{split}_{split_id}.feather
      {domain}_outcomes_{split}_{split_id}.feather

    Augmented data:
      {domain}_data_{aug_suffix}_{split}_{split_id}.feather
      e.g. eicu_data_aug_train_1.feather

    Static / outcomes are always loaded from the original filenames.
    """
    split_dir = _split_dir(data_root, split_id)

    df_name = _resolve_data_filename(
        domain,
        split,
        split_id,
        use_aug=use_aug,
        aug_suffix=aug_suffix,
    )

    df_path = os.path.join(split_dir, df_name)
    st_path = os.path.join(split_dir, f"{domain}_data_static_{split}_{split_id}.feather")
    oc_path = os.path.join(split_dir, f"{domain}_outcomes_{split}_{split_id}.feather")

    for p in (df_path, st_path, oc_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing feather file: {p}")

    df = pd.read_feather(df_path)
    st = pd.read_feather(st_path)
    oc = pd.read_feather(oc_path)
    return df, st, oc


def build_feather_split_loaders(
    cfg,
    split_id: int,
):
    """
    Per split:
      - source train/valid/test: split_id
      - target test: ALWAYS split 1
      - normalization stats: source train of split_id

    Augmentation options:
      - cfg.use_source_aug: bool
      - cfg.source_aug_suffix: str
      - cfg.use_target_aug: bool
      - cfg.target_aug_suffix: str

    Returns:
      num_features,
      pretrain_train_loader, pretrain_val_loader,
      train_loader, val_loader,
      source_test_loader,
      target_test_loader
    """
    split_id = int(split_id)

    use_source_aug = getattr(cfg, "use_source_aug", False)
    source_aug_suffix = getattr(cfg, "source_aug_suffix", "aug")

    use_target_aug = getattr(cfg, "use_target_aug", False)
    target_aug_suffix = getattr(cfg, "target_aug_suffix", "aug")

    # ---------- SOURCE (split k) ----------
    tr_df, tr_st, tr_oc = _load_split_from_dir(
        cfg.data_root,
        split_id,
        cfg.source_domain,
        "train",
        use_aug=use_source_aug,
        aug_suffix=source_aug_suffix,
    )
    va_df, va_st, va_oc = _load_split_from_dir(
        cfg.data_root,
        split_id,
        cfg.source_domain,
        "valid",
        use_aug=use_source_aug,
        aug_suffix=source_aug_suffix,
    )
    te_df, te_st, te_oc = _load_split_from_dir(
        cfg.data_root,
        split_id,
        cfg.source_domain,
        "test",
        use_aug=use_source_aug,
        aug_suffix=source_aug_suffix,
    )

    # ---------- TARGET TEST (split 1, always) ----------
    tgt_df, tgt_st, tgt_oc = _load_split_from_dir(
        cfg.data_root,
        1,
        cfg.target_domain,
        "test",
        use_aug=use_target_aug,
        aug_suffix=target_aug_suffix,
    )

    # ---------- num_features ----------
    num_features = int(
        max(
            tr_df["itemid"].max(),
            va_df["itemid"].max(),
            te_df["itemid"].max(),
            tgt_df["itemid"].max(),
        )
    ) + 1

    # ---------- build raw samples ----------
    tr_raw = prepare_dataset_samples(tr_df, tr_st)
    va_raw = prepare_dataset_samples(va_df, va_st)
    te_raw = prepare_dataset_samples(te_df, te_st)

    # ---------- normalize using SOURCE TRAIN ----------
    normed, stats = normalize_many(
        tr_raw,
        categorical_itemids=cfg.categorical_itemids,
        train=tr_raw,
        valid=va_raw,
        test=te_raw,
    )

    tr_s = normed["train"]
    va_s = normed["valid"]
    te_s = normed["test"]

    # ---------- downstream loaders (source) ----------
    (ds_tr, ds_va, ds_te), (ld_tr, ld_va, ld_te) = _make_loaders(
        tr_s, va_s, te_s,
        tr_oc, va_oc, te_oc,
        outcome_cols=cfg.outcome_cols,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        make_pre_mask_train=False,
        pre_mask_p_train=cfg.pre_mask_p_train,
    )

    # ---------- pretrain loaders ----------
    pre_ld_tr = make_dataloader(
        ds_tr,
        cfg.batch_size,
        num_workers=cfg.num_workers,
        make_pre_mask=True,
        pre_mask_p=cfg.pre_mask_p_train,
    )
    pre_ld_va = make_dataloader(
        ds_va,
        cfg.batch_size,
        num_workers=cfg.num_workers,
        make_pre_mask=True,
        pre_mask_p=cfg.pre_mask_p_train,
    )

    # ---------- TARGET TEST ----------
    tgt_raw = prepare_dataset_samples(tgt_df, tgt_st)
    tgt_norm = normalize_single_samples(tgt_raw, stats)

    tgt_ds = TimeSeriesDataset(
        tgt_norm,
        tgt_oc,
        outcome_cols=cfg.outcome_cols,
    )
    tgt_loader = make_dataloader(
        tgt_ds,
        cfg.batch_size,
        num_workers=cfg.num_workers,
        make_pre_mask=False,
        pre_mask_p=0.0,
    )

    return (
        num_features,
        pre_ld_tr, pre_ld_va,
        ld_tr, ld_va,
        ld_te,
        tgt_loader,
    )

def build_loaders_for_vis(
    cfg,
    split_id: int,
    *,
    return_raw: bool = True,
):
    """
    Visualization-friendly loader builder.

    Per split:
      - source train/valid/test: split_id
      - target test: ALWAYS split 1
      - normalization stats: source train of split_id

    Same behavior as build_feather_split_loaders, but additionally returns:
      - stats used for normalization
      - optionally raw samples before normalization
      - normalized sample lists
      - raw/normalized target samples

    Returns:
      {
        "num_features": int,
        "pretrain_train_loader": ...,
        "pretrain_val_loader": ...,
        "train_loader": ...,
        "val_loader": ...,
        "source_test_loader": ...,
        "target_test_loader": ...,
        "stats": stats,
        "source_samples_raw": {"train": ..., "valid": ..., "test": ...},   # optional
        "source_samples_norm": {"train": ..., "valid": ..., "test": ...},
        "target_samples_raw": ...,   # optional
        "target_samples_norm": ...,
        "source_outcomes": {"train": ..., "valid": ..., "test": ...},
        "target_outcomes": ...,
        "dataframes": {
            "source_train_df": ...,
            "source_valid_df": ...,
            "source_test_df": ...,
            "target_test_df": ...,
            "source_train_static": ...,
            "source_valid_static": ...,
            "source_test_static": ...,
            "target_test_static": ...,
        }
      }
    """
    split_id = int(split_id)

    use_source_aug = getattr(cfg, "use_source_aug", False)
    source_aug_suffix = getattr(cfg, "source_aug_suffix", "aug")

    use_target_aug = getattr(cfg, "use_target_aug", False)
    target_aug_suffix = getattr(cfg, "target_aug_suffix", "aug")

    # ---------- SOURCE (split k) ----------
    tr_df, tr_st, tr_oc = _load_split_from_dir(
        cfg.data_root,
        split_id,
        cfg.source_domain,
        "train",
        use_aug=use_source_aug,
        aug_suffix=source_aug_suffix,
    )
    va_df, va_st, va_oc = _load_split_from_dir(
        cfg.data_root,
        split_id,
        cfg.source_domain,
        "valid",
        use_aug=use_source_aug,
        aug_suffix=source_aug_suffix,
    )
    te_df, te_st, te_oc = _load_split_from_dir(
        cfg.data_root,
        split_id,
        cfg.source_domain,
        "test",
        use_aug=use_source_aug,
        aug_suffix=source_aug_suffix,
    )

    # ---------- TARGET TEST (split 1, always) ----------
    tgt_df, tgt_st, tgt_oc = _load_split_from_dir(
        cfg.data_root,
        1,
        cfg.target_domain,
        "test",
        use_aug=use_target_aug,
        aug_suffix=target_aug_suffix,
    )

    # ---------- num_features ----------
    num_features = int(
        max(
            tr_df["itemid"].max(),
            va_df["itemid"].max(),
            te_df["itemid"].max(),
            tgt_df["itemid"].max(),
        )
    ) + 1

    # ---------- build raw samples ----------
    tr_raw = prepare_dataset_samples(tr_df, tr_st)
    va_raw = prepare_dataset_samples(va_df, va_st)
    te_raw = prepare_dataset_samples(te_df, te_st)
    tgt_raw = prepare_dataset_samples(tgt_df, tgt_st)

    # ---------- normalize using SOURCE TRAIN ----------
    normed, stats = normalize_many(
        tr_raw,
        categorical_itemids=cfg.categorical_itemids,
        train=tr_raw,
        valid=va_raw,
        test=te_raw,
    )

    tr_s = normed["train"]
    va_s = normed["valid"]
    te_s = normed["test"]

    # target is normalized with source-train stats
    tgt_norm = normalize_single_samples(tgt_raw, stats)

    # ---------- downstream loaders (source) ----------
    (ds_tr, ds_va, ds_te), (ld_tr, ld_va, ld_te) = _make_loaders(
        tr_s, va_s, te_s,
        tr_oc, va_oc, te_oc,
        outcome_cols=cfg.outcome_cols,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        make_pre_mask_train=False,
        pre_mask_p_train=cfg.pre_mask_p_train,
    )

    # ---------- pretrain loaders ----------
    pre_ld_tr = make_dataloader(
        ds_tr,
        cfg.batch_size,
        num_workers=cfg.num_workers,
        make_pre_mask=True,
        pre_mask_p=cfg.pre_mask_p_train,
    )
    pre_ld_va = make_dataloader(
        ds_va,
        cfg.batch_size,
        num_workers=cfg.num_workers,
        make_pre_mask=True,
        pre_mask_p=cfg.pre_mask_p_train,
    )

    # ---------- TARGET TEST loader ----------
    tgt_ds = TimeSeriesDataset(
        tgt_norm,
        tgt_oc,
        outcome_cols=cfg.outcome_cols,
    )
    tgt_loader = make_dataloader(
        tgt_ds,
        cfg.batch_size,
        num_workers=cfg.num_workers,
        make_pre_mask=False,
        pre_mask_p=0.0,
    )

    out = {
        "num_features": num_features,

        "pretrain_train_loader": pre_ld_tr,
        "pretrain_val_loader": pre_ld_va,
        "train_loader": ld_tr,
        "val_loader": ld_va,
        "source_test_loader": ld_te,
        "target_test_loader": tgt_loader,

        "stats": stats,

        "source_samples_norm": {
            "train": tr_s,
            "valid": va_s,
            "test": te_s,
        },
        "target_samples_norm": tgt_norm,

        "source_outcomes": {
            "train": tr_oc,
            "valid": va_oc,
            "test": te_oc,
        },
        "target_outcomes": tgt_oc,

        "dataframes": {
            "source_train_df": tr_df,
            "source_valid_df": va_df,
            "source_test_df": te_df,
            "target_test_df": tgt_df,
            "source_train_static": tr_st,
            "source_valid_static": va_st,
            "source_test_static": te_st,
            "target_test_static": tgt_st,
        },
    }

    if return_raw:
        out["source_samples_raw"] = {
            "train": tr_raw,
            "valid": va_raw,
            "test": te_raw,
        }
        out["target_samples_raw"] = tgt_raw

    return out