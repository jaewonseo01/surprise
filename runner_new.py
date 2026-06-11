from typing import Dict, List, Tuple, Iterable, Optional, Literal, Sequence, Any
from functools import partial
import json
import pandas as pd
import numpy as np
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl

def prepare_dataset_samples(
    data: pd.DataFrame,
    static: pd.DataFrame,
    *,
    sort_obs_by: Iterable[str] = ("offset", "itemid"),
    dropna_in_obs_value: bool = True,
    cast_itemid_to_int: bool = True,
    dtypes: Tuple[str, str] = ("float32", "float32"),  # (obs dtype, static dtype)
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Prepare variable-length observation matrices per pid for a PyTorch Dataset.

    Returns
    -------
    samples : list of (pid, obs_matrix, static_vec)
        - pid: int
        - obs_matrix: np.ndarray shape (Ni, 3) with rows [itemid, offset, value]
        - static_vec: np.ndarray shape (3,) = [age, gender, height]
      Ni differs per pid. dtypes = (obs_dtype, static_dtype)

    Inputs
    ------
    data : DataFrame with columns ['pid','itemid','offset','value']
    static : DataFrame with columns ['pid','age','gender','height', ...]
             (gender is already binary; we keep only age, gender, height)
    """
    obs_dtype, static_dtype = dtypes

    # --- sanity checks
    req_data = {"pid", "itemid", "offset", "value"}
    req_static = {"pid", "age", "gender", "height"}
    md = req_data - set(data.columns)
    ms = req_static - set(static.columns)
    if md:
        raise ValueError(f"`data` missing columns: {sorted(md)}")
    if ms:
        raise ValueError(f"`static` missing columns: {sorted(ms)}")

    # --- static: keep only needed, ensure numeric
    static_slim = static.loc[:, ["pid", "age", "gender", "height"]].copy()
    for col in ["age", "gender", "height"]:
        static_slim[col] = pd.to_numeric(static_slim[col], errors="coerce")
    # gender is already binary; optional clamp for robustness
    static_slim["gender"] = static_slim["gender"].clip(0, 1)

    # drop bad static rows
    static_slim = static_slim.dropna(subset=["age", "gender", "height"])
    static_slim = static_slim.drop_duplicates(subset=["pid"], keep="first")

    # build lookup: pid -> [age, gender, height]
    static_vec_lookup = {
        int(pid): np.array([row.age, row.gender, row.height], dtype=static_dtype)
        for pid, row in static_slim.set_index("pid").iterrows()
    }

    # --- data: clean & sort
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
    data_clean = data_clean.sort_values(sort_cols, kind="mergesort")  # stable

    # --- group and assemble outputs
    samples: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for pid, df_pid in data_clean.groupby("pid", sort=False):
        # if static is unexpectedly missing for this pid, skip quietly (preprocessing should’ve ensured presence)
        static_vec = static_vec_lookup.get(pid, None)
        if static_vec is None:
            continue

        obs_mat = df_pid.loc[:, ["itemid", "offset", "value"]].to_numpy(dtype=obs_dtype)
        if obs_mat.shape[0] == 0:
            continue

        samples.append((int(pid), obs_mat, static_vec))

    return samples


Sample = Tuple[int, np.ndarray, np.ndarray]  # (pid, obs[N,3], static[3])

def compute_norm_stats(
    reference_samples: List[Sample],
    categorical_itemids: Iterable[float | int] = ()
) -> Dict:
    """
    Build normalization stats from reference samples (e.g., train).
    - value: per-item mean/std for NON-categorical items only
    - static: age, height mean/std (gender untouched)
    - categorical_itemids are excluded entirely from value stats.

    categorical_itemids: iterable of itemids (int/float). These will be excluded
      from stats and later left unnormalized.
    """
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
                # skip categorical values from stats
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

    # global fallback (for unseen non-categorical items)
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
            "categorical": list(cat_set),  # keep for downstream
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
    """
    Apply normalization:
      - value: z-score per itemid (skip if categorical)
      - static: z on age, height; gender unchanged
    """
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
        # st = [age, gender, height]
        st2[0] = (st2[0] - age_m)    / age_s      # age
        # st2[1] = gender (keep as is)
        st2[2] = (st2[2] - height_m) / height_s   # height

        if obs.size > 0:
            obs2 = obs.astype(np.float32, copy=True)
            for i in range(obs2.shape[0]):
                it = float(obs2[i, 0])
                if it in cat_set:
                    # leave categorical as-is
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
    """
    Normalize multiple datasets using stats from `reference_samples`.
    Pass `categorical_itemids` to exclude those itemids from normalization.

    Usage:
      normalized, stats = normalize_many(
          train_samples, categorical_itemids={19,20,21},  # example
          train=train_samples, valid=valid_samples, test=test_samples
      )
    """
    stats = compute_norm_stats(reference_samples, categorical_itemids=categorical_itemids)
    normalized = {name: normalize_single_samples(samples, stats)
                  for name, samples in datasets.items()}
    return normalized, stats

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
        samples: List[Tuple[int, np.ndarray, np.ndarray]],
        df_outcome,
        outcome_cols: Sequence[str],
        dtype_obs: torch.dtype = torch.float32,
        dtype_static: torch.dtype = torch.float32,
        dtype_y: torch.dtype = torch.float32,
    ):
        self.samples = samples
        self.outcome_cols = list(outcome_cols)

        # pid -> y vector
        # expect df_outcome has exactly one row per pid
        out = df_outcome.set_index('pid')[self.outcome_cols].astype(float)
        self.pid_to_y = {int(pid): torch.tensor(out.loc[pid].values, dtype=dtype_y) 
                         for pid in out.index}

        # precompute lengths
        self.lengths = [int(s[1].shape[0]) for s in samples]

        self.dtype_obs = dtype_obs
        self.dtype_static = dtype_static

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, obs, static_vec = self.samples[idx]
        # obs: [itemid, offset, value]
        itemid = torch.from_numpy(obs[:, 0]).to(self.dtype_obs)
        offset = torch.from_numpy(obs[:, 1]).to(self.dtype_obs)/2880
        value  = torch.from_numpy(obs[:, 2]).to(self.dtype_obs)

        # if you plan to use an embedding for itemid, cast to long at collate
        static = torch.from_numpy(static_vec).to(self.dtype_static)

        y = self.pid_to_y.get(int(pid), None)
        if y is None:
            # if outcomes missing, create zeros (or raise)
            y = torch.zeros(len(self.outcome_cols), dtype=torch.float32)

        return {
            "pid": int(pid),
            "times": offset,
            "varis": itemid,   # will cast to long in collate
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
    pre_mask_on: str = "values"  # or "all" to also mask varis/times if you expand later
):
    """
    Pad to max length in batch. Returns tuple in the exact order:
      [times, varis, values, pad_mask, pre_mask, y, pid, static]

    pad_mask: BoolTensor [B, T] — True for PAD positions (so it's directly usable as key_padding_mask)
    pre_mask: BoolTensor [B, T] — True where pretraining mask applies (casing only here; values left untouched)
    """
    B = len(batch)
    lengths = [b["length"] for b in batch]
    max_len = max(lengths)

    times  = torch.zeros(B, max_len, dtype=torch.float32)
    varis  = torch.zeros(B, max_len, dtype=torch.long)     # embed-friendly
    values = torch.zeros(B, max_len, dtype=torch.float32)

    pad_mask = torch.ones(B, max_len, dtype=torch.bool)    # start as all PAD=True
    pre_mask = torch.zeros(B, max_len, dtype=torch.bool)

    y = torch.stack([b["y"] for b in batch], dim=0)        # [B, K]
    pid = torch.tensor([b["pid"] for b in batch], dtype=torch.long)
    static = torch.stack([b["static"] for b in batch], dim=0)  # [B, 3]

    for i, b in enumerate(batch):
        L = b["length"]
        # times/varis/values
        times[i, :L]  = b["times"]
        # cast varis to long for embeddings
        varis[i, :L]  = b["varis"].to(torch.long)
        values[i, :L] = b["values"]

        # flip PAD to False for valid tokens
        pad_mask[i, :L] = False

        # pretrain mask "casing" (just provide the field; no corruption here)
        if make_pre_mask and L > 0:
            # simple Bernoulli over valid tokens
            m = torch.rand(L) < float(pre_mask_p)
            # at least one token masked to avoid all-zero rows (optional)
            if not m.any():
                m[torch.randint(0, L, (1,))] = True
            pre_mask[i, :L] = m

    # final tuple order as requested
    return (times, varis, values, pad_mask, pre_mask, y, pid, static)


class BucketBatchSampler(Sampler[List[int]]):
    """
    Group examples with similar lengths to reduce padding/memory fragmentation.
    Strategy:
      - sort indices by length
      - chunk into contiguous batches of batch_size
      - shuffle batches each epoch (but keep intra-batch order for efficiency)
    """
    def __init__(self, lengths: List[int], batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last

        # sort indices by length ascending
        self.indices = np.argsort(np.array(lengths)).tolist()

        # precompute batches (contiguous slices)
        self.batches = [self.indices[i:i+batch_size] for i in range(0, len(self.indices), batch_size)]
        if self.drop_last and len(self.batches[-1]) < batch_size:
            self.batches = self.batches[:-1]

    def __iter__(self):
        if self.shuffle:
            # shuffle the order of batches
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
    persistent_workers: bool = False,   # 선택: 1epoch 뒤 워커 유지
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
        collate_fn=collate_fn,          # ← top-level 함수 + partial (picklable)
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )


# -------------------------
# 0) reproducibility
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)

# -------------------------
# 1) Load raw dataframes
# -------------------------
def load_dataframes(data_source:str, data_dir: str):
    """
    Expected files (adapt as needed):
      - data.feather           (concatenated time series: pid,itemid,offset,value)
      - data_static.feather    (pid, age, gender, height, ...)
      - outcomes.feather       (pid, <outcomes...>)
      - pid_splits.json        ({train:[], valid:[], test:[]})
    """
    data       = pd.read_feather(f"{data_dir}/{data_source}_data.feather")
    static     = pd.read_feather(f"{data_dir}/{data_source}_data_static.feather")
    df_outcome = pd.read_feather(f"{data_dir}/{data_source}_outcomes.feather")
    splits_fp  = f"{data_dir}/pid_splits.json"
    with open(splits_fp, "r", encoding="utf-8") as f:
        splits = json.load(f)
    return data, static, df_outcome, splits

# -------------------------
# 2) Build per-split samples
# -------------------------
def build_samples_by_split(
    data: pd.DataFrame,
    static: pd.DataFrame,
    df_outcome: pd.DataFrame,
    splits: Dict[str, List[int]],
    *,
    sort_obs_by: Iterable[str] = ("offset", "itemid"),
    dtypes: Tuple[str, str] = ("float32", "float32"),
):
    # filter per split
    pid_train = set(splits["train"])
    pid_valid = set(splits["valid"])
    pid_test  = set(splits["test"])

    data_tr  = data[data["pid"].isin(pid_train)]
    data_va  = data[data["pid"].isin(pid_valid)]
    data_te  = data[data["pid"].isin(pid_test)]

    stat_tr  = static[static["pid"].isin(pid_train)]
    stat_va  = static[static["pid"].isin(pid_valid)]
    stat_te  = static[static["pid"].isin(pid_test)]

    df_tr    = df_outcome[df_outcome["pid"].isin(pid_train)].copy()
    df_va    = df_outcome[df_outcome["pid"].isin(pid_valid)].copy()
    df_te    = df_outcome[df_outcome["pid"].isin(pid_test)].copy()

    # prepare variable-length matrices
    train_samples = prepare_dataset_samples(
        data_tr, stat_tr, sort_obs_by=sort_obs_by, dtypes=dtypes
    )
    valid_samples = prepare_dataset_samples(
        data_va, stat_va, sort_obs_by=sort_obs_by, dtypes=dtypes
    )
    test_samples  = prepare_dataset_samples(
        data_te, stat_te, sort_obs_by=sort_obs_by, dtypes=dtypes
    )
    return (train_samples, valid_samples, test_samples), (df_tr, df_va, df_te)

# -------------------------
# 3) Normalize using train stats
# -------------------------
def normalize_splits(
    train_samples, valid_samples, test_samples, *,
    categorical_itemids: Iterable[int | float] = ()
):
    normalized, stats = normalize_many(
        train_samples,
        categorical_itemids=categorical_itemids,
        train=train_samples, valid=valid_samples, test=test_samples
    )
    return normalized["train"], normalized["valid"], normalized["test"], stats

# ETT

Sample = Tuple[int, np.ndarray, np.ndarray]  # (pid, obs[N,3], static[3])

class TimeSeriesDatasetETT(Dataset):
    """
    ETT용: offset 이미 [0,1]이므로 추가 scaling 없음.
    y는 [K] (K = pred_len) regression vector로 들어감.
    """
    def __init__(self, samples, df_outcome: pd.DataFrame, outcome_cols: Sequence[str]):
        self.samples = samples
        self.outcome_cols = list(outcome_cols)
        out = df_outcome.set_index("pid")[self.outcome_cols].astype(float)
        self.pid_to_y = {int(pid): torch.tensor(out.loc[pid].values, dtype=torch.float32)
                         for pid in out.index}
        self.lengths = [int(s[1].shape[0]) for s in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, obs, static_vec = self.samples[idx]
        itemid = torch.from_numpy(obs[:, 0]).float()      # collate에서 long으로 바꿀거면 거기서 처리
        offset = torch.from_numpy(obs[:, 1]).float()      # 이미 0~1
        value  = torch.from_numpy(obs[:, 2]).float()
        static = torch.from_numpy(static_vec).float()

        y = self.pid_to_y.get(int(pid))
        if y is None:
            y = torch.zeros(len(self.outcome_cols), dtype=torch.float32)

        return {
            "pid": int(pid),
            "times": offset,
            "varis": itemid,
            "values": value,
            "static": static,
            "y": y,
            "length": int(offset.numel()),
        }
    
def make_dataloader_random(
    dataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    make_pre_mask: bool = False,
    pre_mask_p: float = 0.15,
    persistent_workers: bool = False,
):
    collate_fn = partial(
        collate_pad_with_args,
        make_pre_mask=make_pre_mask,
        pre_mask_p=pre_mask_p,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )

def build_ett_windows_as_samples(
    csv_path: str,
    split: Literal["train","val","test"],
    *,
    lookback: int = 96,
    pred_len: int = 96,
    features: Literal["S","M"] = "M",
    target_col: str = "OT",
    scale: bool = True,
    official_split: bool = True,
    scaler_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    return_scaler_stats: bool = False,
):
    df_raw = pd.read_csv(csv_path)

    if "date" in df_raw.columns:
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")

    # ---- 입력 / 출력 컬럼 정의 ----
    if features == "M":
        input_cols  = ["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]
        target_cols = input_cols[:]   # 7개 전부 예측
    else:  # "S"
        input_cols  = [target_col]
        target_cols = [target_col]

    missing = [c for c in input_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"[{csv_path}] missing columns: {missing}")

    X_all = df_raw[input_cols].to_numpy(np.float32)   # [N, F_in]
    N, F_in = X_all.shape
    F_out = len(target_cols)

    # target_cols가 input_cols 안에서 몇 번째인지 인덱스 구해두기
    target_idx = [input_cols.index(c) for c in target_cols]

    # ---- split 경계 ----
    if official_split:
        train_end = int(round(N * 12 / 20))
        val_end   = int(round(N * 16 / 20))
        test_end  = N
        train_end = max(train_end, lookback + pred_len + 1)
        val_end   = max(val_end, train_end + 1)
        val_end   = min(val_end, N - 1)
    else:
        train_end = int(round(N * 0.7))
        val_end   = int(round(N * 0.8))
        test_end  = N
        train_end = max(train_end, lookback + pred_len + 1)
        val_end   = max(val_end, train_end + 1)
        val_end   = min(val_end, N - 1)

    if split == "train":
        border1, border2 = 0, train_end
    elif split == "val":
        border1, border2 = max(0, train_end - lookback), val_end
    elif split == "test":
        border1, border2 = max(0, val_end - lookback), test_end
    else:
        raise ValueError(split)

    # ---- scaling (입력/타깃 같이) ----
    fitted_stats = None
    if scale:
        if scaler_stats is None:
            X_train = X_all[:train_end]                    # [T_train, F_in]
            mean = X_train.mean(axis=0).astype(np.float32) # [F_in]
            std  = X_train.std(axis=0).astype(np.float32)
            std  = np.where(std < 1e-12, 1.0, std).astype(np.float32)
            fitted_stats = (mean, std)
        else:
            mean, std = scaler_stats
            mean = np.asarray(mean, np.float32)
            std  = np.asarray(std, np.float32)
            if mean.shape != (F_in,) or std.shape != (F_in,):
                raise ValueError(f"scaler_stats must have shape (F_in,), got mean{mean.shape}, std{std.shape}")

        if scaler_stats is None:
            mean, std = fitted_stats

        X_all = (X_all - mean) / std

    # ---- split slice ----
    X = X_all[border1:border2]   # [T_split, F_in]
    Tsplit = X.shape[0]

    n_win = Tsplit - lookback - pred_len + 1
    if n_win <= 0:
        raise ValueError(f"Split {split} too short: T={Tsplit}, lookback={lookback}, pred_len={pred_len}")

    # offset: 0~1 사이 균등
    offsets = (np.arange(lookback, dtype=np.float32) / float(lookback - 1)) if lookback > 1 else np.zeros((lookback,), np.float32)
    itemids = np.repeat(np.arange(F_in, dtype=np.float32), lookback)   # 0..F_in-1
    offs    = np.tile(offsets, F_in)

    samples, out_rows = [], []
    pid_base = {"train": 0, "val": 10_000_000, "test": 20_000_000}[split]

    for i in range(n_win):
        pid = pid_base + i
        x_block = X[i:i+lookback]                   # [lookback, F_in]
        # 미래 구간 [lookback : lookback+pred_len]에서 타깃들만 추출
        y_block = X[i+lookback:i+lookback+pred_len, :][:, target_idx]   # [pred_len, F_out]

        vals = x_block.T.reshape(-1).astype(np.float32)
        obs  = np.stack([itemids, offs, vals], axis=1).astype(np.float32)
        static = np.zeros((3,), dtype=np.float32)

        samples.append((int(pid), obs, static))

        row = {"pid": int(pid)}
        for f, col in enumerate(target_cols):
            for t in range(pred_len):
                row[f"{col}_t{t}"] = float(y_block[t, f])
        out_rows.append(row)

    df_out = pd.DataFrame(out_rows)
    outcome_cols = [f"{col}_t{t}" for f, col in enumerate(target_cols) for t in range(pred_len)]

    if return_scaler_stats:
        if not scale:
            return samples, df_out, outcome_cols, F_in, None
        return samples, df_out, outcome_cols, F_in, (scaler_stats if scaler_stats is not None else fitted_stats)

    return samples, df_out, outcome_cols, F_in

