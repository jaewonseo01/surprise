import os
import numpy as np
import pandas as pd
import pickle
import random
import torch.backends.cudnn as cudnn

import torch
import torch.nn as nn
from datetime import datetime
from transformers import set_seed

from pytz import timezone
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import roc_auc_score, average_precision_score

import torch.nn.functional as F

# ---------------------------
# Helpers
# ---------------------------

def _path_data(args, split: str) -> str:
    return os.path.join(args.data_dir, f"{args.source}_data_{split}.feather")

def _path_static(args, split: str) -> str:
    return os.path.join(args.data_dir, f"{args.source}_data_static_{split}.feather")

def _path_outcomes(args, split: str) -> str:
    return os.path.join(args.data_dir, f"{args.source}_outcomes_{split}.feather")

def _encode_gender(x):
    # robust encoding: supports {M/F}, {male/female}, {0/1}, already numeric
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().lower()
    if s in ["m", "male", "man", "1"]:
        return 1.0
    if s in ["f", "female", "woman", "0"]:
        return 0.0
    # unknown category -> NaN, will be mean-filled
    return np.nan

def _safe_std(x: np.ndarray) -> np.ndarray:
    s = x.std(axis=0, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return s

def set_all_seeds(seed: int) -> None:
    """Function to set seeds for all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    set_seed(seed)
# ---------------------------
# CycleIndex (keep your existing one)
# ---------------------------

class CycleIndex:
    def __init__(self, indices: Union[int, list], batch_size: int, shuffle: bool = True) -> None:
        if type(indices) == int:
            indices = np.arange(indices)
        self.indices = np.array(indices)
        self.num_samples = len(self.indices)
        self.batch_size = batch_size
        self.pointer = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def get_batch_ind(self):
        start, end = self.pointer, self.pointer + self.batch_size
        if end <= self.num_samples:
            if end == self.num_samples:
                self.pointer = 0
                if self.shuffle:
                    np.random.shuffle(self.indices)
            else:
                self.pointer = end
            return self.indices[start:end]

        last = self.indices[start:]
        remaining = self.batch_size - (self.num_samples - start)
        self.pointer = remaining
        if self.shuffle:
            np.random.shuffle(self.indices)
        return np.concatenate((last, self.indices[:remaining]))


# ---------------------------
# FeatherDataset (SUPERVISED)
# ---------------------------

class Dataset:
    """
    Supervised dataset with domain shift:
      - source domain: train/valid/test
      - target domain: test only -> split name "target"
    Normalization (value mean/std, item vocab, time max_offset) is based ONLY on source train.
    """

    def __init__(self, args) -> None:
        self.args = args
        args.data_dir = getattr(args, "data_dir", "./data")

        if not hasattr(args, "target") or args.target is None:
            raise ValueError("args.target must be set (e.g., 'mimic' or 'eicu')")

        args.logger.write(f"\nPreparing dataset source={args.source}, target={args.target}")

        # ---------------------------
        # Load SOURCE splits
        # ---------------------------
        src_train = pd.read_feather(os.path.join(args.data_dir, f"{args.source}_data_train.feather"))
        src_valid = pd.read_feather(os.path.join(args.data_dir, f"{args.source}_data_valid.feather"))
        src_test  = pd.read_feather(os.path.join(args.data_dir, f"{args.source}_data_test.feather"))

        src_st_train = pd.read_feather(os.path.join(args.data_dir, f"{args.source}_data_static_train.feather"))
        src_st_valid = pd.read_feather(os.path.join(args.data_dir, f"{args.source}_data_static_valid.feather"))
        src_st_test  = pd.read_feather(os.path.join(args.data_dir, f"{args.source}_data_static_test.feather"))

        src_oc_train = pd.read_feather(os.path.join(args.data_dir, f"{args.source}_outcomes_train.feather"))
        src_oc_valid = pd.read_feather(os.path.join(args.data_dir, f"{args.source}_outcomes_valid.feather"))
        src_oc_test  = pd.read_feather(os.path.join(args.data_dir, f"{args.source}_outcomes_test.feather"))

        # ---------------------------
        # Load TARGET test split only
        # ---------------------------
        tgt_test = pd.read_feather(os.path.join(args.data_dir, f"{args.target}_data_test.feather"))
        tgt_st_test = pd.read_feather(os.path.join(args.data_dir, f"{args.target}_data_static_test.feather"))
        tgt_oc_test = pd.read_feather(os.path.join(args.data_dir, f"{args.target}_outcomes_test.feather"))

        # schema checks
        required_ts_cols = {"pid", "offset", "itemid", "value"}
        for name, df in [("src_train", src_train), ("src_valid", src_valid), ("src_test", src_test),
                         ("tgt_test", tgt_test)]:
            if not required_ts_cols.issubset(df.columns):
                raise ValueError(f"{name} missing cols: {required_ts_cols - set(df.columns)}")

        required_static_cols = {"pid", "age", "gender", "height"}
        for name, df in [("src_st_train", src_st_train), ("src_st_valid", src_st_valid), ("src_st_test", src_st_test),
                         ("tgt_st_test", tgt_st_test)]:
            if not required_static_cols.issubset(df.columns):
                raise ValueError(f"{name} missing cols: {required_static_cols - set(df.columns)}")

        for name, df in [("src_oc_train", src_oc_train), ("src_oc_valid", src_oc_valid), ("src_oc_test", src_oc_test),
                         ("tgt_oc_test", tgt_oc_test)]:
            if "pid" not in df.columns:
                raise ValueError(f"{name} outcomes must include pid")

        # ---------------------------
        # Label columns (canonical from SOURCE outcomes)
        # ---------------------------
        drop_labels = {"aki_label", "cf_label"}
        outcome_cols = [c for c in src_oc_train.columns if c != "pid" and c not in drop_labels]
        if len(outcome_cols) == 0:
            raise ValueError("No outcome columns found after dropping aki_label/cf_label")

        # ensure target has same label columns
        missing_in_target = [c for c in outcome_cols if c not in tgt_oc_test.columns]
        if missing_in_target:
            raise ValueError(f"Target outcomes missing label cols: {missing_in_target[:10]} (and more)" 
                             if len(missing_in_target) > 10 else
                             f"Target outcomes missing label cols: {missing_in_target}")

        self.outcome_cols = outcome_cols
        args.outcome_cols = outcome_cols
        args.num_labels = len(outcome_cols)
        args.logger.write(f"Using {len(outcome_cols)} outcome labels (dropped {sorted(list(drop_labels))})")

        # ---------------------------
        # PIDs for splits (source train/val/test + target test)
        # ---------------------------
        src_train_pids = np.intersect1d(src_train["pid"].unique(), src_oc_train["pid"].unique())
        src_valid_pids = np.intersect1d(src_valid["pid"].unique(), src_oc_valid["pid"].unique())
        src_test_pids  = np.intersect1d(src_test["pid"].unique(),  src_oc_test["pid"].unique())

        tgt_test_pids  = np.intersect1d(tgt_test["pid"].unique(),  tgt_oc_test["pid"].unique())

        # optional train_frac slicing (keep behavior)
        if hasattr(args, "train_frac") and args.train_frac < 1.0:
            num_train = int(np.ceil(args.train_frac * len(src_train_pids)))
            src_train_pids = np.array(src_train_pids)[:num_train]
            num_valid = int(np.ceil(args.train_frac * len(src_valid_pids)))
            src_valid_pids = np.array(src_valid_pids)[:num_valid]

        # global pid list with target appended
        self.pids = np.concatenate([src_train_pids, src_valid_pids, src_test_pids, tgt_test_pids])
        pid_to_ind = {pid: i for i, pid in enumerate(self.pids)}

        self.splits = {
            "train":  [pid_to_ind[p] for p in src_train_pids],
            "val":    [pid_to_ind[p] for p in src_valid_pids],
            "test":   [pid_to_ind[p] for p in src_test_pids],
            "target": [pid_to_ind[p] for p in tgt_test_pids],   # ✅ new split
        }
        self.splits["eval_train"] = self.splits["train"][:2000]

        self.N = len(self.pids)
        args.logger.write(f"# source train/val/test PIDs: {[len(src_train_pids), len(src_valid_pids), len(src_test_pids)]}")
        args.logger.write(f"# target test PIDs: {len(tgt_test_pids)}")

        # ---------------------------
        # Build label matrix Y: [N, C] for all splits
        # ---------------------------
        src_oc_all = pd.concat([src_oc_train, src_oc_valid, src_oc_test], axis=0, ignore_index=True)
        tgt_oc_all = tgt_oc_test.copy()

        oc_all = pd.concat([src_oc_all, tgt_oc_all], axis=0, ignore_index=True)
        oc_all = oc_all.loc[oc_all["pid"].isin(self.pids)].copy()
        oc_all["ts_ind"] = oc_all["pid"].map(pid_to_ind)
        oc_all = oc_all.sort_values("ts_ind")

        y = oc_all[self.outcome_cols].to_numpy(dtype=np.float32)
        if y.shape[0] != self.N:
            missing = np.setdiff1d(self.pids, oc_all["pid"].unique())
            raise ValueError(f"Missing outcomes for {len(missing)} pids (example={missing[:10]})")
        self.y = y

        # pos_weight computed ONLY from SOURCE train
        train_y = self.y[self.splits["train"]]
        pos = train_y.sum(axis=0)
        neg = train_y.shape[0] - pos
        pos_weight = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0).astype(np.float32)
        args.pos_weight = pos_weight
        args.logger.write(f"pos_weight (per-label, from source train) min={pos_weight.min():.3f}, max={pos_weight.max():.3f}")

        # ---------------------------
        # Static features: concat source(train/val/test) + target(test)
        # Normalize based ONLY on SOURCE train
        # ---------------------------
        st_all = pd.concat([src_st_train, src_st_valid, src_st_test, tgt_st_test], axis=0, ignore_index=True)
        st_all = st_all.loc[st_all["pid"].isin(self.pids)].copy()
        st_all["ts_ind"] = st_all["pid"].map(pid_to_ind)
        st_all["gender"] = st_all["gender"].apply(_encode_gender)

        static_cols = ["age", "gender", "height"]
        D = len(static_cols)
        demo = np.full((self.N, D), np.nan, dtype=np.float32)
        for row in st_all.itertuples(index=False):
            i = int(row.ts_ind)
            demo[i, 0] = float(row.age) if not pd.isna(row.age) else np.nan
            demo[i, 1] = float(row.gender) if not pd.isna(row.gender) else np.nan
            demo[i, 2] = float(row.height) if not pd.isna(row.height) else np.nan

        tr = np.array(self.splits["train"], dtype=np.int64)
        col_means = np.nanmean(demo[tr], axis=0, keepdims=True)
        inds = np.where(np.isnan(demo))
        demo[inds] = np.take(col_means, inds[1])

        means = demo[tr].mean(axis=0, keepdims=True)
        stds = _safe_std(demo[tr])
        demo = (demo - means) / stds

        self.demo = demo
        args.D = D
        args.logger.write(f"# static features: {D} (normalized by source train)")

        # ---------------------------
        # Vocab + normalization stats from SOURCE TRAIN ONLY
        # ---------------------------
        src_train_itemids = src_train.loc[src_train["pid"].isin(src_train_pids), "itemid"].unique()
        itemids = np.array(sorted(src_train_itemids))
        itemid_to_ind = {int(v): i for i, v in enumerate(itemids)}
        self.var_ind = {str(k): v for k, v in itemid_to_ind.items()}  # keep compat
        V = len(itemids)
        args.V = V
        args.logger.write(f"# TS variables (itemid in source train): {V}")

        # filter all splits to source-train vocab
        def _filter_vocab(df):
            return df.loc[df["itemid"].isin(src_train_itemids)].copy()

        src_train = _filter_vocab(src_train)
        src_valid = _filter_vocab(src_valid)
        src_test  = _filter_vocab(src_test)
        tgt_test  = _filter_vocab(tgt_test)

        # compute mean/std ONLY on source train
        means_stds = (
            src_train.groupby("itemid")["value"]
            .agg(["mean", "std"])
            .reset_index()
        )
        means_stds.loc[means_stds["std"] == 0, "std"] = 1.0

        # max_offset used for time scaling (source-based)
        max_offset = float(src_train["offset"].max())
        if max_offset <= 0:
            # fallback: if offsets are all 0 (unlikely), use global max over source splits
            max_offset = float(pd.concat([src_train, src_valid, src_test], axis=0)["offset"].max())
        self.max_offset = max_offset

        # normalize values for ALL splits using source train stats
        def _apply_norm(df):
            out = df.merge(means_stds, on="itemid", how="left")
            out["value"] = (out["value"] - out["mean"]) / out["std"]
            return out

        src_train = _apply_norm(src_train)
        src_valid = _apply_norm(src_valid)
        src_test  = _apply_norm(src_test)
        tgt_test  = _apply_norm(tgt_test)

        # ---------------------------
        # Build event sequences for ALL PIDs (source+target), time scaled using SOURCE max_offset
        # ---------------------------
        df_all = pd.concat([src_train, src_valid, src_test, tgt_test], axis=0, ignore_index=True)
        df_all = df_all.loc[df_all["pid"].isin(self.pids)].copy()
        df_all["ts_ind"] = df_all["pid"].map(pid_to_ind)

        # optional trimming for event-based models
        if args.model_type in ["strats", "istrats", "ehrmamba", "duett", "trimba"]:
            df_all = df_all.sample(frac=1.0, random_state=getattr(args, "seed", 0))
            df_all = df_all.groupby("pid").head(getattr(args, "max_obs", 4096))
        else:
            raise NotImplementedError("This feather pipeline currently supports strats-like event models only.")

        # time normalization using SOURCE max_offset
        df_all["t"] = df_all["offset"].astype(np.float32) / float(max_offset) * 2.0 - 1.0

        values = [[] for _ in range(self.N)]
        times  = [[] for _ in range(self.N)]
        varis  = [[] for _ in range(self.N)]

        df_all = df_all.sort_values(["ts_ind", "offset"], kind="mergesort")
        for row in df_all.itertuples(index=False):
            i = int(row.ts_ind)
            values[i].append(float(row.value))
            times[i].append(float(row.t))
            varis[i].append(int(itemid_to_ind[int(row.itemid)]))

        self.values, self.times, self.varis = values, times, varis

        # cycler uses SOURCE train
        self.train_cycler = CycleIndex(self.splits["train"], args.train_batch_size)

# ---------------------------
# FeatherPretrainDataset (UNSUPERVISED forecasting)
# ---------------------------

class PretrainDataset:
    """
    Pretraining: given history up to time t1, predict last observed value per variable
    in (t1, t1+pred_window] (mask indicates which vars appear in that window).
    """

    def __init__(self, args):
        self.args = args
        args.data_dir = getattr(args, "data_dir", "./data")

        # load train/valid only for pretraining (no test)
        df_train = pd.read_feather(_path_data(args, "train"))
        df_valid = pd.read_feather(_path_data(args, "valid"))
        st_train = pd.read_feather(_path_static(args, "train"))
        st_valid = pd.read_feather(_path_static(args, "valid"))

        # build pid universe
        train_pids = df_train["pid"].unique()
        valid_pids = df_valid["pid"].unique()
        self.pids = np.concatenate([train_pids, valid_pids])
        pid_to_ind = {pid: i for i, pid in enumerate(self.pids)}

        self.splits = {
            "train": [pid_to_ind[p] for p in train_pids],
            "val":   [pid_to_ind[p] for p in valid_pids],
        }
        self.N = len(self.pids)

        # static
        st_all = pd.concat([st_train, st_valid], axis=0, ignore_index=True)
        st_all = st_all.loc[st_all["pid"].isin(self.pids)].copy()
        st_all["ts_ind"] = st_all["pid"].map(pid_to_ind)
        st_all["gender"] = st_all["gender"].apply(_encode_gender)

        demo = np.full((self.N, 3), np.nan, dtype=np.float32)
        for row in st_all.itertuples(index=False):
            i = int(row.ts_ind)
            demo[i, 0] = float(row.age) if not pd.isna(row.age) else np.nan
            demo[i, 1] = float(row.gender) if not pd.isna(row.gender) else np.nan
            demo[i, 2] = float(row.height) if not pd.isna(row.height) else np.nan

        tr = np.array(self.splits["train"], dtype=np.int64)
        col_means = np.nanmean(demo[tr], axis=0, keepdims=True)
        inds = np.where(np.isnan(demo))
        demo[inds] = np.take(col_means, inds[1])

        means = demo[tr].mean(axis=0, keepdims=True)
        stds = _safe_std(demo[tr])
        self.demo = (demo - means) / stds
        args.D = 3

        # variable mapping from TRAIN only
        train_itemids = df_train["itemid"].unique()
        itemids = np.array(sorted(train_itemids))
        itemid_to_ind = {int(v): i for i, v in enumerate(itemids)}
        self.itemids = itemids
        self.V = len(itemids)
        args.V = self.V

        # normalize values using TRAIN
        means_stds = (
            df_train.groupby("itemid")["value"]
            .agg(["mean", "std"])
            .reset_index()
        )
        means_stds.loc[means_stds["std"] == 0, "std"] = 1.0
        max_offset = float(pd.concat([df_train, df_valid], axis=0)["offset"].max())
        self.max_offset = max_offset

        def _apply_norm(df):
            out = df.loc[df["itemid"].isin(train_itemids)].copy()
            out = out.merge(means_stds, on="itemid", how="left")
            out["value"] = (out["value"] - out["mean"]) / out["std"]
            return out

        df_train = _apply_norm(df_train)
        df_valid = _apply_norm(df_valid)

        # save for downstream finetune compatibility if desired
        if getattr(args, "output_dir", None) is not None:
            pickle.dump([itemids, means_stds, max_offset],
                        open(os.path.join(args.output_dir, "pt_saved_variables.pkl"), "wb"))

        df_all = pd.concat([df_train, df_valid], axis=0, ignore_index=True)
        df_all = df_all.loc[df_all["pid"].isin(self.pids)].copy()
        df_all["ts_ind"] = df_all["pid"].map(pid_to_ind)
        df_all = df_all.sort_values(["ts_ind", "offset"], kind="mergesort")

        # build sequences
        values = [[] for _ in range(self.N)]
        times  = [[] for _ in range(self.N)]
        varis  = [[] for _ in range(self.N)]
        offsets = [[] for _ in range(self.N)]  # raw offsets (minutes)

        for row in df_all.itertuples(index=False):
            i = int(row.ts_ind)
            offsets[i].append(float(row.offset))
            times[i].append(float(row.offset / max_offset * 2.0 - 1.0))
            values[i].append(float(row.value))
            varis[i].append(int(itemid_to_ind[int(row.itemid)]))

        self.values = values
        self.times = times
        self.varis = varis
        self.offsets = offsets

        self.max_obs = getattr(args, "max_obs", 4096)
        self.pred_window = getattr(args, "pred_window", 120)  # minutes, default 2h
        self.min_history = getattr(args, "min_history", 720)  # minutes, default 12h

        # candidate t1 timestamps per sample: unique offsets >= min_history, and not last point
        self.timestamps = []
        for i in range(self.N):
            uniq = np.array(sorted(set(offsets[i])), dtype=np.float32)
            # need something after t1
            uniq = uniq[uniq >= self.min_history]
            if len(uniq) == 0:
                self.timestamps.append(np.array([], dtype=np.float32))
            else:
                # exclude max offset to ensure future window exists
                uniq = uniq[uniq < (np.max(offsets[i]) - 1e-6)]
                self.timestamps.append(uniq)

        # remove samples with no valid timestamps
        delete = np.array([i for i in range(self.N) if len(self.timestamps[i]) == 0], dtype=np.int64)
        self.splits = {k: np.setdiff1d(np.array(v, dtype=np.int64), delete) for k, v in self.splits.items()}

        self.train_cycler = CycleIndex(self.splits["train"], args.train_batch_size)

    def get_batch(self, ind=None):
        if ind is None:
            ind = self.train_cycler.get_batch_ind()
        bsz = len(ind)
        input_values, input_times, input_varis = [], [], []

        forecast_values = torch.zeros((bsz, self.V), dtype=torch.float32)
        forecast_mask   = torch.zeros((bsz, self.V), dtype=torch.int32)

        for b, i in enumerate(ind):
            # choose a cut time t1 (raw offset minutes)
            t1 = float(np.random.choice(self.timestamps[i]))

            # find index boundary: last index with offset <= t1
            offs = self.offsets[i]
            # since sequences are sorted by offset, find rightmost
            t1_ix = 0
            for ix in range(len(offs) - 1, -1, -1):
                if offs[ix] <= t1:
                    t1_ix = ix + 1
                    break

            t0_ix = max(0, t1_ix - self.max_obs)

            input_values.append(self.values[i][t0_ix:t1_ix])
            input_times.append(self.times[i][t0_ix:t1_ix])   # already normalized [-1,1]
            input_varis.append(self.varis[i][t0_ix:t1_ix])

            # prediction window
            t2 = t1 + float(self.pred_window)

            # gather last observed value for each variable in (t1, t2]
            # walk forward from t1_ix until offset > t2
            last_val = {}
            for ix in range(t1_ix, len(offs)):
                if offs[ix] > t2:
                    break
                v = self.varis[i][ix]
                last_val[v] = self.values[i][ix]

            for v, val in last_val.items():
                forecast_mask[b, v] = 1
                forecast_values[b, v] = float(val)

        # pad inputs
        num_obs = list(map(len, input_values))
        max_obs = max(num_obs) if len(num_obs) > 0 else 0
        pad_lens = max_obs - np.array(num_obs)

        values = [x + [0.0] * int(l) for x, l in zip(input_values, pad_lens)]
        times  = [x + [0.0] * int(l) for x, l in zip(input_times, pad_lens)]
        varis  = [x + [0]   * int(l) for x, l in zip(input_varis, pad_lens)]

        values = torch.FloatTensor(values)
        times  = torch.FloatTensor(times)
        varis  = torch.IntTensor(varis)

        obs_mask = [[1] * int(l1) + [0] * int(l2) for l1, l2 in zip(num_obs, pad_lens)]
        obs_mask = torch.IntTensor(obs_mask)

        return {
            "values": values,
            "times": times,
            "varis": varis,
            "obs_mask": obs_mask,
            "demo": torch.FloatTensor(self.demo[ind]),
            "forecast_values": forecast_values,
            "forecast_mask": forecast_mask,
        }


class Evaluator:
    def __init__(self, args):
        self.args = args

    def evaluate(self, model, dataset, split, train_step=None):
        eval_ind = dataset.splits[split]
        num_samples = len(eval_ind)
        model.eval()

        true_list, logit_list = [], []

        pbar = tqdm(range(0, num_samples, self.args.eval_batch_size),
                    desc=f"eval forward ({split})")
        for start in pbar:
            batch_ind = eval_ind[start:min(num_samples, start + self.args.eval_batch_size)]
            batch = dataset.get_batch(batch_ind)
            y_true = batch["labels"]  # [B, C]
            del batch["labels"]
            batch = {k: v.to(self.args.device) for k, v in batch.items()}

            with torch.no_grad():
                logits = model(**batch).detach().cpu()  # [B, C]

            true_list.append(y_true)
            logit_list.append(logits)

        y_true = torch.cat(true_list, dim=0).numpy().astype(np.float32)   # [N,C]
        y_logit = torch.cat(logit_list, dim=0).numpy().astype(np.float32) # [N,C]
        y_prob = 1.0 / (1.0 + np.exp(-y_logit))

        # loss (scalar) — like Raindrop's test/source_loss etc.
        bce_loss = F.binary_cross_entropy_with_logits(
            torch.tensor(y_logit, dtype=torch.float32),
            torch.tensor(y_true, dtype=torch.float32),
            reduction="mean",
        ).item()

        # per-label metrics + valid mask (labels with both pos/neg)
        C = y_true.shape[1]
        auroc_per = np.full((C,), np.nan, dtype=np.float32)
        auprc_per = np.full((C,), np.nan, dtype=np.float32)
        valid_mask = np.zeros((C,), dtype=bool)

        for j in range(C):
            yt = y_true[:, j]
            # need both classes present
            if yt.max() == yt.min():
                continue
            valid_mask[j] = True
            auroc_per[j] = roc_auc_score(yt, y_prob[:, j])
            auprc_per[j] = average_precision_score(yt, y_prob[:, j])

        valid_count = int(valid_mask.sum())
        auroc_macro = float(np.nanmean(auroc_per)) if valid_count > 0 else float("nan")
        auprc_macro = float(np.nanmean(auprc_per)) if valid_count > 0 else float("nan")

        # micro accuracy at 0.5 threshold
        y_hat = (y_prob >= 0.5).astype(np.float32)
        acc_micro = float((y_hat == y_true).mean())

        # Return BOTH:
        # - old keys (for your training loop compatibility if needed)
        # - raindrop-compatible keys (auroc/auprc/acc + per arrays/mask)
        result = {
            "train_loss": np.round(bce_loss, 6),

            # macro metrics (naming used in Raindrop: auroc/auprc/acc)
            "auroc": np.round(auroc_macro, 6),
            "auprc": np.round(auprc_macro, 6),
            "acc":  np.round(acc_micro,  6),

            # per-task arrays + mask
            "auroc_per": auroc_per,         # np.ndarray [C]
            "auprc_per": auprc_per,         # np.ndarray [C]
            "valid_mask": valid_mask,       # np.ndarray [C] bool
            "valid_labels": valid_count,    # int

            # keep these too (optional)
            "auroc_macro": np.round(auroc_macro, 6),
            "auprc_macro": np.round(auprc_macro, 6),
            "acc_micro": np.round(acc_micro, 6),
            "num_valid_labels": valid_count,
        }
        return result

class PretrainEvaluator:
    def __init__(self, args):
        self.args = args
        self.io = {}

    def evaluate(self, model, dataset, split, train_step=None):
        if split not in self.io:
            batches = []
            eval_ind = dataset.splits[split]
            num_samples = len(eval_ind)
            for start in tqdm(range(0, num_samples, self.args.eval_batch_size),
                              desc=f"generating io ({split})"):
                batch_ind = eval_ind[start:min(num_samples, start + self.args.eval_batch_size)]
                for _ in range(3):
                    batches.append(dataset.get_batch(batch_ind))
            self.io[split] = batches

        model.eval()
        pbar = tqdm(self.io[split], desc=f"pretrain eval forward ({split})")
        loss_sum, count_sum = 0.0, 0.0

        for batch in pbar:
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            with torch.no_grad():
                loss = model(**batch)  # scalar
                num_pred = batch["forecast_mask"].sum().item()
                if num_pred == 0:
                    continue
                loss_sum += float(loss.item()) * float(num_pred)
                count_sum += float(num_pred)

        if count_sum == 0:
            result = {"loss_neg": float("nan")}
        else:
            result = {"loss_neg": - (loss_sum / count_sum)}

        return result

def get_curr_time() -> str:
    """Get current date and time as str."""
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


class Logger:
    """Class to write message to both output_dir/filename.txt and terminal."""

    def __init__(self, output_dir: str = None, filename: str = None) -> None:
        if filename is not None:
            self.log = os.path.join(output_dir, filename)

    def write(self, message, show_time: bool = True) -> None:
        "write the message"
        message = str(message)
        if show_time:
            # if message starts with \n, print the \n first before printing time
            if message.startswith('\n'):
                message = '\n' + get_curr_time() + ' >> ' + message[1:]
            else:
                message = get_curr_time() + ' >> ' + message
        print(message)
        if hasattr(self, 'log'):
            with open(self.log, 'a') as f:
                f.write(message + '\n')

def count_parameters(logger: Logger, model: nn.Module):
    """Print no. of parameters in model, no. of traininable parameters,
     no. of parameters in each dtype."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.write('\nModel details:')
    logger.write('# parameters: ' + str(total))
    logger.write('# trainable parameters: ' + str(trainable) + ', ' \
                 + str(100 * trainable / total) + '%')

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    logger.write('#params by dtype:')
    for k, v in dtypes.items():
        logger.write(str(k) + ': ' + str(v) + ', ' + str(100 * v / total) + '%')