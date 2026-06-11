# mimic_eicu_duett_multilabel.py

import os
# os.environ["CUDA_VISIBLE_DEVICES"]= "1"
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import torchmetrics
import x_transformers

import re


def _safe_metric_key(name: str) -> str:
    return str(name).strip().replace(" ", "_").replace("/", "_")


@torch.no_grad()
def evaluate_on_dm(model: pl.LightningModule, dm: pl.LightningDataModule, device: torch.device):
    model.eval().to(device)

    dl = dm.test_dataloader()
    bce = nn.BCEWithLogitsLoss(reduction="sum")

    L = dm.d_target()
    auroc_per = torchmetrics.classification.MultilabelAUROC(num_labels=L, average=None).to(device)
    auprc_per = torchmetrics.classification.MultilabelAveragePrecision(num_labels=L, average=None).to(device)
    auroc_macro = torchmetrics.classification.MultilabelAUROC(num_labels=L, average="macro").to(device)
    auprc_macro = torchmetrics.classification.MultilabelAveragePrecision(num_labels=L, average="macro").to(device)
    acc_macro = torchmetrics.classification.MultilabelAccuracy(num_labels=L, average="macro", threshold=0.5).to(device)

    total_loss = 0.0
    total_n = 0

    for x, y in dl:
        y = y.to(device).float()
        bs = y.shape[0]
        logits = model(model.feats_to_input(x, bs))
        loss = bce(logits, y)

        probs = torch.sigmoid(logits)

        auroc_per.update(probs, y.int())
        auprc_per.update(probs, y.int())
        auroc_macro.update(probs, y.int())
        auprc_macro.update(probs, y.int())
        acc_macro.update(probs, y.int())

        total_loss += float(loss.item())
        total_n += bs

    avg_loss = total_loss / max(total_n, 1)

    auroc_macro_v = float(auroc_macro.compute().item())
    auprc_macro_v = float(auprc_macro.compute().item())
    acc_macro_v = float(acc_macro.compute().item())

    auroc_per_v = auroc_per.compute().detach().cpu().numpy()
    auprc_per_v = auprc_per.compute().detach().cpu().numpy()

    # valid mask: label all-0 or all-1 => undefined AUROC/AUPRC
    y_all = dm.ds_test.y
    pos = y_all.sum(dim=0)
    neg = y_all.shape[0] - pos
    valid_mask = ((pos > 0) & (neg > 0)).cpu().numpy()

    return avg_loss, auroc_macro_v, auprc_macro_v, acc_macro_v, auroc_per_v, auprc_per_v, valid_mask


# -----------------------------
# Utils
# -----------------------------
class BatchNormLastDim(nn.Module):
    def __init__(self, d, **kwargs):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(d, **kwargs)

    def forward(self, x):
        if x.ndim == 2:
            return self.batch_norm(x)
        elif x.ndim == 3:
            return self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        else:
            raise NotImplementedError("BatchNormLastDim not implemented for ndim > 3")


def simple_mlp(
    d_in,
    d_out,
    n_hidden,
    d_hidden,
    final_activation=False,
    input_batch_norm=False,
    hidden_batch_norm=False,
    dropout=0.0,
    activation=nn.ReLU,
):
    if n_hidden == 0:
        layers = ([BatchNormLastDim(d_in)] if input_batch_norm else []) + [nn.Linear(d_in, d_out)]
    else:
        layers = (
            ([BatchNormLastDim(d_in)] if input_batch_norm else [])
            + [nn.Linear(d_in, d_hidden), activation(), nn.Dropout(dropout)]
            + [
                l
                for _ in range(n_hidden - 1)
                for l in (
                    ([BatchNormLastDim(d_hidden)] if hidden_batch_norm else [])
                    + [nn.Linear(d_hidden, d_hidden), activation(), nn.Dropout(dropout)]
                )
            ]
            + ([BatchNormLastDim(d_hidden)] if hidden_batch_norm else [])
            + [nn.Linear(d_hidden, d_out)]
        )
    if final_activation:
        layers.append(activation())
    return nn.Sequential(*layers)


def collate_into_seqs(batch):
    xs, ys = zip(*batch)
    x_ts, x_static, bin_ends = zip(*xs)
    return (list(x_ts), list(x_static), list(bin_ends)), torch.stack(list(ys))


# -----------------------------
# Dataset / DataModule
# -----------------------------
@dataclass
class Normalizers:
    ts_mean: torch.Tensor
    ts_std: torch.Tensor
    static_mean: torch.Tensor
    static_std: torch.Tensor


class MimicEicuDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        source: str,
        stage: str,
        split: int,
        n_timesteps: int,
        itemid_to_idx: Dict[int, int],
        norm: Optional[Normalizers],
        use_temp_cache: bool = False,
        presence_as_count: bool = True,
        offset_unit: str = "minutes",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.source = source
        self.stage = stage
        self.split = split
        self.n_timesteps = n_timesteps
        self.itemid_to_idx = itemid_to_idx
        self.norm = norm
        self.temp_cache = {} if use_temp_cache else None
        self.presence_as_count = presence_as_count
        self.offset_unit = offset_unit

        self.df_ts = None
        self.df_static = None
        self.df_y = None
        self.pids: List[int] = []
        self.y: torch.Tensor = None
        self.outcome_cols: List[str] = []

    def _path(self, kind: str):
        return os.path.join(self.data_dir, f"{self.source}_{kind}_{self.stage}_{self.split}.feather")

    def d_static_num(self):
        return 3

    def d_time_series_num(self):
        return len(self.itemid_to_idx)

    def d_target(self):
        return len(self.outcome_cols)

    def setup(self, outcome_exclude=("aki_label", "cf_label")):
        ts_path = self._path("data")
        st_path = self._path("data_static")
        y_path = self._path("outcomes")

        self.df_ts = pd.read_feather(ts_path)
        self.df_static = pd.read_feather(st_path)
        self.df_y = pd.read_feather(y_path)

        all_cols = [c for c in self.df_y.columns if c != "pid"]
        self.outcome_cols = [c for c in all_cols if c not in set(outcome_exclude)]
        if len(self.outcome_cols) == 0:
            raise ValueError("No outcome columns left after exclusion.")

        self.pids = self.df_y["pid"].astype(int).tolist()
        y_mat = self.df_y[self.outcome_cols].to_numpy(dtype=np.float32)
        self.y = torch.from_numpy(y_mat)

        self._ts_gb = self.df_ts.groupby("pid", sort=False)
        self._st_gb = self.df_static.set_index("pid", drop=False)

    def pos_frac(self):
        return self.y.mean(dim=0).cpu().numpy()

    def _offset_to_days(self, offset: np.ndarray) -> np.ndarray:
        if self.offset_unit == "minutes":
            return offset / 60.0 / 24.0
        if self.offset_unit == "hours":
            return offset / 24.0
        if self.offset_unit == "seconds":
            return offset / 3600.0 / 24.0
        if self.offset_unit == "two_days":
            return offset * 2
        raise ValueError(f"Unknown offset_unit: {self.offset_unit}")

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx: int):
        if self.temp_cache is not None and idx in self.temp_cache:
            return self.temp_cache[idx]

        pid = self.pids[idx]

        if pid in self._ts_gb.indices:
            g = self._ts_gb.get_group(pid)
            offsets = g["offset"].to_numpy(dtype=np.float64)
            t_days = self._offset_to_days(offsets)
            itemids = g["itemid"].to_numpy()
            values = g["value"].to_numpy(dtype=np.float32)
        else:
            t_days = np.array([], dtype=np.float64)
            itemids = np.array([], dtype=np.int64)
            values = np.array([], dtype=np.float32)

        if pid in self._st_gb.index:
            st_row = self._st_gb.loc[pid]
            age = float(st_row.get("age", 0.0))
            height = float(st_row.get("height", 0.0))
            gender = st_row.get("gender", 0)
        else:
            age, height, gender = 0.0, 0.0, 0

        if isinstance(gender, str):
            g_up = gender.strip().upper()
            if g_up in ["M", "MALE"]:
                gender_val = 1.0
            elif g_up in ["F", "FEMALE"]:
                gender_val = 0.0
            else:
                gender_val = 0.0
        else:
            gender_val = 1.0 if float(gender) > 0 else 0.0

        x_static = torch.tensor([age, gender_val, height], dtype=torch.float32)

        if self.norm is not None:
            x_static = (x_static - self.norm.static_mean) / (self.norm.static_std + 1e-7)
            x_static = torch.nan_to_num(x_static, nan=0.0, posinf=0.0, neginf=0.0)

        d_ts = self.d_time_series_num()
        x_ts = torch.zeros((self.n_timesteps, d_ts * 2), dtype=torch.float32)

        if len(t_days) > 0:
            t_last = float(np.max(t_days))
            if t_last <= 0:
                t_last = 1e-6

            for t, itemid, val in zip(t_days, itemids, values):
                itemid = int(itemid)
                if itemid not in self.itemid_to_idx:
                    continue
                j = self.itemid_to_idx[itemid]

                b = (self.n_timesteps - 1) if (t == t_last) else int(t / t_last * self.n_timesteps)
                b = min(max(int(b), 0), self.n_timesteps - 1)

                v = torch.tensor(val, dtype=torch.float32)
                if self.norm is not None:
                    v = (v - self.norm.ts_mean[j]) / (self.norm.ts_std[j] + 1e-7)
                    if torch.isnan(v) or torch.isinf(v):
                        continue

                x_ts[b, j] = v
                if self.presence_as_count:
                    x_ts[b, j + d_ts] += 1.0
                else:
                    x_ts[b, j + d_ts] = 1.0

            bin_ends = torch.arange(1, self.n_timesteps + 1, dtype=torch.float32) / self.n_timesteps * float(t_last)
        else:
            bin_ends = torch.zeros((self.n_timesteps,), dtype=torch.float32)

        x = (x_ts, x_static, bin_ends)
        y = self.y[idx]

        if self.temp_cache is not None:
            self.temp_cache[idx] = (x, y)

        return x, y


def build_item_vocab_from_train(data_dir: str, source: str, split: int) -> Dict[int, int]:
    path = os.path.join(data_dir, f"{source}_data_train_{split}.feather")
    df = pd.read_feather(path)
    itemids = df["itemid"].dropna().astype(int).unique()
    itemids = np.sort(itemids)
    return {int(it): i for i, it in enumerate(itemids)}


def compute_normalizers_from_train(data_dir: str, source: str, itemid_to_idx: Dict[int, int], split: int) -> Normalizers:
    df = pd.read_feather(os.path.join(data_dir, f"{source}_data_train_{split}.feather"))
    df = df.dropna(subset=["itemid", "value"])
    df["itemid"] = df["itemid"].astype(int)

    d_ts = len(itemid_to_idx)
    ts_mean = torch.zeros((d_ts,), dtype=torch.float32)
    ts_std = torch.ones((d_ts,), dtype=torch.float32)

    for itemid, g in df.groupby("itemid"):
        itemid = int(itemid)
        if itemid not in itemid_to_idx:
            continue
        j = itemid_to_idx[itemid]
        vals = torch.tensor(g["value"].to_numpy(dtype=np.float32))
        if vals.numel() == 0:
            continue
        ts_mean[j] = vals.mean()
        ts_std[j] = vals.std(unbiased=False) if vals.numel() > 1 else torch.tensor(1.0)

    st = pd.read_feather(os.path.join(data_dir, f"{source}_data_static_train_{split}.feather"))

    def gender_to_float(x):
        if isinstance(x, str):
            u = x.strip().upper()
            if u in ["M", "MALE"]:
                return 1.0
            if u in ["F", "FEMALE"]:
                return 0.0
            return 0.0
        try:
            return 1.0 if float(x) > 0 else 0.0
        except Exception:
            return 0.0

    age = st["age"].fillna(0).astype(float).to_numpy()
    height = st["height"].fillna(0).astype(float).to_numpy()
    gender = st["gender"].apply(gender_to_float).to_numpy(dtype=np.float32)

    static = torch.tensor(np.stack([age, gender, height], axis=1), dtype=torch.float32)
    static_mean = static.mean(dim=0)
    static_std = static.std(dim=0, unbiased=False)
    static_std = torch.where(static_std > 0, static_std, torch.ones_like(static_std))

    return Normalizers(ts_mean=ts_mean, ts_std=ts_std, static_mean=static_mean, static_std=static_std)


class MimicEicuDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        source: str = "mimic",
        split: int = 1,
        n_timesteps: int = 32,
        batch_size: int = 64,
        num_workers: int = 8,
        prefetch_factor: int = 2,
        use_temp_cache: bool = False,
        presence_as_count: bool = True,
        offset_unit: str = "minutes",
        exclude_outcomes=("aki_label", "cf_label"),
        # ✅ NEW: allow injecting source-train vocab/norm for target scaling
        itemid_to_idx: Optional[Dict[int, int]] = None,
        norm: Optional[Normalizers] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.source = source
        self.split = split
        self.n_timesteps = n_timesteps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.use_temp_cache = use_temp_cache
        self.presence_as_count = presence_as_count
        self.offset_unit = offset_unit
        self.exclude_outcomes = exclude_outcomes

        # ✅ if provided, we will not recompute
        self.itemid_to_idx = itemid_to_idx
        self.norm = norm

        self.ds_train = None
        self.ds_val = None
        self.ds_test = None

        self.dl_args = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_into_seqs,
            pin_memory=True,
            persistent_workers=True,
        )

    def prepare_data(self):
        for i in ["train", "valid", "test"]:
            for kind in ["data", "data_static", "outcomes"]:
                p = os.path.join(self.data_dir, f"{self.source}_{kind}_{i}_{self.split}.feather")
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Missing file: {p}")

    def setup(self, stage=None):
        # ✅ vocab + normalizers always from (source) TRAIN unless injected
        if self.itemid_to_idx is None:
            self.itemid_to_idx = build_item_vocab_from_train(self.data_dir, self.source, self.split)
        if self.norm is None:
            self.norm = compute_normalizers_from_train(self.data_dir, self.source, self.itemid_to_idx, self.split)

        def make(stage):
            ds = MimicEicuDataset(
                data_dir=self.data_dir,
                source=self.source,
                stage=stage,
                split=self.split,
                n_timesteps=self.n_timesteps,
                itemid_to_idx=self.itemid_to_idx,
                norm=self.norm,
                use_temp_cache=self.use_temp_cache,
                presence_as_count=self.presence_as_count,
                offset_unit=self.offset_unit,
            )
            ds.setup(outcome_exclude=self.exclude_outcomes)
            return ds

        if stage is None or stage == "fit":
            self.ds_train = make("train")
            self.ds_val = make("valid")
            self.ds_test = make("test")
        elif stage == "validate":
            self.ds_val = make("valid")
        elif stage == "test":
            self.ds_test = make("test")

    def train_dataloader(self):
        return DataLoader(self.ds_train, shuffle=True, **self.dl_args)

    def val_dataloader(self):
        return DataLoader(self.ds_val, shuffle=False, **self.dl_args)

    def test_dataloader(self):
        return DataLoader(self.ds_test, shuffle=False, **self.dl_args)

    def d_static_num(self):
        return self.ds_train.d_static_num() if self.ds_train is not None else self.ds_test.d_static_num()

    def d_time_series_num(self):
        return self.ds_train.d_time_series_num() if self.ds_train is not None else self.ds_test.d_time_series_num()

    def d_target(self):
        return self.ds_train.d_target() if self.ds_train is not None else self.ds_test.d_target()

    def pos_frac(self):
        return self.ds_train.pos_frac()

    def outcome_cols(self):
        return self.ds_train.outcome_cols if self.ds_train is not None else self.ds_test.outcome_cols


# -----------------------------
# Model (DUETT-style) with multilabel
# -----------------------------
def pretrain_model(d_static_num, d_time_series_num, d_target, **kwargs):
    return Model(d_static_num, d_time_series_num, d_target, **kwargs)


def fine_tune_model(ckpt_path, **kwargs):
    return Model.load_from_checkpoint(
        ckpt_path,
        pretrain=False,
        aug_noise=0.0,
        aug_mask=0.5,
        transformer_dropout=0.5,
        lr=1e-3,
        weight_decay=1e-5,
        fusion_method="rep_token",
        **kwargs,
    )


class Model(pl.LightningModule):
    def __init__(
        self,
        d_static_num,
        d_time_series_num,
        d_target,
        lr=1e-3,
        weight_decay=1e-1,
        glu=False,
        scalenorm=True,
        n_hidden_mlp_embedding=1,
        d_hidden_mlp_embedding=64,
        d_embedding=24,
        d_feedforward=512,
        max_len=48,
        n_transformer_head=2,
        n_duett_layers=2,
        d_hidden_tab_encoder=128,
        n_hidden_tab_encoder=1,
        norm_first=True,
        fusion_method="masked_embed",
        n_hidden_head=1,
        d_hidden_head=64,
        aug_noise=0.0,
        aug_mask=0.0,
        pretrain=True,
        pretrain_masked_steps=1,
        pretrain_n_hidden=0,
        pretrain_d_hidden=64,
        pretrain_dropout=0.5,
        pretrain_value=True,
        pretrain_presence=True,
        pretrain_presence_weight=0.2,
        predict_events=True,
        transformer_dropout=0.0,
        pos_frac=None,
        freeze_encoder=False,
        seed=0,
        save_representation=None,
        masked_transform_timesteps=32,
        outcome_names: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["outcome_names"])

        self.lr = lr
        self.weight_decay = weight_decay
        self.d_time_series_num = d_time_series_num
        self.d_target = d_target
        self.d_embedding = d_embedding
        self.max_len = max_len
        self.pretrain = pretrain
        self.pretrain_masked_steps = pretrain_masked_steps
        self.pretrain_dropout = pretrain_dropout
        self.freeze_encoder = freeze_encoder
        self.rng = np.random.default_rng(seed)
        self.aug_noise = aug_noise
        self.aug_mask = aug_mask
        self.fusion_method = fusion_method
        self.pretrain_presence = pretrain_presence
        self.pretrain_presence_weight = pretrain_presence_weight
        self.predict_events = predict_events
        self.masked_transform_timesteps = masked_transform_timesteps
        self.pretrain_value = pretrain_value
        self.save_representation = save_representation

        self.outcome_names = outcome_names

        self.register_buffer("MASKED_EMBEDDING_KEY", torch.tensor(0))
        self.register_buffer("REPRESENTATION_EMBEDDING_KEY", torch.tensor(1))

        self.special_embeddings = nn.Embedding(8, d_embedding)
        self.embedding_layers = nn.ModuleList(
            [
                simple_mlp(2, d_embedding, n_hidden_mlp_embedding, d_hidden_mlp_embedding, hidden_batch_norm=True)
                for _ in range(d_time_series_num)
            ]
        )

        self.n_obs_embedding = nn.Embedding(16, 1)

        if d_feedforward is None:
            d_feedforward = d_embedding * 4

        et_dim = d_embedding * (masked_transform_timesteps + 1)
        tt_dim = d_embedding * (d_time_series_num + 1)

        self.event_transformers = nn.ModuleList(
            [
                x_transformers.Encoder(
                    dim=et_dim,
                    depth=1,
                    heads=n_transformer_head,
                    pre_norm=norm_first,
                    use_scalenorm=scalenorm,
                    attn_dim_head=d_embedding // n_transformer_head,
                    ff_glu=glu,
                    ff_mult=d_feedforward / et_dim,
                    attn_dropout=transformer_dropout,
                    ff_dropout=transformer_dropout,
                )
                for _ in range(n_duett_layers)
            ]
        )
        self.full_event_embedding = nn.Embedding(d_time_series_num + 1, et_dim)

        self.time_transformers = nn.ModuleList(
            [
                x_transformers.Encoder(
                    dim=tt_dim,
                    depth=1,
                    heads=n_transformer_head,
                    pre_norm=norm_first,
                    use_scalenorm=scalenorm,
                    attn_dim_head=d_embedding // n_transformer_head,
                    ff_glu=glu,
                    ff_mult=d_feedforward / tt_dim,
                    attn_dropout=transformer_dropout,
                    ff_dropout=transformer_dropout,
                )
                for _ in range(n_duett_layers)
            ]
        )

        self.full_time_embedding = self.cve(batch_norm=True, d_embedding=tt_dim)
        self.full_rep_embedding = nn.Embedding(tt_dim, 1)

        d_representation = d_embedding * (d_time_series_num + 1)
        self.head = simple_mlp(
            d_representation, d_target, n_hidden_head, d_hidden_head, hidden_batch_norm=True, final_activation=False
        )

        self.pretrain_value_proj = simple_mlp(
            d_representation, d_time_series_num, pretrain_n_hidden, pretrain_d_hidden, hidden_batch_norm=True
        )
        if self.pretrain_presence:
            self.pretrain_presence_proj = simple_mlp(
                d_representation, d_time_series_num, pretrain_n_hidden, pretrain_d_hidden, hidden_batch_norm=True
            )
        if self.predict_events:
            self.predict_events_proj = simple_mlp(
                et_dim, masked_transform_timesteps, pretrain_n_hidden, pretrain_d_hidden, hidden_batch_norm=True
            )
            if self.pretrain_presence:
                self.predict_events_presence_proj = simple_mlp(
                    et_dim, masked_transform_timesteps, pretrain_n_hidden, pretrain_d_hidden, hidden_batch_norm=True
                )

        self.tab_encoder = simple_mlp(
            d_static_num, d_embedding, n_hidden_tab_encoder, d_hidden_tab_encoder, hidden_batch_norm=True
        )

        self.pretrain_loss = F.mse_loss
        self.pretrain_presence_loss = F.binary_cross_entropy_with_logits

        self.loss_function = nn.BCEWithLogitsLoss(reduction="mean")

        self.train_auroc = torchmetrics.classification.MultilabelAUROC(num_labels=d_target, average="macro")
        self.val_auroc = torchmetrics.classification.MultilabelAUROC(num_labels=d_target, average="macro")
        self.test_auroc = torchmetrics.classification.MultilabelAUROC(num_labels=d_target, average="macro")

        self.train_ap = torchmetrics.classification.MultilabelAveragePrecision(num_labels=d_target, average="macro")
        self.val_ap = torchmetrics.classification.MultilabelAveragePrecision(num_labels=d_target, average="macro")
        self.test_ap = torchmetrics.classification.MultilabelAveragePrecision(num_labels=d_target, average="macro")

    def cve(self, d_embedding=None, batch_norm=False):
        if d_embedding is None:
            d_embedding = self.d_embedding
        d_hidden = int(np.sqrt(d_embedding))
        if batch_norm:
            return nn.Sequential(
                nn.Linear(1, d_hidden),
                nn.Tanh(),
                BatchNormLastDim(d_hidden),
                nn.Linear(d_hidden, d_embedding),
            )
        return nn.Sequential(nn.Linear(1, d_hidden), nn.Tanh(), nn.Linear(d_hidden, d_embedding))

    def feats_to_input(self, x, batch_size):
        xs_ts, xs_static, times = x
        xs_ts = list(xs_ts)

        for i, f in enumerate(xs_ts):
            n_vars = f.shape[1] // 2
            if f.shape[0] > self.max_len:
                f = f[-self.max_len:]
                times[i] = times[i][-self.max_len:]

            if self.training and self.aug_noise > 0 and (not self.pretrain):
                f[:, :n_vars] += self.aug_noise * torch.randn_like(f[:, :n_vars]) * f[:, n_vars:]

            f = torch.cat((f, torch.zeros_like(f[:, :1])), dim=1)

            if self.training and self.aug_mask > 0 and (not self.pretrain):
                mask = torch.rand(f.shape[0]) < self.aug_mask
                f[mask, :] = 0.0
                f[mask, -1] = 1.0

            xs_ts[i] = f

        n_timesteps = [len(ts) for ts in times]
        pad_to = int(np.max(n_timesteps)) if len(n_timesteps) > 0 else 1

        xs_ts = torch.stack([F.pad(t, (0, 0, 0, pad_to - t.shape[0])) for t in xs_ts]).to(self.device)
        xs_times = torch.stack([F.pad(t, (0, pad_to - t.shape[0])) for t in times]).to(self.device)
        xs_static = torch.stack(xs_static).to(self.device)

        if self.training and self.aug_noise > 0 and (not self.pretrain):
            xs_static += self.aug_noise * torch.randn_like(xs_static)

        return xs_static, xs_ts, xs_times, n_timesteps

    def pretrain_prep_batch(self, x, batch_size):
        xs_static, xs_ts, xs_times, n_timesteps = self.feats_to_input(x, batch_size)
        n_vars = (xs_ts.shape[2] - 1) // 2

        y_ts, y_ts_n_obs = [], []
        y_events, y_events_mask = [], []
        xs_ts_clipped = xs_ts.clone()

        for batch_i, n in enumerate(n_timesteps):
            if n < 2:
                mask_i = n
            elif self.pretrain_masked_steps > 1:
                if self.pretrain_masked_steps > n:
                    mask_i = np.arange(n)
                else:
                    mask_i = self.rng.choice(np.arange(n), size=self.pretrain_masked_steps, replace=False)
            else:
                mask_i = self.rng.choice(np.arange(0, n))

            y_ts.append(xs_ts[batch_i, mask_i, :n_vars])
            y_ts_n_obs.append(xs_ts[batch_i, mask_i, n_vars : 2 * n_vars])

            xs_ts_clipped[batch_i, mask_i, :] = 0.0
            xs_ts_clipped[batch_i, mask_i, -1] = 1.0

            if self.predict_events:
                event_mask_i = self.rng.choice(np.arange(0, self.d_time_series_num))
                y_events.append(xs_ts[batch_i, :, event_mask_i])
                y_events_mask.append(xs_ts[batch_i, :, event_mask_i + n_vars].clip(0, 1))
                xs_ts_clipped[batch_i, :, event_mask_i] = 0
                xs_ts_clipped[batch_i, :, event_mask_i + n_vars] = -1

        y_ts = torch.stack(y_ts)
        y_ts_n_obs = torch.stack(y_ts_n_obs)
        y_ts_masks = y_ts_n_obs.clip(0, 1)

        if len(y_events) > 0:
            y_events = torch.stack(y_events)
            y_events_mask = torch.stack(y_events_mask)

        if self.pretrain_dropout > 0:
            keep = self.rng.random((batch_size, n_vars)) > self.pretrain_dropout
            keep = torch.tensor(keep, device=xs_ts.device)

            if y_ts_masks.ndim > 2:
                keep = torch.logical_or(1 - y_ts_masks.sum(dim=1).clip(0, 1), keep)
            else:
                keep = torch.logical_or(1 - y_ts_masks, keep)

            keep = torch.cat((keep.tile(1, 2), torch.ones((batch_size, 1), device=keep.device)), dim=1)
            xs_ts_clipped *= torch.logical_or(keep.unsqueeze(1), xs_ts_clipped == -1)

        return (xs_static, xs_ts_clipped, xs_times, n_timesteps), y_ts, y_ts_masks, y_events, y_events_mask

    def forward(self, x, pretrain=False, representation=False):
        xs_static, xs_feats, xs_times, n_timesteps = x
        n_vars = xs_feats.shape[2] // 2

        if self.predict_events:
            event_mask_inds = xs_feats[:, :, n_vars : n_vars * 2] == -1
            event_mask_inds = torch.cat(
                (event_mask_inds, torch.zeros(xs_feats.shape[:2] + (1,), device=xs_feats.device, dtype=torch.bool)),
                dim=2,
            )
            event_mask_inds = torch.cat((event_mask_inds, event_mask_inds[:, :1, :]), dim=1)

        n_obs_inds = xs_feats[:, :, n_vars : n_vars * 2].to(int).clip(0, self.n_obs_embedding.num_embeddings - 1)
        xs_feats[:, :, n_vars : n_vars * 2] = self.n_obs_embedding(n_obs_inds).squeeze(-1)

        embedding_layer_input = torch.empty(xs_feats.shape[:-1] + (n_vars, 2), dtype=xs_feats.dtype, device=xs_feats.device)
        embedding_layer_input[:, :, :, 0] = xs_feats[:, :, :n_vars]
        embedding_layer_input[:, :, :, 1] = xs_feats[:, :, n_vars : n_vars * 2]

        psi = torch.zeros(
            (xs_feats.shape[0], xs_feats.shape[1] + 1, n_vars + 1, self.d_embedding),
            dtype=xs_feats.dtype,
            device=xs_feats.device,
        )
        for i, el in enumerate(self.embedding_layers):
            psi[:, :-1, i, :] = el(embedding_layer_input[:, :, i, :])

        psi[:, :-1, -1, :] = self.tab_encoder(xs_static).unsqueeze(1)
        psi[:, -1, :, :] = self.special_embeddings(self.REPRESENTATION_EMBEDDING_KEY.to(self.device)).unsqueeze(0).unsqueeze(1)

        mask_inds = torch.cat(
            (xs_feats[:, :, -1] == 1, torch.zeros((xs_feats.shape[0], 1), device=xs_feats.device, dtype=torch.bool)),
            dim=1,
        )
        psi[mask_inds, :, :] = self.special_embeddings(self.MASKED_EMBEDDING_KEY.to(self.device))

        if self.predict_events:
            psi[event_mask_inds, :] = self.special_embeddings(self.MASKED_EMBEDDING_KEY.to(self.device))

        time_embeddings = self.full_time_embedding(xs_times.unsqueeze(2))
        time_embeddings = torch.cat(
            (time_embeddings, self.full_rep_embedding.weight.T.unsqueeze(0).expand(xs_feats.shape[0], -1, -1)),
            dim=1,
        )

        for event_transformer, time_transformer in zip(self.event_transformers, self.time_transformers):
            et_out_shape = (psi.shape[0], psi.shape[2], psi.shape[1], psi.shape[3])
            embeddings = psi.transpose(1, 2).flatten(2) + self.full_event_embedding.weight.unsqueeze(0)
            event_outs = event_transformer(embeddings).view(et_out_shape).transpose(1, 2)

            tt_out_shape = event_outs.shape
            embeddings = event_outs.flatten(2) + time_embeddings
            psi = time_transformer(embeddings).view(tt_out_shape)

        transformed = psi.flatten(2)

        if self.fusion_method == "rep_token":
            z_ts = transformed[:, -1, :]
        elif self.fusion_method == "masked_embed":
            if self.pretrain_masked_steps > 1:
                masked_ind = F.pad(xs_feats[:, :, -1] > 0, (0, 1), value=False)
                z_ts = []
                for i in range(transformed.shape[0]):
                    z_ts.append(
                        F.pad(
                            transformed[i, masked_ind[i], :],
                            (0, 0, 0, self.pretrain_masked_steps - masked_ind[i].sum()),
                            value=0.0,
                        )
                    )
                z_ts = torch.stack(z_ts)
            else:
                masked_ind = xs_feats[:, :, -1:]
                z_ts = []
                for i in range(transformed.shape[0]):
                    z_ts.append(transformed[i, torch.nonzero(masked_ind[i].squeeze() == 1), :])
                z_ts = torch.cat(z_ts, dim=0).squeeze()
        elif self.fusion_method == "averaging":
            z_ts = torch.mean(transformed[:, :-1, :], dim=1)
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")

        z = z_ts
        if representation:
            return z

        if pretrain:
            y_hat_presence = self.pretrain_presence_proj(z).squeeze() if self.pretrain_presence else None
            y_hat_value = self.pretrain_value_proj(z).squeeze(1) if self.pretrain_value else None

            y_hat_events, y_hat_events_presence = None, None
            if self.predict_events:
                z_events = []
                for i in range(event_mask_inds.shape[0]):
                    z_events.append(psi[i][event_mask_inds[i].nonzero(as_tuple=True)].flatten())
                z_events = torch.stack(z_events)
                y_hat_events = self.predict_events_proj(z_events).squeeze()
                y_hat_events_presence = self.predict_events_presence_proj(z_events).squeeze() if self.pretrain_presence else None

            return y_hat_value, y_hat_presence, y_hat_events, y_hat_events_presence

        out = self.head(z)
        if self.save_representation:
            return out, z
        return out

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.to(self.device).float()
        batch_size = y.shape[0]

        if self.pretrain:
            x_pretrain, y_ts, mask, y_events, y_events_mask = self.pretrain_prep_batch(x, batch_size)
            y_hat_value, y_hat_presence, y_hat_events, y_hat_events_presence = self.forward(x_pretrain, pretrain=True)

            loss = 0.0
            if self.pretrain_value:
                if self.pretrain_masked_steps > 1:
                    vloss = 0.0
                    for i in range(self.pretrain_masked_steps):
                        vloss += self.pretrain_loss(y_hat_value[:, i] * mask[:, i], y_ts[:, i] * mask[:, i])
                    vloss /= self.pretrain_masked_steps
                else:
                    vloss = self.pretrain_loss(y_hat_value * mask, y_ts * mask)
                loss += vloss

            if self.pretrain_presence:
                if self.pretrain_masked_steps > 1:
                    ploss = 0.0
                    for i in range(self.pretrain_masked_steps):
                        ploss += self.pretrain_presence_loss(y_hat_presence[:, i], mask[:, i]) * self.pretrain_presence_weight
                    ploss /= self.pretrain_masked_steps
                else:
                    ploss = self.pretrain_presence_loss(y_hat_presence, mask) * self.pretrain_presence_weight
                loss += ploss

            if self.predict_events:
                if self.pretrain_value:
                    loss += self.pretrain_loss(y_hat_events * y_events_mask, y_events * y_events_mask)
                if self.pretrain_presence:
                    loss += self.pretrain_presence_loss(y_hat_events_presence, y_events_mask) * self.pretrain_presence_weight

            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            return loss

        logits = self.forward(self.feats_to_input(x, batch_size))
        loss = self.loss_function(logits, y)

        probs = torch.sigmoid(logits)
        self.train_auroc.update(probs, y.int())
        self.train_ap.update(probs, y.int())

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        if not self.pretrain:
            self.log("train_auroc", self.train_auroc.compute(), sync_dist=True)
            self.log("train_ap", self.train_ap.compute(), sync_dist=True)
            self.train_auroc.reset()
            self.train_ap.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.to(self.device).float()
        batch_size = y.shape[0]

        if self.pretrain:
            x_pretrain, y_ts, mask, y_events, y_events_mask = self.pretrain_prep_batch(x, batch_size)
            y_hat_value, y_hat_presence, y_hat_events, y_hat_events_presence = self.forward(x_pretrain, pretrain=True)

            loss = 0.0
            if self.pretrain_value:
                if self.pretrain_masked_steps > 1:
                    vloss = 0.0
                    for i in range(self.pretrain_masked_steps):
                        vloss += self.pretrain_loss(y_hat_value[:, i] * mask[:, i], y_ts[:, i] * mask[:, i])
                    vloss /= self.pretrain_masked_steps
                else:
                    vloss = self.pretrain_loss(y_hat_value * mask, y_ts * mask)
                loss += vloss
                self.log("val_next_loss", vloss, on_epoch=True, sync_dist=True, prog_bar=False)

            if self.pretrain_presence:
                if self.pretrain_masked_steps > 1:
                    ploss = 0.0
                    for i in range(self.pretrain_masked_steps):
                        ploss += self.pretrain_presence_loss(y_hat_presence[:, i], mask[:, i]) * self.pretrain_presence_weight
                    ploss /= self.pretrain_masked_steps
                else:
                    ploss = self.pretrain_presence_loss(y_hat_presence, mask) * self.pretrain_presence_weight
                loss += ploss
                self.log("val_presence_loss", ploss, on_epoch=True, sync_dist=True, prog_bar=False)

            if self.predict_events:
                eloss = self.pretrain_loss(y_hat_events * y_events_mask, y_events * y_events_mask)
                loss += eloss
                self.log("val_event_loss", eloss, on_epoch=True, sync_dist=True, prog_bar=False)

            self.log("val_loss", loss, on_epoch=True, sync_dist=True, prog_bar=True)
            return loss

        logits = self.forward(self.feats_to_input(x, batch_size))
        loss = self.loss_function(logits, y)
        probs = torch.sigmoid(logits)

        self.val_auroc.update(probs, y.int())
        self.val_ap.update(probs, y.int())

        self.log("val_loss", loss, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if not self.pretrain:
            val_auroc = self.val_auroc.compute()
            val_ap = self.val_ap.compute()
            self.log("val_auroc", val_auroc, sync_dist=True)
            self.log("val_ap", val_ap, sync_dist=True)
            self.val_auroc.reset()
            self.val_ap.reset()


# -----------------------------
# Warmup / Averaging
# -----------------------------
class WarmUpCallback(pl.callbacks.Callback):
    def __init__(self, steps=1000, base_lr=None, invsqrt=True, decay=None):
        self.warmup_steps = steps
        self.decay = steps if decay is None else decay
        self.state = {"steps": 0, "base_lr": None if base_lr is None else float(base_lr)}
        self.invsqrt = invsqrt

    def set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        opt = model.optimizers()
        if self.state["steps"] < self.warmup_steps:
            if self.state["base_lr"] is None:
                self.state["base_lr"] = opt.param_groups[0]["lr"]
            lr = self.state["steps"] / self.warmup_steps * self.state["base_lr"]
            self.set_lr(opt, lr)
            self.state["steps"] += 1
        elif self.invsqrt:
            if self.state["base_lr"] is None:
                self.state["base_lr"] = opt.param_groups[0]["lr"]
            lr = self.state["base_lr"] * (self.decay / (self.state["steps"] - self.warmup_steps + self.decay)) ** 0.5
            self.set_lr(opt, lr)
            self.state["steps"] += 1

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()


def average_models(models: List[Model]) -> Model:
    models = list(models)
    n = len(models)
    sds = [m.state_dict() for m in models]
    averaged = {}
    for k in sds[0]:
        averaged[k] = sum(sd[k] for sd in sds) / n
    models[0].load_state_dict(averaged)
    return models[0]


# -----------------------------
# Train script
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--split", type=int, default=1, choices=[1,2,3,4,5])
    parser.add_argument("--source", type=str, choices=["mimic", "eicu"], required=True)
    # ✅ target is mandatory
    parser.add_argument("--target", type=str, choices=["mimic", "eicu"], required=True)

    parser.add_argument("--n_timesteps", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--use_temp_cache", action="store_true")
    parser.add_argument("--offset_unit", type=str, default="minutes", choices=["seconds", "minutes", "hours"])

    parser.add_argument("--seed", type=int, default=9871)
    parser.add_argument("--pretrain_patience", type=int, default=7)
    parser.add_argument("--finetune_patience", type=int, default=7)
    parser.add_argument("--min_delta", type=float, default=0.0)
    # wandb
    parser.add_argument("--use_wandb", action="store_false") # No input defaults to using wandb
    parser.add_argument("--wandb_project", type=str, default="Baselines")
    parser.add_argument("--wandb_entity", type=str, default="jwseo118-korea-university")
    parser.add_argument("--wandb_name", type=str, default=None)

    parser.add_argument("--pretrain_epochs", type=int, default=30)
    parser.add_argument("--finetune_epochs", type=int, default=30)

    parser.add_argument("--save_dir", type=str, default="./baseline_models_duett")

    args = parser.parse_args()

    src_data_dir = os.path.join(args.data_root, f"split{args.split}")
    tgt_data_dir = os.path.join(args.data_root, "split1")  # ✅ target test fixed
    pl.seed_everything(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    pre_dir = os.path.join(args.save_dir, "pretrain")
    ft_dir = os.path.join(args.save_dir, "finetune")
    os.makedirs(pre_dir, exist_ok=True)
    os.makedirs(ft_dir, exist_ok=True)

    # -------- source dm (build vocab/norm from source TRAIN)
    dm_src = MimicEicuDataModule(
        data_dir=src_data_dir,
        source=args.source,
        split=args.split,
        n_timesteps=args.n_timesteps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        use_temp_cache=args.use_temp_cache,
        offset_unit=args.offset_unit,
        exclude_outcomes=("aki_label", "cf_label"),
    )
    dm_src.prepare_data()
    dm_src.setup()

    outcome_cols = dm_src.outcome_cols()
    d_target = dm_src.d_target()

    # -------- target dm (MANDATORY) but scale with SOURCE TRAIN stats (vocab+norm)
    dm_tgt = MimicEicuDataModule(
        data_dir=tgt_data_dir,
        source=args.target,
        split=1,
        n_timesteps=args.n_timesteps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        use_temp_cache=args.use_temp_cache,
        offset_unit=args.offset_unit,
        exclude_outcomes=("aki_label", "cf_label"),
        # ✅ inject source train vocab & normalizers
        itemid_to_idx=dm_src.itemid_to_idx,
        norm=dm_src.norm,
    )
    dm_tgt.prepare_data()
    dm_tgt.setup(stage="test")

    if dm_tgt.outcome_cols() != outcome_cols:
        raise ValueError("source/target outcome columns mismatch (name/order must be identical).")

    # -------- wandb
    logger = None
    wb = None
    if args.use_wandb:
        run_name = args.wandb_name if args.wandb_name else f"DuETT_{args.source}_{args.split}"  # ✅ correct

        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            log_model=False,
        )
        wb = logger.experiment
        wb.config.update(
            {
                "source": args.source,
                "split": args.split,
                "target": args.target,
                "n_timesteps": args.n_timesteps,
                "batch_size": args.batch_size,
                "d_time_series_num": dm_src.d_time_series_num(),
                "d_static_num": dm_src.d_static_num(),
                "d_target": d_target,
                "outcomes": outcome_cols,
                "exclude_outcomes": ["aki_label", "cf_label"],
                "seed": args.seed,
                "target_scaled_by": f"{args.source}_train_stats",
            },
            allow_val_change=True,
        )

    # -------- pretrain
    pre_model = pretrain_model(
        d_static_num=dm_src.d_static_num(),
        d_time_series_num=dm_src.d_time_series_num(),
        d_target=d_target,
        pos_frac=dm_src.pos_frac().tolist(),
        seed=args.seed,
        outcome_names=outcome_cols,
    )

    pre_ckpt = ModelCheckpoint(
        save_last=True,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=pre_dir,
        filename="pretrain-{epoch:03d}-{val_loss:.4f}",
    )
    warmup = WarmUpCallback(steps=2000)
    pre_early = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.pretrain_patience,
        min_delta=args.min_delta,
        verbose=True,
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[1],
        logger=logger,
        num_sanity_val_steps=2,
        max_epochs=args.pretrain_epochs,
        gradient_clip_val=1.0,
        callbacks=[warmup, pre_ckpt, pre_early],
    )
    trainer.fit(pre_model, dm_src)

    pretrained_path = pre_ckpt.best_model_path or pre_ckpt.last_model_path
    if not pretrained_path:
        raise RuntimeError("No pretrain checkpoint produced.")

    # -------- fine-tune
    best_paths = []
    ft_model = fine_tune_model(
        pretrained_path,
        d_static_num=dm_src.d_static_num(),
        d_time_series_num=dm_src.d_time_series_num(),
        d_target=d_target,
        pos_frac=dm_src.pos_frac().tolist(),
        seed=args.seed,
        outcome_names=outcome_cols,
    )

    ft_ckpt = ModelCheckpoint(
        save_top_k=1,             # single best
        save_last=True,           # keep last too (optional)
        mode="max",
        monitor="val_ap",         # macro AUPRC
        dirpath=ft_dir,
        filename=f"finetune-seed{args.seed}" + "-{epoch:03d}-{val_ap:.4f}",
    )

    ft_early = EarlyStopping(
        monitor="val_ap",         # macro AUPRC
        mode="max",
        patience=args.finetune_patience,
        min_delta=args.min_delta,
        verbose=True,
    )

    warmup = WarmUpCallback(steps=1000)

    ft_trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[1],
        logger=logger,
        max_epochs=args.finetune_epochs,
        gradient_clip_val=1.0,
        callbacks=[warmup, ft_ckpt, ft_early],
    )
    ft_trainer.fit(ft_model, dm_src)

    # load best finetuned checkpoint as final_model
    finetuned_path = ft_ckpt.best_model_path or ft_ckpt.last_model_path
    if not finetuned_path:
        raise RuntimeError("No finetune checkpoint produced.")

    final_model = fine_tune_model(
        finetuned_path,
        d_static_num=dm_src.d_static_num(),
        d_time_series_num=dm_src.d_time_series_num(),
        d_target=d_target,
        pos_frac=dm_src.pos_frac().tolist(),
        seed=args.seed,
        outcome_names=outcome_cols,
    )

    # -------- save final (explicit)
    final_path = os.path.join(args.save_dir, f"DuETT_{args.source}_{args.split}.pt")
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "hparams": dict(final_model.hparams),
            "outcome_cols": outcome_cols,
            "source": args.source,
            "target": args.target,
            "target_scaled_by": f"{args.source}_train_stats",
        },
        final_path,
    )

    # -------- evaluate source + target (target is mandatory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_test_loss, src_auroc, src_auprc, src_acc, src_auroc_per, src_auprc_per, src_valid_mask = evaluate_on_dm(
        final_model, dm_src, device
    )
    tgt_test_loss, tgt_auroc, tgt_auprc, tgt_acc, tgt_auroc_per, tgt_auprc_per, tgt_valid_mask = evaluate_on_dm(
        final_model, dm_tgt, device
    )

    if wb is not None:
        log_dict = {
            "test/source_loss": float(src_test_loss),
            "test/source_auroc": float(src_auroc),
            "test/source_auprc": float(src_auprc),
            "test/source_acc": float(src_acc),
            "test/target_loss": float(tgt_test_loss),
            "test/target_auroc": float(tgt_auroc),
            "test/target_auprc": float(tgt_auprc),
            "test/target_acc": float(tgt_acc),
            
        }

        for i, name in enumerate(outcome_cols):
            key = _safe_metric_key(name)
            if i < len(src_valid_mask) and bool(src_valid_mask[i]):
                log_dict[f"test/source_auroc_{key}"] = float(src_auroc_per[i])
                log_dict[f"test/source_auprc_{key}"] = float(src_auprc_per[i])
            if i < len(tgt_valid_mask) and bool(tgt_valid_mask[i]):
                log_dict[f"test/target_auroc_{key}"] = float(tgt_auroc_per[i])
                log_dict[f"test/target_auprc_{key}"] = float(tgt_auprc_per[i])

        wb.log(log_dict)
        wb.finish()


if __name__ == "__main__":
    main()
# python train_duett_multilabel.py --source mimic --target eicu