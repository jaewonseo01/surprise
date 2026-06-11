# train_transformer_multilabel.py
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from runner_raindrop import create_inputs_from_data, prepare_data, seed_everything
from utils_rd import (
    mean_imputation,
    metrics_multilabel_batched,
    tensorize_normalize_multilabel,
    getStats_fixed,
    getStats_static
)

# Conventions (must match runner/utils)
MISSING_VALUE = 0.0     # feature missing
MISSING_TIME_MIN = -1.0 # time padding in minutes
MISSING_TIME_HR = MISSING_TIME_MIN / 2880.0

def _aug_tag(
    use_source_aug: bool,
    source_aug_suffix: str,
    use_target_aug: bool,
    target_aug_suffix: str,
) -> str:
    src_tag = f"srcAug-{source_aug_suffix}" if use_source_aug else "srcOrig"
    tgt_tag = f"tgtAug-{target_aug_suffix}" if use_target_aug else "tgtOrig"
    return f"{src_tag}__{tgt_tag}"

def _compute_feature_means_from_raw(
    X_features: np.ndarray,   # [N,T,F], missing=0
    X_time: np.ndarray,       # [N,T,1] or [N,T], padding=-1
    missing_value_num: float = MISSING_VALUE,
    missing_time_num: float = MISSING_TIME_MIN,
) -> np.ndarray:
    """Feature means over observed entries only, within valid time window (before padding)."""
    if X_time.ndim == 3:
        times = X_time[:, :, 0]
    else:
        times = X_time

    N, T, F = X_features.shape
    means = np.zeros((F,), dtype=np.float32)

    for f in range(F):
        vals = []
        for i in range(N):
            pad_pos = np.where(times[i] == missing_time_num)[0]
            L = int(pad_pos[0]) if pad_pos.size > 0 else T

            xf = X_features[i, :L, f]
            vf = xf[xf != missing_value_num]
            if vf.size > 0:
                vals.append(vf.astype(np.float32))

        means[f] = float(np.concatenate(vals).mean()) if len(vals) else 0.0

    return means


def prepare_data_with_imputation_raw(
    processed_dir_source: str,
    processed_dir_target_test: str,
    source: str,
    target: str,
):
    """
    Returns RAW (numpy) dicts BEFORE normalization/tensorize.
    Applies mean imputation on RAW arr stage using SOURCE TRAIN means.
    Conventions:
      - values missing = 0.0
      - time padding   = -1.0 (minutes)
      - binary itemids are already converted to -1/+1 in runner preprocessing.
    """
    assert source in {"mimic", "eicu"} and target in {"mimic", "eicu"} and source != target

    def load_split_from(processed_dir: str, domain: str, split: str):
        P = np.load(os.path.join(processed_dir, f"PTdict_list_{domain}_{split}.npy"), allow_pickle=True)
        y = np.load(os.path.join(processed_dir, f"arr_outcomes_{domain}_{split}.npy"), allow_pickle=True).astype(np.float32)
        return P, y



    # source splits from split{i}
    Ptrain, ytrain = load_split_from(processed_dir_source, source, "train")
    Pval,   yval   = load_split_from(processed_dir_source, source, "valid")
    Ptest,  ytest  = load_split_from(processed_dir_source, source, "test")

    # target test ONLY from split1 (fixed)
    Ptest_tgt, ytest_tgt = load_split_from(processed_dir_target_test, target, "test")


    # ---- extract raw arrays ----
    def extract_raw(P_list, y_arr):
        N = len(P_list)
        T, F = P_list[0]["arr"].shape
        D = len(P_list[0]["extended_static"])

        X = np.stack([p["arr"] for p in P_list], axis=0).astype(np.float32)          # [N,T,F]
        TT = np.stack([p["time"] for p in P_list], axis=0).astype(np.float32)       # [N,T,1]
        S = np.stack([p["extended_static"] for p in P_list], axis=0).astype(np.float32)  # [N,D]
        Y = y_arr.astype(np.float32)                                                # [N,C]
        return X, TT, S, Y, T, F, D

    Xtr, TTtr, Str, Ytr, T, F, D = extract_raw(Ptrain, ytrain)
    Xva, TTva, Sva, Yva, _, _, _ = extract_raw(Pval,   yval)
    Xte, TTte, Ste, Yte, _, _, _ = extract_raw(Ptest,  ytest)

    Xtt, TTtt, Stt, Ytt, _, _, _ = extract_raw(Ptest_tgt, ytest_tgt)

    # ---- mean impute on RAW stage using SOURCE TRAIN means ----
    feat_means = _compute_feature_means_from_raw(
        Xtr, TTtr, missing_value_num=MISSING_VALUE, missing_time_num=MISSING_TIME_MIN
    )

    Xtr = mean_imputation(Xtr, TTtr, feat_means, missing_value_num=MISSING_VALUE, missing_time_num=MISSING_TIME_MIN)
    Xva = mean_imputation(Xva, TTva, feat_means, missing_value_num=MISSING_VALUE, missing_time_num=MISSING_TIME_MIN)
    Xte = mean_imputation(Xte, TTte, feat_means, missing_value_num=MISSING_VALUE, missing_time_num=MISSING_TIME_MIN)
    Xtt = mean_imputation(Xtt, TTtt, feat_means, missing_value_num=MISSING_VALUE, missing_time_num=MISSING_TIME_MIN)

    source_raw = {
        "Xtrain": Xtr, "Ttrain": TTtr, "Strain": Str, "Ytrain": Ytr,
        "Xval":   Xva, "Tval":   TTva, "Sval":   Sva, "Yval":   Yva,
        "Xtest":  Xte, "Ttest":  TTte, "Stest":  Ste, "Ytest":  Yte,
        "shape": {"T": T, "F": F, "D": D},
    }
    target_raw = {
        "Xtest": Xtt, "Ttest": TTtt, "Stest": Stt, "Ytest": Ytt,
        "shape": {"T": T, "F": F, "D": D},
    }
    return source_raw, target_raw


def tensorize_from_raw_dicts(
    processed_dir: str,
    source: str,
    source_raw: dict,
    target_raw: dict,
):
    """
    Convert RAW dicts -> normalized tensors, using SOURCE TRAIN stats only,
    and excluding categorical/binary features from normalization via categorical_idx_{source}.npy.
    """
    T = source_raw["shape"]["T"]
    F = source_raw["shape"]["F"]
    D = source_raw["shape"]["D"]

    # categorical indices (binary itemids) excluded from normalization
    cat_idx_path = os.path.join(processed_dir, f"categorical_idx_{source}.npy")
    cat_idx = np.load(cat_idx_path, allow_pickle=True).astype(np.int64)
    cat_idx = cat_idx[(cat_idx >= 0) & (cat_idx < F)]

    # stats on SOURCE TRAIN only (missing=0)
    mf, stdf = getStats_fixed(source_raw["Xtrain"], missing_value=MISSING_VALUE)  # [F], [F]
    ms, ss   = getStats_static(source_raw["Strain"], dataset="Default")           # [D], [D]

    mf = mf.copy(); stdf = stdf.copy()
    mf[cat_idx] = 0.0
    stdf[cat_idx] = 1.0

    # helper: pack back into PT-like list for tensorize_normalize_multilabel (expects list of dicts)
    def wrap_as_P_list(X, TT, S):
        N = X.shape[0]
        P_list = []
        for i in range(N):
            P_list.append({
                "arr": X[i],
                "time": TT[i],
                "extended_static": S[i],
            })
        return P_list

    def tz(X, TT, S, Y):
        P_list = wrap_as_P_list(X, TT, S)
        return tensorize_normalize_multilabel(
            P_list, Y, mf, stdf, ms, ss,
            missing_value=MISSING_VALUE,
            missing_time=MISSING_TIME_MIN,
        )

    # source
    Xtr, Str, TTtr, Ytr = tz(source_raw["Xtrain"], source_raw["Ttrain"], source_raw["Strain"], source_raw["Ytrain"])
    Xva, Sva, TTva, Yva = tz(source_raw["Xval"],   source_raw["Tval"],   source_raw["Sval"],   source_raw["Yval"])
    Xte, Ste, TTte, Yte = tz(source_raw["Xtest"],  source_raw["Ttest"],  source_raw["Stest"],  source_raw["Ytest"])

    # target test only
    Xtt, Stt, TTtt, Ytt = tz(target_raw["Xtest"], target_raw["Ttest"], target_raw["Stest"], target_raw["Ytest"])

    # permute: [N,T,2F] -> [T,N,2F], time [N,T,1] -> [T,N]
    def permute(X, TM):
        return X.permute(1, 0, 2), TM.squeeze(-1).permute(1, 0)

    Xtr, TTtr = permute(Xtr, TTtr)
    Xva, TTva = permute(Xva, TTva)
    Xte, TTte = permute(Xte, TTte)
    Xtt, TTtt = permute(Xtt, TTtt)

    C = Ytr.shape[1]
    empty_P  = torch.zeros((T, 0, Xtt.shape[2]), dtype=Xtt.dtype)
    empty_TM = torch.zeros((T, 0), dtype=TTtt.dtype)
    empty_S  = torch.zeros((0, D), dtype=Stt.dtype)
    empty_Y  = torch.zeros((0, C), dtype=Ytt.dtype)

    source_dict = {
        "Ptrain": Xtr, "Pval": Xva, "Ptest": Xte,
        "Ptrain_time": TTtr, "Pval_time": TTva, "Ptest_time": TTte,
        "Ptrain_static": Str, "Pval_static": Sva, "Ptest_static": Ste,
        "ytrain": Ytr, "yval": Yva, "ytest": Yte,
        "shape": {"T": T, "F": F, "D": D},
    }
    target_dict = {
        "Ptrain": empty_P, "Pval": empty_P, "Ptest": Xtt,
        "Ptrain_time": empty_TM, "Pval_time": empty_TM, "Ptest_time": TTtt,
        "Ptrain_static": empty_S, "Pval_static": empty_S, "Ptest_static": Stt,
        "ytrain": empty_Y, "yval": empty_Y, "ytest": Ytt,
        "shape": {"T": T, "F": F, "D": D},
    }

    outcome_cols = np.load(os.path.join(processed_dir, f"outcome_cols_{source}.npy"), allow_pickle=True).tolist()
    source_dict["outcome_cols"] = outcome_cols
    target_dict["outcome_cols"] = outcome_cols
    return source_dict, target_dict


class PositionalEncodingTF(nn.Module):
    """
    Time-based sin/cos encoding (same spirit as original).
    - Uses P_time in HOURS with padding=-1/2880.
    - Output on same device as input.
    """
    def __init__(self, d_model: int, max_len: int = 800, MAX: int = 10000):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time: torch.Tensor) -> torch.Tensor:
        # P_time: [T,B] in hours (padding=-1/2880)
        device = P_time.device
        P_time = P_time.float()

        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)
        timescales = torch.tensor(timescales, dtype=torch.float32, device=device)  # [d/2]

        times = P_time.unsqueeze(2)  # [T,B,1]
        scaled_time = times / timescales[None, None, :]  # [T,B,d/2]

        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)  # [T,B,d]
        return pe

    def forward(self, P_time: torch.Tensor) -> torch.Tensor:
        return self.getPE(P_time)


class TransformerModel2(nn.Module):
    """
    Original baseline-style Transformer, but fixed to:
      - be device-safe (no hardcoded .cuda())
      - build correct src_key_padding_mask shape [B, T] (bool)
      - not break when src is already [T,B,F] (not [T,B,2F])
      - avoid the buggy squeeze(1) mask logic
      - safe mean aggregation denominator (no "+1" bias)

    NOTE: This model still intentionally DROPS the mask channel if you pass [T,B,2F]
          (baseline behavior). You said you accept that.
    """
    def __init__(
        self,
        d_inp, d_model, nhead, nhid, nlayers, dropout,
        max_len, d_static, MAX, aggreg, n_classes,
        static=True
    ):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.model_type = "Transformer"

        d_pe = 16
        d_enc = d_inp

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        encoder_layers = TransformerEncoderLayer(d_pe + d_enc, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.encoder = nn.Linear(d_inp, d_enc)

        self.static = static
        if self.static:
            self.emb = nn.Linear(d_static, d_inp)

        d_fi = (d_enc + d_pe) + (d_inp if self.static else 0)

        self.mlp = nn.Sequential(
            nn.Linear(d_fi, d_fi),
            nn.ReLU(),
            nn.Linear(d_fi, n_classes),
        )

        self.aggreg = aggreg
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, static, times, lengths):
        """
        src:    [T,B,2F] (values+mask) OR [T,B,F]
        times:  [T,B]  (hours; padding = -1/2880)
        lengths:[B]    (#valid timesteps, >=1)
        """
        device = src.device
        T, B, Fin = src.shape

        # baseline behavior: if input looks like [T,B,2F], drop mask half
        if Fin % 2 == 0:
            # If you *always* pass 2F this is fine.
            # If you sometimes pass F where F is even, this would wrongly halve.
            # So guard: only halve if it matches expected 2*d_inp.
            if hasattr(self.encoder, "in_features") and Fin == 2 * self.encoder.in_features:
                src = src[:, :, : Fin // 2]

        # now src should be [T,B,d_inp]
        src = self.encoder(src)                # [T,B,d_enc]
        pe = self.pos_encoder(times).to(device)  # [T,B,d_pe]
        x = torch.cat([pe, src], dim=2)        # [T,B,d_pe+d_enc]
        x = self.dropout(x)

        emb = None
        if static is not None:
            emb = self.emb(static)             # [B,d_inp]

        # src_key_padding_mask: [B,T] where True means PAD
        lengths = lengths.to(device).clamp(min=1)
        pad_mask = (torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1))  # [B,T] bool

        out = self.transformer_encoder(x, src_key_padding_mask=pad_mask)  # [T,B,H]

        # aggregation with valid mask
        valid = (~pad_mask).transpose(0, 1).unsqueeze(2).to(out.dtype)  # [T,B,1]
        if self.aggreg == "mean":
            denom = lengths.unsqueeze(1).to(out.dtype)  # [B,1]
            out = (out * valid).sum(dim=0) / denom      # [B,H]
        elif self.aggreg == "max":
            out = out.masked_fill(~valid.bool(), float("-inf"))
            out, _ = out.max(dim=0)                     # [B,H]
            out = torch.nan_to_num(out, neginf=0.0)      # if a sequence is fully padded (shouldn't happen)
        else:
            raise ValueError(f"Unknown aggreg={self.aggreg}")

        if emb is not None:
            out = torch.cat([out, emb], dim=1)           # [B, H + d_inp]

        logits = self.mlp(out)                           # [B,C]
        return logits


def run_train_eval(
    source: str,
    target: str,
    split: int = 1,
    data_root: str = "./data",                 # ✅ base folder that contains split{i}/
    processed_root: str = "./data_rd_template",# ✅ root to store per-split processed files
    model_name: str = "TransformerModel2",
    use_wandb: bool = False,
    wandb_project: str = "Baselines",
    wandb_entity: str | None = None,
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_len: int = 800,
    d_static: int = 3,
    d_model: int = 36,
    nhead: int = 1,
    nlayers: int = 1,
    dropout: float = 0.2,
    MAX: int = 100,
    aggreg: str = "mean",
    imputation: str = "no_imputation",  # "no_imputation" | "mean"
    seed: int = 9871,
    use_source_aug: bool = False,
    source_aug_suffix: str = "aug",
    use_target_aug: bool = False,
    target_aug_suffix: str = "aug",
):
    
    assert imputation in {"no_imputation", "mean"}

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ build (or reuse) per-split processed inputs for both domains
    aug_tag = _aug_tag(
    use_source_aug, source_aug_suffix,
    use_target_aug, target_aug_suffix,
    )
    processed_root_run = os.path.join(processed_root, aug_tag)
    os.makedirs(processed_root_run, exist_ok=True)

    create_inputs_from_data(
        domain=source,
        split=split,
        data_dir=data_root,
        processed_root=processed_root_run,
        use_aug=use_source_aug,
        aug_suffix=source_aug_suffix,
    )

    create_inputs_from_data(
        domain=target,
        split=1,
        data_dir=data_root,
        processed_root=processed_root_run,
        use_aug=use_target_aug,
        aug_suffix=target_aug_suffix,
    )

    if imputation == "mean":
        src_aug_tag = source_aug_suffix if use_source_aug else "orig"
        tgt_aug_tag = target_aug_suffix if use_target_aug else "orig"

        processed_dir_source = os.path.join(
            processed_root_run, f"split{split}", f"{source}_{src_aug_tag}"
        )
        processed_dir_target = os.path.join(
            processed_root_run, "split1", f"{target}_{tgt_aug_tag}"
        )

        source_raw, target_raw = prepare_data_with_imputation_raw(
            processed_dir_source=processed_dir_source,
            processed_dir_target_test=processed_dir_target,
            source=source,
            target=target,
        )

        source_dict, target_dict = tensorize_from_raw_dicts(
            processed_dir=processed_dir_source,
            source=source,
            source_raw=source_raw,
            target_raw=target_raw,
        )
    else:
        source_dict, target_dict = prepare_data(
            processed_root=processed_root_run,
            split_source=split,
            split_target_test=1,
            source=source,
            target=target,
            use_source_aug=use_source_aug,
            source_aug_suffix=source_aug_suffix,
            use_target_aug=use_target_aug,
            target_aug_suffix=target_aug_suffix,
        )

    Xtr = source_dict["Ptrain"]         # [T,N,2F]
    Ttr = source_dict["Ptrain_time"]    # [T,N] hours
    Str = source_dict["Ptrain_static"]  # [N,D]
    Ytr = source_dict["ytrain"].float() # [N,C]

    Xva = source_dict["Pval"]
    Tva = source_dict["Pval_time"]
    Sva = source_dict["Pval_static"]
    Yva = source_dict["yval"].float()

    Xte_s = source_dict["Ptest"]
    Tte_s = source_dict["Ptest_time"]
    Ste_s = source_dict["Ptest_static"]
    Yte_s = source_dict["ytest"].float()

    Xte_t = target_dict["Ptest"]
    Tte_t = target_dict["Ptest_time"]
    Ste_t = target_dict["Ptest_static"]
    Yte_t = target_dict["ytest"].float()

    outcome_cols = source_dict.get("outcome_cols", [f"task_{i}" for i in range(Ytr.shape[1])])


    # move to device
    Xtr, Ttr, Str, Ytr = Xtr.to(device), Ttr.to(device), Str.to(device), Ytr.to(device)
    Xva, Tva, Sva, Yva = Xva.to(device), Tva.to(device), Sva.to(device), Yva.to(device)
    Xte_s, Tte_s, Ste_s, Yte_s = Xte_s.to(device), Tte_s.to(device), Ste_s.to(device), Yte_s.to(device)
    Xte_t, Tte_t, Ste_t, Yte_t = Xte_t.to(device), Tte_t.to(device), Ste_t.to(device), Yte_t.to(device)

    # infer dims
    d_inp = Xtr.shape[2]     # ✅ 2F (value+mask)
    C = Ytr.shape[1]

    model = TransformerModel2(
        d_inp=d_inp,
        d_model=d_model,
        nhead=nhead,
        nhid=2 * d_model,
        nlayers=nlayers,
        dropout=dropout,
        max_len=max_len,
        d_static=d_static,
        MAX=MAX,
        aggreg=aggreg,
        n_classes=C,
        static=True,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # wandb
    wb = None
    run_name = f"{model_name}_{source}_to_{target}_split{split}__{aug_tag}"
    if use_wandb:
        import wandb
        wb = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config={
                "model": model_name,
                "source": source,
                "split":split,
                "target": target,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "max_len": max_len,
                "d_model": d_model,
                "nhead": nhead,
                "nlayers": nlayers,
                "dropout": dropout,
                "MAX": MAX,
                "aggreg": aggreg,
                "use_source_aug": use_source_aug,
                "source_aug_suffix": source_aug_suffix,
                "use_target_aug": use_target_aug,
                "target_aug_suffix": target_aug_suffix,
                "aug_tag": aug_tag,
            },
        )
        wandb.watch(model, log="gradients", log_freq=100)

    def lengths_from_times(hours_TN: torch.Tensor) -> torch.Tensor:
        # hours_TN: [T,N], padding=-1/2880
        return (hours_TN != MISSING_TIME_HR).sum(dim=0).clamp(min=1)

    patience = 7
    no_improve_epochs = 0
    best_val_auprc = -1.0
    best_state = None

    Ntr = Xtr.shape[1]
    steps = max(1, int(np.ceil(Ntr / batch_size)))

    t0 = time.time()
    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(Ntr, device=device)

        loss_sum = 0.0
        pbar = tqdm(range(steps), desc=f"Epoch {epoch:02d}", leave=False)
        for s in pbar:
            idx = perm[s * batch_size:(s + 1) * batch_size]
            if idx.numel() == 0:
                continue

            Xb = Xtr[:, idx, :]
            Tb = Ttr[:, idx]
            Sb = Str[idx] if Str is not None else None
            Yb = Ytr[idx]

            lengths = lengths_from_times(Tb)
            logits = model(Xb, Sb, Tb, lengths)
            loss = criterion(logits, Yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        train_loss = loss_sum / max(1, steps)

        val_loss, val_auroc, val_auprc, val_acc, val_valid, val_auroc_per, val_auprc_per, val_valid_mask = metrics_multilabel_batched(
            model, Xva, Tva, Sva, Yva,
            criterion=criterion,
            batch_size=batch_size,
            device=device,
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"train loss={train_loss:.4f} | "
            f"val loss={val_loss:.4f} AUROC={val_auroc:.4f} AUPRC={val_auprc:.4f} ACC={val_acc:.4f} "
            f"(valid_labels={val_valid})"
        )

        if wb is not None:
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/auroc": val_auroc,
                "val/auprc": val_auprc,
                "val/acc": val_acc,
                "val/valid_labels": val_valid,
            }

            # per-label logging with column names
            for i, name in enumerate(outcome_cols):
                if i < len(val_valid_mask) and val_valid_mask[i]:
                    safe_name = str(name).replace(" ", "_")
                    log_dict[f"val/auroc_{safe_name}"] = float(val_auroc_per[i])
                    log_dict[f"val/auprc_{safe_name}"] = float(val_auprc_per[i])

            wb.log(log_dict)

        if np.isfinite(val_auprc) and val_auprc > best_val_auprc:
            best_val_auprc = float(val_auprc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"🛑 Early stopping at epoch {epoch:02d} (best val AUPRC={best_val_auprc:.4f})")
            if wb is not None:
                wb.log({"early_stop_epoch": epoch})
            break

    elapsed_min = (time.time() - t0) / 60.0
    print(f"Done. elapsed={elapsed_min:.2f} min | best src-val AUPRC={best_val_auprc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    save_dir = "./baseline_models_rd"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{run_name}.pt")

    ckpt = {
        "run_name": run_name,
        "best_val_auprc": best_val_auprc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "source": source,
            "target": target,
            "split": split,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "aggreg": aggreg,
            "imputation": imputation,
            "use_source_aug": use_source_aug,
            "source_aug_suffix": source_aug_suffix,
            "use_target_aug": use_target_aug,
            "target_aug_suffix": target_aug_suffix,
            "aug_tag": aug_tag,
        },
    }

    torch.save(ckpt, save_path)
    print(f"✅ Saved trained model to {save_path}")

    # final tests
    src_test_loss, src_auroc, src_auprc, src_acc, src_valid, src_auroc_per, src_auprc_per, src_valid_mask = metrics_multilabel_batched(
        model, Xte_s, Tte_s, Ste_s, Yte_s, criterion, batch_size=batch_size, device=device
    )
    tgt_test_loss, tgt_auroc, tgt_auprc, tgt_acc, tgt_valid, tgt_auroc_per, tgt_auprc_per, tgt_valid_mask = metrics_multilabel_batched(
        model, Xte_t, Tte_t, Ste_t, Yte_t, criterion, batch_size=batch_size, device=device
    )

    if wb is not None:
        log_dict = {
            "test/source_auroc": src_auroc,
            "test/source_auprc": src_auprc,
            "test/source_acc": src_acc,
            "test/target_auroc": tgt_auroc,
            "test/target_auprc": tgt_auprc,
            "test/target_acc": tgt_acc,
        }
        for i, name in enumerate(outcome_cols):
            safe_name = str(name).replace(" ", "_")
            if i < len(src_valid_mask) and src_valid_mask[i]:
                log_dict[f"test/source_auroc_{safe_name}"] = float(src_auroc_per[i])
                log_dict[f"test/source_auprc_{safe_name}"] = float(src_auprc_per[i])
            if i < len(tgt_valid_mask) and tgt_valid_mask[i]:
                log_dict[f"test/target_auroc_{safe_name}"] = float(tgt_auroc_per[i])
                log_dict[f"test/target_auprc_{safe_name}"] = float(tgt_auprc_per[i])

        wb.log(log_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--source",
    type=str,
    default="mimic",
    choices=["mimic", "eicu"],
    help="Source domain (target will be the other one)"
    )

    parser.add_argument("--split", type=int, default=1, choices=[1,2,3,4,5])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--processed_root", type=str, default="./data_rd_template")
    parser.add_argument("--model_name", type=str, default="Transformer")

    parser.add_argument("--max_len", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--imputation", type=str, default="no_imputation", choices=["no_imputation", "mean"])

    parser.add_argument("--wandb", action="store_false")
    parser.add_argument("--wandb_project", type=str, default="Baselines")
    parser.add_argument("--wandb_entity", type=str, default="jwseo118-korea-university")

    # model knobs
    parser.add_argument("--d_model", type=int, default=36)
    parser.add_argument("--nhead", type=int, default=1)
    parser.add_argument("--nlayers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--aggreg", type=str, default="mean", choices=["mean", "max"])

    parser.add_argument("--use_source_aug", action="store_true")
    parser.add_argument("--source_aug_suffix", type=str, default="aug")
    parser.add_argument("--use_target_aug", action="store_true")
    parser.add_argument("--target_aug_suffix", type=str, default="aug")

    args = parser.parse_args()
    
    if args.source == "mimic":
        args.target = "eicu"
    else:
        args.target = "mimic"

    if args.source == args.target:
        raise ValueError("source and target must be different")

    run_train_eval(
        source=args.source,
        target=args.target,
        split=args.split,
        data_root=args.data_root,
        processed_root=args.processed_root,
        model_name=args.model_name,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        num_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        max_len=args.max_len,
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        dropout=args.dropout,
        aggreg=args.aggreg,
        imputation=args.imputation,
        use_source_aug=args.use_source_aug,
        source_aug_suffix=args.source_aug_suffix,
        use_target_aug=args.use_target_aug,
        target_aug_suffix=args.target_aug_suffix,
    )

# python train_transformer_multilabel.py --model_name Transformer --imputation no_imputation 
# && python train_transformer_multilabel.py --model_name Transmean --imputation mean
if __name__ == "__main__":
    main()
