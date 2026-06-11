# utils_rd.py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# Stats
# ============================================================
def getStats_fixed(P_tensor: np.ndarray, missing_value: float = 0.0):
    """
    Compute per-feature mean/std, treating `missing_value` as missing.

    P_tensor: [N, T, F]
    Returns:
        mf:   [F] float32
        stdf: [F] float32
    """
    assert P_tensor.ndim == 3, f"Expected [N,T,F], got {P_tensor.shape}"
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)

    mf = np.zeros((F,), dtype=np.float32)
    stdf = np.ones((F,), dtype=np.float32)
    eps = 1e-7

    for f in range(F):
        vals = Pf[f]
        vals = vals[vals != missing_value]  # ✅ only drop missing (=0)

        if vals.size == 0:
            mf[f] = 0.0
            stdf[f] = 1.0
            continue

        unique_vals = np.unique(vals)

        # Degenerate/binary-like safeguards
        # - single unique -> can't normalize well
        # - {-1,1} or {0,1} patterns: keep identity
        if (
            unique_vals.size == 1
            or (unique_vals.size == 2 and set(unique_vals.tolist()) <= {-1.0, 1.0})
            or (unique_vals.size == 2 and set(unique_vals.tolist()) <= {0.0, 1.0})
        ):
            mf[f] = 0.0
            stdf[f] = 1.0
        else:
            m = float(np.mean(vals))
            s = float(np.std(vals))
            mf[f] = m
            stdf[f] = max(s, eps)

    return mf, stdf


def getStats_static(P_tensor: np.ndarray, dataset: str = "Default"):
    """
    Static stats. In your setting: no missing in static.
    Returns ms, ss as [S] float32.
    """
    assert P_tensor.ndim == 2
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))

    ms = np.zeros((S,), dtype=np.float32)
    ss = np.ones((S,), dtype=np.float32)

    if dataset == "P12":
        bool_categorical = [0, 1, 1, 0, 1, 1, 1, 1, 0]
    elif dataset == "P19":
        bool_categorical = [0, 1, 0, 0, 0, 0]
    elif dataset == "eICU":
        bool_categorical = [1] * 397 + [0] * 2
    else:  # Default (your data)
        # ['Age', 'Gender', 'Height'] -> treat Gender as categorical
        bool_categorical = [0, 1, 0]

    for s in range(S):
        if bool_categorical[s] == 0:
            vals = Ps[s]
            ms[s] = float(np.mean(vals))
            ss[s] = float(np.std(vals) if np.std(vals) > 1e-7 else 1.0)

    return ms, ss


# ============================================================
# Tensorize + Normalize (multilabel)
# ============================================================
def tensorize_normalize_multilabel(
    P,
    y,
    mf,
    stdf,
    ms,
    ss,
    missing_value: float = 0.0,   # ✅ values missing=0
    missing_time: float = -1.0,   # ✅ time padding=-1 (minutes)
):
    """
    Outputs:
      X2: [N,T,2F]  (normalized values w/ missing filled to 0, mask)
      Sn: [N,D]
      TT_hours: [N,T,1] (mins->hours, missing_time stays -1/2880)
      y_tensor: [N,C] float32
    """
    T, F = P[0]["arr"].shape
    D = len(P[0]["extended_static"])
    N = len(P)

    X = np.zeros((N, T, F), dtype=np.float32)
    TT = np.zeros((N, T, 1), dtype=np.float32)
    S = np.zeros((N, D), dtype=np.float32)

    for i in range(N):
        X[i] = P[i]["arr"]     # contains 0 for missing values
        TT[i] = P[i]["time"]   # contains -1 for padding
        S[i] = P[i]["extended_static"]

    # mask: observed iff value != 0
    M = (X != missing_value).astype(np.float32)  # [N,T,F]

    mf_ = np.asarray(mf, dtype=np.float32).reshape(1, 1, -1)      # [1,1,F]
    stdf_ = np.asarray(stdf, dtype=np.float32).reshape(1, 1, -1)  # [1,1,F]

    # fill missing with 0 in value channel (already 0, but be explicit)
    Xn = X.copy()
    Xn[Xn == missing_value] = 0.0

    # normalize
    Xz = (Xn - mf_) / (stdf_ + 1e-18)
    Xz[M == 0] = 0.0  # keep missing as 0 in value channel

    # concat normalized values + mask
    X2 = np.concatenate([Xz, M], axis=2)  # [N,T,2F]

    # time: minutes->hours (missing stays -1/60)
    # Original code is dividing by 60, but we divide by 2880 to normalize time 0-1
    TT_hours = TT / 2880.0

    # static normalize (no missing assumed)
    ms_ = np.asarray(ms, dtype=np.float32).reshape(1, -1)
    ss_ = np.asarray(ss, dtype=np.float32).reshape(1, -1)
    Sn = (S - ms_) / (ss_ + 1e-18)

    y_tensor = torch.tensor(y, dtype=torch.float32)

    return (
        torch.tensor(X2, dtype=torch.float32),
        torch.tensor(Sn, dtype=torch.float32),
        torch.tensor(TT_hours, dtype=torch.float32),
        y_tensor,
    )


# ============================================================
# Mean imputation (optional)
# ============================================================
def mean_imputation(
    X_features: np.ndarray,
    X_time: np.ndarray,
    mean_features: np.ndarray,
    missing_value_num: float = 0.0,   # ✅ values missing
    missing_time_num: float = -1.0,   # ✅ time padding
):
    """
    Impute missing values in X_features with mean_features, within valid time range inferred from X_time.
    """
    if X_time.ndim == 3:
        times_all = X_time[:, :, 0]
    else:
        times_all = X_time

    N, T, F = X_features.shape

    for i in range(N):
        times = times_all[i]
        pad_pos = np.where(times == missing_time_num)[0]
        L = int(pad_pos[0]) if pad_pos.size > 0 else T

        xf = X_features[i, :L, :]
        missing_idx = np.where(xf == missing_value_num)
        for row, col in zip(*missing_idx):
            X_features[i, row, col] = mean_features[col]

    return X_features


# ============================================================
# Metrics / evaluation
# ============================================================
def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray):
    C = y_true.shape[1]
    aucs, aps = [], []
    for c in range(C):
        yt = y_true[:, c]
        yp = y_prob[:, c]
        if np.unique(yt).size < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
        aps.append(average_precision_score(yt, yp))
    auroc = float(np.mean(aucs)) if len(aucs) else float("nan")
    auprc = float(np.mean(aps)) if len(aps) else float("nan")
    return auroc, auprc, len(aucs)

def multilabel_metrics_per_label(y_true: np.ndarray, y_prob: np.ndarray):
    C = y_true.shape[1]
    auroc = np.full((C,), np.nan, dtype=np.float32)
    auprc = np.full((C,), np.nan, dtype=np.float32)
    valid = np.zeros((C,), dtype=bool)

    for c in range(C):
        yt = y_true[:, c]
        yp = y_prob[:, c]
        if np.unique(yt).size < 2:
            continue
        valid[c] = True
        auroc[c] = roc_auc_score(yt, yp)
        auprc[c] = average_precision_score(yt, yp)

    return auroc, auprc, valid

def multilabel_acc(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5):
    pred = (y_prob >= thresh).astype(np.float32)
    return float((pred == y_true.astype(np.float32)).mean())


def evaluate_multilabel(
    model,
    P_tensor,          # [T, N, F]
    P_time_tensor,     # [T, N]
    P_static_tensor,   # [N, D] or None
    batch_size: int = 128,
    device: torch.device | str = "cuda",
):
    model.eval()
    device = torch.device(device)

    T, N, _ = P_tensor.shape
    out_chunks = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            P = P_tensor[:, start:end, :].to(device, non_blocking=True)
            Ptime = P_time_tensor[:, start:end].to(device, non_blocking=True)

            Pstatic = None if P_static_tensor is None else P_static_tensor[start:end].to(device, non_blocking=True)

            missing_time_hours = -1.0 / 2880.0
            lengths = torch.sum(Ptime != missing_time_hours, dim=0).clamp(min=1)

            logits = model(P, Pstatic, Ptime, lengths)   # [B, C]
            out_chunks.append(logits.detach().cpu())

    return torch.cat(out_chunks, dim=0)


def metrics_multilabel_batched(
    model,
    X, TT, SS, YY,
    criterion,
    batch_size: int = 128,
    device: torch.device | str = "cuda",
):
    model.eval()
    device = torch.device(device)

    _, N, _ = X.shape
    losses = []
    probs_all = []
    y_all = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            P = X[:, start:end, :].to(device, non_blocking=True)
            Ptime = TT[:, start:end].to(device, non_blocking=True)
            Yb = YY[start:end].to(device, non_blocking=True)

            Pstatic = None if SS is None else SS[start:end].to(device, non_blocking=True)

            missing_time_hours = -1.0 / 2880.0
            lengths = torch.sum(Ptime != missing_time_hours, dim=0).clamp(min=1)

            logits = model(P, Pstatic, Ptime, lengths)

            loss = criterion(logits, Yb)
            losses.append(loss.item())

            probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
            y_all.append(Yb.detach().cpu().numpy())

    probs_all = np.concatenate(probs_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    auroc_macro, auprc_macro, n_valid = multilabel_metrics(y_all, probs_all)
    acc = multilabel_acc(y_all, probs_all, thresh=0.5)

    auroc_per, auprc_per, valid_mask = multilabel_metrics_per_label(y_all, probs_all)

    return (
        float(np.mean(losses)),
        auroc_macro,
        auprc_macro,
        acc,
        n_valid,
        auroc_per,
        auprc_per,
        valid_mask,
    )

def masked_softmax(A, epsilon=0.000000001):
    A_max = torch.max(A, dim=1, keepdim=True)[0]
    A_exp = torch.exp(A - A_max)
    A_exp = A_exp * (A != 0).float()
    A_softmax = A_exp / (torch.sum(A_exp, dim=0, keepdim=True) + epsilon)
    return A_softmax