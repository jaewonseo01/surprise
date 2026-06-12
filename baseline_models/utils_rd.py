# utils_rd.py
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_outcome_index_map(outcome_cols):
    return {name: idx for idx, name in enumerate(outcome_cols)}


def _safe_metric_key(name: str) -> str:
    return str(name).strip().replace(" ", "_").replace("/", "_")


def mean_imputation(X, T, feat_means, missing_value=0.0, missing_time=-1.0):
    X_imputed = X.copy()
    mask = (T != missing_time)
    for f in range(X.shape[2]):
        feature_vals = X[:, :, f]
        missing = (feature_vals == missing_value) & mask
        X_imputed[missing, f] = feat_means[f]
    return X_imputed


def getStats_fixed(X, missing_value=0.0):
    mask = X != missing_value
    means = X[mask].mean(axis=0)
    stds = X[mask].std(axis=0)
    return means, stds


def getStats_static(X, dataset="Default"):
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    return means, stds


def tensorize_normalize_multilabel(P_list, Y, mf, stdf, ms, ss, missing_value=0.0, missing_time=-1.0):
    N = len(P_list)
    T = P_list[0]["arr"].shape[0]
    F = P_list[0]["arr"].shape[1]

    X = np.zeros((N, T, F * 2), dtype=np.float32)
    TM = np.zeros((N, T, 1), dtype=np.float32)
    S = np.zeros((N, len(P_list[0]["extended_static"])), dtype=np.float32)

    for i, p in enumerate(P_list):
        arr = p["arr"].astype(np.float32)
        time = p["time"].astype(np.float32)
        ext_static = p["extended_static"].astype(np.float32)

        arr_norm = (arr - mf) / (stdf + 1e-6)
        arr_norm[np.isnan(arr_norm)] = 0.0
        X[i, :, :F] = arr_norm
        X[i, :, F:] = (arr == missing_value).astype(np.float32)
        TM[i, :, 0] = time
        S[i] = (ext_static - ms) / (ss + 1e-6)

    X = torch.from_numpy(X)
    TM = torch.from_numpy(TM)
    S = torch.from_numpy(S)
    Y = torch.from_numpy(Y).float()
    return X, S, TM, Y


def flatten_numpy(arr):
    return arr.reshape(-1)


def metrics_multilabel_batched(model, X, T, S, Y, criterion, batch_size=32, device=None):
    model.eval()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = X.shape[1]
    steps = max(1, int(np.ceil(N / batch_size)))

    total_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for s in range(steps):
            idx = slice(s * batch_size, (s + 1) * batch_size)
            Xb = X[:, idx, :].to(device)
            Tb = T[:, idx].to(device)
            Sb = S[idx].to(device) if S is not None else None
            Yb = Y[idx].to(device)

            lengths = (Tb != MISSING_TIME_HR).sum(dim=0).clamp(min=1)
            logits = model(Xb, Sb, Tb, lengths)
            loss = criterion(logits, Yb)
            total_loss += loss.item() * Yb.size(0)

            all_logits.append(logits.cpu())
            all_targets.append(Yb.cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    avg_loss = total_loss / float(N)
    auroc = np.nan
    auprc = np.nan
    acc = np.nan
    valid = np.nan
    auroc_per = np.zeros(all_targets.shape[1], dtype=np.float32)
    auprc_per = np.zeros(all_targets.shape[1], dtype=np.float32)
    valid_mask = np.zeros(all_targets.shape[1], dtype=np.bool_)

    try:
        for i in range(all_targets.shape[1]):
            y_true = all_targets[:, i]
            y_score = all_logits[:, i]
            if np.any(y_true == 1) and np.any(y_true == 0):
                auroc_i = roc_auc_score(y_true, y_score)
                auprc_i = average_precision_score(y_true, y_score)
                auroc_per[i] = auroc_i
                auprc_per[i] = auprc_i
                valid_mask[i] = True

        if valid_mask.any():
            auroc = float(np.nanmean(auroc_per[valid_mask]))
            auprc = float(np.nanmean(auprc_per[valid_mask]))
            acc = float((all_logits > 0).astype(np.float32).round() == all_targets).mean()
            valid = int(valid_mask.sum())
        else:
            auroc = float("nan")
            auprc = float("nan")
            acc = float((all_logits > 0).astype(np.float32).round() == all_targets).mean()
            valid = 0
    except Exception:
        pass

    return avg_loss, auroc, auprc, acc, valid, auroc_per, auprc_per, valid_mask
