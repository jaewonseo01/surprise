from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def collect_embeddings_df(lit, loader, *, source_tag: str) -> pd.DataFrame:
    device = lit.device
    lit.eval()

    rows = []
    sel_names = [lit.outcome_cols[i] for i in lit.target_indices]

    for batch in loader:
        batch = tuple(x.to(device) if torch.is_tensor(x) else x for x in batch)
        times, varis, values, pad_mask, _pre, y, pid, static = batch

        out = lit.model(
            times=times,
            varis=varis,
            values=values,
            statics=static,
            padding_mask=pad_mask,
            pretrain=False,
        )

        E = out["embs"].detach().cpu().numpy()  # [B, D]
        P = pid.detach().cpu().numpy().astype(int)

        Y = y[:, lit.target_indices].detach().cpu().numpy()
        if Y.ndim == 1:
            Y = Y[:, None]

        pred = out["pred"].detach().cpu().numpy()
        if pred.ndim == 1:
            pred = pred[:, None]

        df_e = pd.DataFrame(E, columns=[f"e{i}" for i in range(E.shape[1])])
        df_e.insert(0, "pid", P)
        df_e["source"] = source_tag

        for j, name in enumerate(sel_names):
            df_e[name] = Y[:, j]
            df_e[f"pred_{name}"] = pred[:, j]

        rows.append(df_e)

    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()


@torch.no_grad()
def save_check_padding_df(lit, loader, *, tag: str, out_dir: str, run_name: str):
    if not hasattr(lit.model, "check_padding"):
        return

    device = lit.device
    lit.eval()
    rows = []

    for batch in loader:
        times, varis, values, pad_mask, _pre, _y, pid, _static = [
            x.to(device) if torch.is_tensor(x) else x for x in batch
        ]
        cp = lit.model.check_padding(times=times, varis=varis, values=values, padding_mask=pad_mask)

        times_np = cp["times"].detach().cpu().numpy()
        varis_np = cp["varis"].detach().cpu().numpy()
        values_np = cp["values"].detach().cpu().numpy()
        mask_np = cp["mask"].detach().cpu().numpy()
        pad_np = pad_mask.detach().cpu().numpy()
        pid_np = pid.detach().cpu().numpy()

        B, L = times_np.shape
        for b in range(B):
            valid_idx = np.nonzero(~pad_np[b])[0]
            if valid_idx.size == 0:
                continue
            p = int(pid_np[b])
            for t in valid_idx:
                rows.append(
                    dict(
                        pid=p,
                        times=float(times_np[b, t]),
                        varis=int(varis_np[b, t]),
                        values=float(values_np[b, t]),
                        mask=bool(mask_np[b, t]),
                    )
                )

    if not rows:
        return

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{run_name}_mask_{tag}.feather")
    pd.DataFrame(rows).to_feather(path)
