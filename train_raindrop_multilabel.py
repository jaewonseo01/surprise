# train_raindrop_multilabel.py
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Ob_propagation import Observation_progation
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform, glorot, zeros, ones, reset

from runner_raindrop import create_inputs_from_data, prepare_data, seed_everything
from utils_rd import metrics_multilabel_batched

# ------------------------------------------------------------
# Conventions (must match runner/utils)
# ------------------------------------------------------------
MISSING_VALUE = 0.0
MISSING_TIME_MIN = -1.0
MISSING_TIME_HR = MISSING_TIME_MIN / 2880.0  # -1/2880 hours


# ------------------------------------------------------------
# NOTE: import your Raindrop baseline model here
# - This file assumes you already have Raindrop_v2 available.
# - If your Raindrop model class lives elsewhere, change the import accordingly.
# ------------------------------------------------------------

def _aug_tag(
    use_source_aug: bool,
    source_aug_suffix: str,
    use_target_aug: bool,
    target_aug_suffix: str,
) -> str:
    src_tag = f"srcAug-{source_aug_suffix}" if use_source_aug else "srcOrig"
    tgt_tag = f"tgtAug-{target_aug_suffix}" if use_target_aug else "tgtOrig"
    return f"{src_tag}__{tgt_tag}"
class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1]

        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)

        times = torch.Tensor(P_time.cpu()).unsqueeze(2)
        scaled_time = times / torch.Tensor(timescales[None, None, :])
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1)  # T x B x d_model
        pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        pe = pe.cuda()
        return pe



class Raindrop_v2(nn.Module):
    """Implement the raindrop stratey one by one."""
    """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
    Inputs:
        d_inp = number of input features
        d_model = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        dropout = dropout rate (default 0.1)
        max_len = maximum sequence length 
        MAX  = positional encoder MAX parameter
        n_classes = number of classes 
    """

    def __init__(self, d_inp=36, d_model=64, nhead=4, nhid=128, nlayers=2, dropout=0.3, max_len=215, d_static=9,
                 MAX=100, perc=0.5, aggreg='mean', n_classes=2, global_structure=None, sensor_wise_mask=False, static=True):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'

        self.global_structure = global_structure
        self.sensor_wise_mask = sensor_wise_mask

        d_pe = 16
        d_enc = d_inp

        self.d_inp = d_inp
        self.d_model = d_model
        self.static = static
        if self.static:
            self.emb = nn.Linear(d_static, d_inp)

        self.d_ob = int(d_model/d_inp)

        self.encoder = nn.Linear(d_inp*self.d_ob, self.d_inp*self.d_ob)

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)

        if self.sensor_wise_mask == True:
            encoder_layers = TransformerEncoderLayer(self.d_inp*(self.d_ob+16), nhead, nhid, dropout)
        else:
            encoder_layers = TransformerEncoderLayer(d_model+16, nhead, nhid, dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.adj = torch.ones([self.d_inp, self.d_inp]).cuda()

        self.R_u = Parameter(torch.Tensor(1, self.d_inp*self.d_ob)).cuda()

        self.ob_propagation = Observation_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                    n_nodes=d_inp, ob_dim=self.d_ob)

        self.ob_propagation_layer2 = Observation_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                           n_nodes=d_inp, ob_dim=self.d_ob)

        if static == False:
            d_final = d_model + d_pe
        else:
            d_final = d_model + d_pe + d_inp

        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes),
        )

        self.aggreg = aggreg
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)
        glorot(self.R_u)

    def forward(self, src, static, times, lengths):
        """Input to the model:
        src = P: [215, 128, 36] : 36 nodes, 128 samples, each sample each channel has a feature with 215-D vector
        static = Pstatic: [128, 9]: this one doesn't matter; static features
        times = Ptime: [215, 128]: the timestamps
        lengths = lengths: [128]: the number of nonzero recordings.
        """
        maxlen, batch_size = src.shape[0], src.shape[1]
        missing_mask = src[:, :, self.d_inp:int(2*self.d_inp)]
        src = src[:, :, :int(src.shape[2]/2)]
        n_sensor = self.d_inp

        src = torch.repeat_interleave(src, self.d_ob, dim=-1)
        h = F.relu(src*self.R_u)
        pe = self.pos_encoder(times)
        if static is not None:
            emb = self.emb(static)

        h = self.dropout(h)

        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).cuda()

        step1 = True
        x = h
        if step1 == False:
            output = x
            distance = 0
        elif step1 == True:
            adj = self.global_structure.cuda()
            adj.fill_diagonal_(1)

            edge_index = torch.nonzero(adj).T
            edge_weights = adj[edge_index[0], edge_index[1]]

            batch_size = src.shape[1]
            n_step = src.shape[0]
            output = torch.zeros([n_step, batch_size, self.d_inp*self.d_ob]).cuda()

            use_beta = False
            if use_beta == True:
                alpha_all = torch.zeros([int(edge_index.shape[1]/2), batch_size]).cuda()
            else:
                alpha_all = torch.zeros([edge_index.shape[1],  batch_size]).cuda()
            for unit in range(0, batch_size):
                stepdata = x[:, unit, :]
                p_t = pe[:, unit, :]

                stepdata = stepdata.reshape([n_step, self.d_inp, self.d_ob]).permute(1, 0, 2)
                stepdata = stepdata.reshape(self.d_inp, n_step*self.d_ob)

                stepdata, attentionweights = self.ob_propagation(stepdata, p_t=p_t, edge_index=edge_index, edge_weights=edge_weights,
                                 use_beta=use_beta,  edge_attr=None, return_attention_weights=True)

                edge_index_layer2 = attentionweights[0]
                edge_weights_layer2 = attentionweights[1].squeeze(-1)

                stepdata, attentionweights = self.ob_propagation_layer2(stepdata, p_t=p_t, edge_index=edge_index_layer2, edge_weights=edge_weights_layer2,
                                 use_beta=False,  edge_attr=None, return_attention_weights=True)

                stepdata = stepdata.view([self.d_inp, n_step, self.d_ob])
                stepdata = stepdata.permute([1, 0, 2])
                stepdata = stepdata.reshape([-1, self.d_inp*self.d_ob])

                output[:, unit, :] = stepdata
                alpha_all[:, unit] = attentionweights[1].squeeze(-1)

            distance = torch.cdist(alpha_all.T, alpha_all.T, p=2)
            distance = torch.mean(distance)

        if self.sensor_wise_mask == True:
            extend_output = output.view(-1, batch_size, self.d_inp, self.d_ob)
            extended_pe = pe.unsqueeze(2).repeat([1, 1, self.d_inp, 1])
            output = torch.cat([extend_output, extended_pe], dim=-1)
            output = output.view(-1, batch_size, self.d_inp*(self.d_ob+16))
        else:
            output = torch.cat([output, pe], axis=2)

        step2 = True
        if step2 == True:
            r_out = self.transformer_encoder(output, src_key_padding_mask=mask)
        elif step2 == False:
            r_out = output

        sensor_wise_mask = self.sensor_wise_mask

        masked_agg = True
        if masked_agg == True:
            lengths2 = lengths.unsqueeze(1)
            mask2 = mask.permute(1, 0).unsqueeze(2).long()
            if sensor_wise_mask:
                output = torch.zeros([batch_size,self.d_inp, self.d_ob+16]).cuda()
                extended_missing_mask = missing_mask.view(-1, batch_size, self.d_inp)
                for se in range(self.d_inp):
                    r_out = r_out.view(-1, batch_size, self.d_inp, (self.d_ob+16))
                    out = r_out[:, :, se, :]
                    len = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(1)
                    out_sensor = torch.sum(out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0) / (len + 1)
                    output[:, se, :] = out_sensor
                output = output.view([-1, self.d_inp*(self.d_ob+16)])
            elif self.aggreg == 'mean':
                output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
        elif masked_agg == False:
            output = r_out[-1, :, :].squeeze(0)

        if static is not None:
            output = torch.cat([output, emb], dim=1)
        output = self.mlp_static(output)

        return output
def lengths_from_times(hours_TN: torch.Tensor) -> torch.Tensor:
    """hours_TN: [T,N], padding=-1/2880 hours."""
    return (hours_TN != MISSING_TIME_HR).sum(dim=0).clamp(min=1)


def _safe_metric_key(name: str) -> str:
    return str(name).strip().replace(" ", "_").replace("/", "_")


def run_train_eval(
    source: str,
    target: str,
    split: int = 1,
    data_root: str = "./data",
    processed_root: str = "./data_rd_template",
    use_wandb: bool = True,
    wandb_project: str = "Baselines",
    wandb_entity: str | None = None,
    seed: int = 9871,
    # keep original Raindrop baseline hyperparams
    num_epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-4,
    max_len: int = 800,
    nhead: int = 2,
    nlayers: int = 2,
    dropout: float = 0.2,
    MAX: int = 100,
    aggreg: str = "mean",
    sensor_wise_mask: bool = False,
    use_source_aug: bool = False,
    source_aug_suffix: str = "aug",
    use_target_aug: bool = False,
    target_aug_suffix: str = "aug",
):
    seed_everything(seed)

    if source == target:
        raise ValueError("source and target must be different")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        # The original Raindrop code commonly hardcodes .cuda() internally.
        # If you fixed the model to be device-safe, you can remove this guard.
        raise RuntimeError("Raindrop baseline requires CUDA (model uses .cuda() internally).")

    # split별 processed 저장 경로
    aug_tag = _aug_tag(
        use_source_aug, source_aug_suffix,
        use_target_aug, target_aug_suffix,
    )
    processed_root_run = os.path.join(processed_root, aug_tag)
    os.makedirs(processed_root_run, exist_ok=True)

    # ✅ build (or reuse) processed inputs
    create_inputs_from_data(
        domain=source,
        split=split,
        data_dir=data_root,
        processed_root=processed_root_run,
        max_len=max_len,
        use_aug=use_source_aug,
        aug_suffix=source_aug_suffix,
    )
    create_inputs_from_data(
        domain=target,
        split=1,
        data_dir=data_root,
        processed_root=processed_root_run,
        max_len=max_len,
        use_aug=use_target_aug,
        aug_suffix=target_aug_suffix,
    )

    # ✅ prepare tensors: source split=i, target test split=1
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


    # tensors
    Xtr = source_dict["Ptrain"]         # [T,N,2F]
    Ttr = source_dict["Ptrain_time"]    # [T,N]
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

    # outcome column names (saved during data preparation)
    outcome_cols = source_dict.get("outcome_cols", None)
    if outcome_cols is None:
        outcome_cols = [f"task_{i}" for i in range(Ytr.shape[1])]

    # move to device
    # Xtr, Ttr, Str, Ytr = Xtr.to(device), Ttr.to(device), Str.to(device), Ytr.to(device)
    # Xva, Tva, Sva, Yva = Xva.to(device), Tva.to(device), Sva.to(device), Yva.to(device)
    # Xte_s, Tte_s, Ste_s, Yte_s = Xte_s.to(device), Tte_s.to(device), Ste_s.to(device), Yte_s.to(device)
    # Xte_t, Tte_t, Ste_t, Yte_t = Xte_t.to(device), Tte_t.to(device), Ste_t.to(device), Yte_t.to(device)

    # infer dims
    T, Ntr, twoF = Xtr.shape
    F = twoF // 2
    D = Str.shape[1]
    C = Ytr.shape[1]

    # original Raindrop uses d_ob=4 => d_model = F * 4
    d_ob = 4
    d_model = F * d_ob
    nhid = 2 * d_model

    # original baseline uses a global_structure; simplest is all-ones adjacency
    global_structure = torch.ones(F, F, dtype=torch.float32, device=device)

    model = Raindrop_v2(
        d_inp=F,
        d_model=d_model,
        nhead=nhead,
        nhid=nhid,
        nlayers=nlayers,
        dropout=dropout,
        max_len=max_len,
        d_static=D,
        MAX=MAX,
        perc=0.5,
        aggreg=aggreg,
        n_classes=C,                    # multilabel => output dim = #tasks
        global_structure=global_structure,
        sensor_wise_mask=sensor_wise_mask,
        static=True,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # wandb
    wb = None
    run_name = f"Raindrop_{source}_{split}_{aug_tag}"
    if use_wandb:
        import wandb
        wb = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config={
                "model": "Raindrop_v2",
                "source": source,
                "target": target,
                "seed": seed,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "max_len": max_len,
                "F": F,
                "D": D,
                "C": C,
                "d_ob": d_ob,
                "d_model": d_model,
                "nhid": nhid,
                "nhead": nhead,
                "nlayers": nlayers,
                "dropout": dropout,
                "MAX": MAX,
                "aggreg": aggreg,
                "sensor_wise_mask": sensor_wise_mask,
                "outcome_cols": outcome_cols,
                "use_source_aug": use_source_aug,
                "source_aug_suffix": source_aug_suffix,
                "use_target_aug": use_target_aug,
                "target_aug_suffix": target_aug_suffix,
                "aug_tag": aug_tag,
            },
        )
        wandb.watch(model, log="gradients", log_freq=100)

    # early stopping on macro AUPRC (source val)
    patience = 7
    no_improve_epochs = 0
    best_val_auprc = -1.0
    best_state = None

    steps = max(1, int(np.ceil(Ntr / batch_size)))

    t0 = time.time()
    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(Ntr)

        loss_sum = 0.0
        pbar = tqdm(range(steps), desc=f"Epoch {epoch:02d}", leave=False)
        for s in pbar:
            idx = perm[s * batch_size:(s + 1) * batch_size]
            if idx.numel() == 0:
                continue

            Xb = Xtr[:, idx, :].to(device, non_blocking=True)
            Tb = Ttr[:, idx].to(device, non_blocking=True)
            Sb = (Str[idx].to(device, non_blocking=True) if Str is not None else None)
            Yb = Ytr[idx].to(device, non_blocking=True)

            lengths = lengths_from_times(Tb)

            # Raindrop_v2 returns (logits, distance, aux)
            out = model(Xb, Sb, Tb, lengths)
            logits = out[0] if isinstance(out, (tuple, list)) else out

            loss = criterion(logits, Yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())
            pbar.set_postfix({"loss": float(loss.item())})

        train_loss = loss_sum / max(1, steps)

        # validation (includes per-task metrics)
        (
            val_loss,
            val_auroc,
            val_auprc,
            val_acc,
            val_valid,
            val_auroc_per,
            val_auprc_per,
            val_valid_mask,
        ) = metrics_multilabel_batched(
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

            # per-task metrics with outcome column names
            for i, name in enumerate(outcome_cols):
                if i < len(val_valid_mask) and bool(val_valid_mask[i]):
                    key = _safe_metric_key(name)
                    log_dict[f"val/auroc_{key}"] = float(val_auroc_per[i])
                    log_dict[f"val/auprc_{key}"] = float(val_auprc_per[i])

            wb.log(log_dict)

        # early stopping on macro AUPRC
        if np.isfinite(val_auprc) and float(val_auprc) > best_val_auprc:
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

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # save checkpoint
    save_dir = "./baseline_models_rd"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{run_name}.pt")

    ckpt = {
        "run_name": run_name,
        "best_val_auprc": best_val_auprc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "outcome_cols": outcome_cols,
        "config": {
            "source": source,
            "split": split,
            "target": target,
            "seed": seed,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "max_len": max_len,
            "F": F,
            "D": D,
            "C": C,
            "d_ob": d_ob,
            "d_model": d_model,
            "nhid": nhid,
            "nhead": nhead,
            "nlayers": nlayers,
            "dropout": dropout,
            "MAX": MAX,
            "aggreg": aggreg,
            "sensor_wise_mask": sensor_wise_mask,
            "use_source_aug": use_source_aug,
            "source_aug_suffix": source_aug_suffix,
            "use_target_aug": use_target_aug,
            "target_aug_suffix": target_aug_suffix,
            "aug_tag": aug_tag,
        },
    }
    torch.save(ckpt, save_path)
    print(f"✅ Saved trained model to {save_path}")

    # final tests (source + target), with per-task logging
    (
        src_test_loss,
        src_auroc,
        src_auprc,
        src_acc,
        src_valid,
        src_auroc_per,
        src_auprc_per,
        src_valid_mask,
    ) = metrics_multilabel_batched(
        model, Xte_s, Tte_s, Ste_s, Yte_s,
        criterion=criterion,
        batch_size=batch_size,
        device=device,
    )

    (
        tgt_test_loss,
        tgt_auroc,
        tgt_auprc,
        tgt_acc,
        tgt_valid,
        tgt_auroc_per,
        tgt_auprc_per,
        tgt_valid_mask,
    ) = metrics_multilabel_batched(
        model, Xte_t, Tte_t, Ste_t, Yte_t,
        criterion=criterion,
        batch_size=batch_size,
        device=device,
    )

    print(f"[SOURCE TEST] loss={src_test_loss:.4f} AUROC={src_auroc:.4f} AUPRC={src_auprc:.4f} ACC={src_acc:.4f}")
    print(f"[TARGET TEST] loss={tgt_test_loss:.4f} AUROC={tgt_auroc:.4f} AUPRC={tgt_auprc:.4f} ACC={tgt_acc:.4f}")

    if wb is not None:
        log_dict = {
            "test/source_loss": src_test_loss,
            "test/source_auroc": src_auroc,
            "test/source_auprc": src_auprc,
            "test/source_acc": src_acc,
            "test/target_loss": tgt_test_loss,
            "test/target_auroc": tgt_auroc,
            "test/target_auprc": tgt_auprc,
            "test/target_acc": tgt_acc,
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=str,
        default="mimic",
        choices=["mimic", "eicu"],
        help="Source domain (target will be the other one).",
    )
    parser.add_argument("--split", type=int, default=1, choices=[1,2,3,4,5])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--processed_root", type=str, default="./data_rd_template")

    parser.add_argument("--max_len", type=int, default=800)

    # keep original baseline hyperparams by default
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--wandb", action="store_false")
    parser.add_argument("--wandb_project", type=str, default="Baselines")
    parser.add_argument("--wandb_entity", type=str, default="jwseo118-korea-university")

    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--MAX", type=int, default=100)
    parser.add_argument("--aggreg", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--sensor_wise_mask", action="store_true")
    parser.add_argument("--use_source_aug", action="store_true")
    parser.add_argument("--source_aug_suffix", type=str, default="aug")
    parser.add_argument("--use_target_aug", action="store_true")
    parser.add_argument("--target_aug_suffix", type=str, default="aug")
    parser.add_argument("--seed", type=int, default=9871)

    args = parser.parse_args()

    target = "eicu" if args.source == "mimic" else "mimic"

    run_train_eval(
        source=args.source,
        target=target,
        split=args.split,
        data_root=args.data_root,
        processed_root=args.processed_root,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        seed=args.seed,
        num_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        max_len=args.max_len,
        nhead=args.nhead,
        nlayers=args.nlayers,
        dropout=args.dropout,
        MAX=args.MAX,
        aggreg=args.aggreg,
        sensor_wise_mask=args.sensor_wise_mask,
        use_source_aug=args.use_source_aug,
        source_aug_suffix=args.source_aug_suffix,
        use_target_aug=args.use_target_aug,
        target_aug_suffix=args.target_aug_suffix,
    )


if __name__ == "__main__":
    main()