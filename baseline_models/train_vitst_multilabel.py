# train_vitst_multilabel.py
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from runner_raindrop import create_inputs_from_data, prepare_data, seed_everything
from utils_rd import metrics_multilabel_batched, tensorize_normalize_multilabel

# Conventions (must match runner/utils)
MISSING_VALUE = 0.0
MISSING_TIME_MIN = -1.0
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


def run_train_eval(
    source: str,
    target: str,
    split: int = 1,
    data_root: str = "./data",
    processed_root: str = "./data_rd_template",
    use_wandb: bool = False,
    wandb_project: str = "Baselines",
    wandb_entity: str | None = None,
    num_epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    max_len: int = 800,
    nhead: int = 4,
    nlayers: int = 2,
    dropout: float = 0.2,
    MAX: int = 100,
    aggreg: str = "mean",
    use_source_aug: bool = False,
    source_aug_suffix: str = "aug",
    use_target_aug: bool = False,
    target_aug_suffix: str = "aug",
    seed: int = 9871,
):
    seed_everything(seed)

    if source == target:
        raise ValueError("source and target must be different")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("ViT/Swin baselines require CUDA.")

    # build or reuse processed input files
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

    Xtr = source_dict["Ptrain"]
    Ttr = source_dict["Ptrain_time"]
    Str = source_dict["Ptrain_static"]
    Ytr = source_dict["ytrain"].float()

    Xva = source_dict["Pval"]
    Tva = source_dict["Ptrain_time"]
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

    Xtr, Ttr, Str, Ytr = Xtr.to(device), Ttr.to(device), Str.to(device), Ytr.to(device)
    Xva, Tva, Sva, Yva = Xva.to(device), Tva.to(device), Sva.to(device), Yva.to(device)
    Xte_s, Tte_s, Ste_s, Yte_s = Xte_s.to(device), Tte_s.to(device), Ste_s.to(device), Yte_s.to(device)
    Xte_t, Tte_t, Ste_t, Yte_t = Xte_t.to(device), Tte_t.to(device), Ste_t.to(device), Yte_t.to(device)

    T, N, Cx = Xtr.shape

    # import model builders from baseline model files
    from modeling_vit import BeitForImageClassification
    from modeling_swin import SwinForImageClassification
    from configuration_vit import ViTConfig
    from configuration_swin import SwinConfig

    if "vit" in target.lower():
        config = ViTConfig(
            image_size=128,
            patch_size=16,
            num_channels=Cx,
            num_labels=Ytr.shape[1],
        )
        model = BeitForImageClassification(config).to(device)
    else:
        config = SwinConfig(
            image_size=128,
            patch_size=16,
            num_channels=Cx,
            num_labels=Ytr.shape[1],
        )
        model = SwinForImageClassification(config).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    wb = None
    run_name = f"ViTST_{source}_to_{target}_split{split}__{aug_tag}"
    if use_wandb:
        import wandb
        wb = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config={
                "source": source,
                "target": target,
                "split": split,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "max_len": max_len,
                "aggreg": aggreg,
                "nhead": nhead,
                "nlayers": nlayers,
                "dropout": dropout,
                "MAX": MAX,
                "aug_tag": aug_tag,
            },
        )
        wandb.watch(model, log="gradients", log_freq=100)

    def lengths_from_times(hours_TN: torch.Tensor) -> torch.Tensor:
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

    torch.save({
        "run_name": run_name,
        "best_val_auprc": best_val_auprc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "outcome_cols": outcome_cols,
        "config": {
            "source": source,
            "target": target,
            "split": split,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "max_len": max_len,
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
    }, save_path)

    print(f"✅ Saved trained model to {save_path}")

    src_test_loss, src_auroc, src_auprc, src_acc, src_valid, src_auroc_per, src_auprc_per, src_valid_mask = metrics_multilabel_batched(
        model, Xte_s, Tte_s, Ste_s, Yte_s, criterion, batch_size=batch_size, device=device
    )
    tgt_test_loss, tgt_auroc, tgt_auprc, tgt_acc, tgt_valid, tgt_auroc_per, tgt_auprc_per, tgt_valid_mask = metrics_multilabel_batched(
        model, Xte_t, Tte_t, Ste_t, Yte_t, criterion, batch_size=batch_size, device=device
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
            safe_name = str(name).replace(" ", "_")
            if i < len(src_valid_mask) and src_valid_mask[i]:
                log_dict[f"test/source_auroc_{safe_name}"] = float(src_auroc_per[i])
                log_dict[f"test/source_auprc_{safe_name}"] = float(src_auprc_per[i])
            if i < len(tgt_valid_mask) and tgt_valid_mask[i]:
                log_dict[f"test/target_auroc_{safe_name}"] = float(tgt_auroc_per[i])
                log_dict[f"test/target_auprc_{safe_name}"] = float(tgt_valid_mask[i])
        wb.log(log_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="mimic", choices=["mimic", "eicu"])
    parser.add_argument("--split", type=int, default=1, choices=[1,2,3,4,5])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--processed_root", type=str, default="./data_rd_template")
    parser.add_argument("--max_len", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wandb", action="store_false")
    parser.add_argument("--wandb_project", type=str, default="Baselines")
    parser.add_argument("--wandb_entity", type=str, default="jwseo118-korea-university")
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--MAX", type=int, default=100)
    parser.add_argument("--aggreg", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--use_source_aug", action="store_true")
    parser.add_argument("--source_aug_suffix", type=str, default="aug")
    parser.add_argument("--use_target_aug", action="store_true")
    parser.add_argument("--target_aug_suffix", type=str, default="aug")
    args = parser.parse_args()

    target = "eicu" if args.source == "mimic" else "mimic"
    run_train_eval(
        source=args.source,
        target=target,
        split=args.split,
        data_root=args.data_root,
        processed_root=args.processed_root,
        num_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        max_len=args.max_len,
        nhead=args.nhead,
        nlayers=args.nlayers,
        dropout=args.dropout,
        MAX=args.MAX,
        aggreg=args.aggreg,
        use_source_aug=args.use_source_aug,
        source_aug_suffix=args.source_aug_suffix,
        use_target_aug=args.use_target_aug,
        target_aug_suffix=args.target_aug_suffix,
    )


if __name__ == "__main__":
    main()
