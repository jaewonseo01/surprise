from __future__ import annotations

import os
import json
import argparse
import wandb
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

try:
    from pytorch_lightning.loggers import WandbLogger
    _WANDB_OK = True
except Exception:
    WandbLogger = None
    _WANDB_OK = False

# ---- project modules ----
from surprise_data.feather_domains import build_feather_split_loaders
from surprise_configs_feather import build_feather_cfg_from_args, FeatherConfig
from surprise_models.build import build_model
from surprise_lit.pretrain import PretrainLit
from surprise_lit.downstream import STRaTSLit


# ---------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------
def seed_everything(seed: int):
    pl.seed_everything(int(seed), workers=True)


def _safe_metric_key(name: str) -> str:
    return str(name).strip().replace(" ", "_").replace("/", "_")


def _boolify(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def _aug_tag(cfg) -> str:
    parts = []
    if getattr(cfg, "use_source_aug", False):
        parts.append(f"srcAug-{getattr(cfg, 'source_aug_suffix', 'aug')}")
    else:
        parts.append("srcOrig")

    if getattr(cfg, "use_target_aug", False):
        parts.append(f"tgtAug-{getattr(cfg, 'target_aug_suffix', 'aug')}")
    else:
        parts.append("tgtOrig")

    return "__".join(parts)


def make_trainer(
    *,
    cfg: FeatherConfig,
    logger,
    stage: str,        # "pretrain" or "train"
    monitor: str,
    mode: str,
    max_epochs: int,
    patience: int,
    out_dir: str,
    run_name: str | None = None,
    progress_bar: bool = False,
    gpu_id: int = 0,
):
    callbacks = []

    ckpt_dir = os.path.join(out_dir, stage, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    prefix = run_name or f"{cfg.model_name}_{cfg.source_domain}"
    prefix = prefix.replace("/", "_").replace("\\", "_").replace(" ", "_")

    callbacks.append(ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{prefix}-{stage}" + "-{epoch}-{" + monitor + ":.4f}",
        monitor=monitor,
        mode=mode,
        save_top_k=1,
    ))

    callbacks.append(EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=int(patience),
        check_on_train_epoch_end=False,
    ))

    trainer = pl.Trainer(
        max_epochs=int(max_epochs),
        logger=logger,
        callbacks=callbacks,
        precision=cfg.precision,
        accelerator=cfg.accelerator,
        devices=[gpu_id],
        enable_checkpointing=True,
        enable_progress_bar=progress_bar,
        log_every_n_steps=50,
    )
    return trainer


# ---------------------------------------------------------------------
# one split
# ---------------------------------------------------------------------
def run_one_split(cfg: FeatherConfig, *, split_id: int, progress_bar: bool = False, gpu_id: int = 0):
    split_id = int(split_id)
    seed_everything(cfg.seed)

    split_out = os.path.join(cfg.output_directory, f"{cfg.project_name}_split{split_id}")
    os.makedirs(split_out, exist_ok=True)

    logger = None
    wb_run = None

    aug_tag = _aug_tag(cfg)

    if cfg.use_source_aug or cfg.use_target_aug:

        run_name = f"{cfg.model_name}_{cfg.source_domain}_to_{cfg.target_domain}_split{split_id}__{aug_tag}"
    else:
        run_name = f"{cfg.model_name}_{cfg.source_domain}_{split_id}"
    if cfg.run_name_tag:
        run_name += f"-{cfg.run_name_tag}"

    try:
        # -------------------------------------------------------------
        # W&B: ONE run per split
        # -------------------------------------------------------------
        if cfg.log_to_wandb:
            if not _WANDB_OK:
                raise RuntimeError("log_to_wandb=True but WandbLogger import failed.")

            logger = WandbLogger(
                project=cfg.project_name,
                entity=cfg.wandb_entity,
                name=run_name,
                save_dir=split_out,
                reinit=True,
            )
            wb_run = logger.experiment

        # -------------------------------------------------------------
        # loaders
        # -------------------------------------------------------------
        (
            num_features,
            pre_ld_tr, pre_ld_va,
            ld_tr, ld_va, ld_te,
            target_loader,
        ) = build_feather_split_loaders(cfg, split_id=split_id)
        print("data loaded yay")
        # -------------------------------------------------------------
        # build model
        # -------------------------------------------------------------
        model = build_model(
            cfg.model_name,
            cfg.model_cfg,
            num_features=num_features,
            n_output=cfg.n_output
        )

        with open(
            os.path.join(
                split_out,
                f"config_{cfg.source_domain}_to_{cfg.target_domain}__{aug_tag}.json"
            ),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(cfg.__dict__, f, indent=2, default=str)

        # -------------------------------------------------------------
        # pretrain
        # -------------------------------------------------------------
        if cfg.do_pretrain:
            pretrainer = make_trainer(
                cfg=cfg,
                logger=logger,
                stage="pretrain",
                monitor="pretrain_val_loss",
                mode="min",
                max_epochs=cfg.max_epochs_pretrain,
                patience=cfg.earlystop_pretrain_patience,
                out_dir=split_out,
                run_name=run_name,
                progress_bar=progress_bar,
                gpu_id=gpu_id,
            )

            lit_pre = PretrainLit(
                model=model,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                optimizer=cfg.optimizer,
            )

            pretrainer.fit(lit_pre, train_dataloaders=pre_ld_tr, val_dataloaders=pre_ld_va)

            best_path = pretrainer.checkpoint_callback.best_model_path
            if best_path:
                ckpt = torch.load(best_path, map_location="cpu")
                lit_pre.load_state_dict(ckpt["state_dict"], strict=False)

            model = lit_pre.model

        # -------------------------------------------------------------
        # downstream train
        # -------------------------------------------------------------
        trainer = make_trainer(
            cfg=cfg,
            logger=logger,
            stage="train",
            monitor="val_AUPRC_macro",
            mode="max",
            max_epochs=cfg.max_epochs_train,
            patience=cfg.earlystop_train_patience,
            out_dir=split_out,
            run_name=run_name,
            progress_bar=progress_bar,
            gpu_id=gpu_id,
        )

        lit_train = STRaTSLit(
            model=model,
            task_type=cfg.task_type,
            outcome_cols=cfg.outcome_cols,
            target_indices=cfg.target_indices,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            optimizer=cfg.optimizer,
        )

        trainer.fit(lit_train, train_dataloaders=ld_tr, val_dataloaders=ld_va)

        # -------------------------------------------------------------
        # evaluation
        # -------------------------------------------------------------
        trainer.test(lit_train, dataloaders=ld_te, ckpt_path="best")
        source_pack = lit_train._last_test

        trainer.test(lit_train, dataloaders=target_loader, ckpt_path="best")
        target_pack = lit_train._last_test

        # -------------------------------------------------------------
        # manual W&B logging
        # -------------------------------------------------------------
        if cfg.log_to_wandb and wb_run is not None:
            log_dict = {
                "test/source_auroc": float(source_pack["auroc_macro"]),
                "test/source_auprc": float(source_pack["auprc_macro"]),
                "test/source_acc":   float(source_pack["acc_macro"]),
                "test/target_auroc": float(target_pack["auroc_macro"]),
                "test/target_auprc": float(target_pack["auprc_macro"]),
                "test/target_acc":   float(target_pack["acc_macro"]),
                "split": split_id,
                "source": cfg.source_domain,
                "target": cfg.target_domain,
                "use_source_aug": bool(getattr(cfg, "use_source_aug", False)),
                "source_aug_suffix": getattr(cfg, "source_aug_suffix", "aug"),
                "use_target_aug": bool(getattr(cfg, "use_target_aug", False)),
                "target_aug_suffix": getattr(cfg, "target_aug_suffix", "aug"),
            }

            outcome_cols = list(cfg.outcome_cols)

            src_valid = source_pack["valid_mask"]
            src_auroc_per = source_pack["auroc_per"]
            src_auprc_per = source_pack["auprc_per"]

            tgt_valid = target_pack["valid_mask"]
            tgt_auroc_per = target_pack["auroc_per"]
            tgt_auprc_per = target_pack["auprc_per"]

            for i, name in enumerate(outcome_cols):
                key = _safe_metric_key(name)

                if i < len(src_valid) and bool(src_valid[i]):
                    log_dict[f"test/source_auroc_{key}"] = float(src_auroc_per[i])
                    log_dict[f"test/source_auprc_{key}"] = float(src_auprc_per[i])

                if i < len(tgt_valid) and bool(tgt_valid[i]):
                    log_dict[f"test/target_auroc_{key}"] = float(tgt_auroc_per[i])
                    log_dict[f"test/target_auprc_{key}"] = float(tgt_auprc_per[i])

            wb_run.log(log_dict)

    finally:
        if wb_run is not None:
            try:
                wb_run.finish()
            except Exception:
                pass
        else:
            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    _wandb.finish()
            except Exception:
                pass


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    torch.set_float32_matmul_precision("high")

    ocs_df = pd.read_feather('./data/split1/mimic_outcomes_train_1.feather')

    ocs = ocs_df.columns.values.tolist()
    ocs.remove('pid')
    print(ocs)

    ap = argparse.ArgumentParser()

    # config args
    ap.add_argument("--project_name", type=str, default="CV_5fold")
    ap.add_argument("--wandb_entity", type=str, default="jwseo118-korea-university")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--source_domain", type=str, default="mimic", choices=["mimic", "eicu"])
    ap.add_argument("--target_domain", type=str, default="eicu", choices=["mimic", "eicu"])
    ap.add_argument("--num_splits", type=int, default=5)
    ap.add_argument("--fixed_target_test_split", type=int, default=1)

    ap.add_argument("--model_name", type=str, default="surprise",
                    choices=["strats", "surprise", "surprise_vt", "surprise_vttg"])
    ap.add_argument("--model_cfg_json", type=str, default="{}")

    ap.add_argument("--task_type", type=str, default="multilabel", choices=["binary", "multilabel"])

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam"])

    ap.add_argument("--precision", type=str, default="16-mixed")
    ap.add_argument("--accelerator", type=str, default="auto")
    ap.add_argument("--devices", default="auto")
    ap.add_argument("--seed", type=int, default=9871)

    ap.add_argument("--do_pretrain", type=str, default="True")
    ap.add_argument("--max_epochs_pretrain", type=int, default=50)
    ap.add_argument("--pre_mask_p_train", type=float, default=0.15)
    ap.add_argument("--earlystop_pretrain_patience", type=int, default=7)

    ap.add_argument("--max_epochs_train", type=int, default=50)
    ap.add_argument("--earlystop_train_patience", type=int, default=7)

    ap.add_argument("--log_to_wandb", type=str, default="True")
    ap.add_argument("--output_directory", type=str, default="./surprise_results")

    # additional run name tag
    ap.add_argument("--run_name_tag", type=str, default="")

    # NEW: augmentation args
    ap.add_argument("--use_source_aug", type=str, default="False")
    ap.add_argument("--source_aug_suffix", type=str, default="aug")
    ap.add_argument("--use_target_aug", type=str, default="False")
    ap.add_argument("--target_aug_suffix", type=str, default="aug")
    ap.add_argument("--progress_bar", type=str, default="False")
    ap.add_argument("--gpu_id", type=int, default=0, choices=[0, 1])

    args = ap.parse_args()

    # argparse bool string -> real bool
    args.do_pretrain = _boolify(args.do_pretrain)
    args.log_to_wandb = _boolify(args.log_to_wandb)
    args.use_source_aug = _boolify(args.use_source_aug)
    args.use_target_aug = _boolify(args.use_target_aug)
    args.progress_bar = _boolify(args.progress_bar)

    cfg = build_feather_cfg_from_args(args, ocs)

    #for split_id in range(1, cfg.num_splits + 1):
    for split_id in range(6-cfg.num_splits, 6):
        run_one_split(cfg, split_id=split_id, progress_bar=args.progress_bar, gpu_id=args.gpu_id)


if __name__ == "__main__":
    main()
