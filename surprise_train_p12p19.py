from __future__ import annotations

import os
import json
import argparse
from dataclasses import asdict

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from surprise_configs_p12p19 import build_p12p19_cfg_from_args
from surprise_models.build import build_model
from surprise_lit.pretrain import PretrainLit
from surprise_lit.downstream import STRaTSLit
from surprise_data.p12p19 import P12P19DataModule  # must exist in your codebase

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def _make_logger(cfg, split_id: int, run_name: str):
    if not cfg.log_to_wandb:
        return None, None

    import wandb
    from pytorch_lightning.loggers import WandbLogger

    run = wandb.init(
        project=cfg.project_name,
        entity=cfg.wandb_entity,
        name=run_name,
        config={**asdict(cfg), "split_id": split_id},
        reinit=True,
    )
    return run, WandbLogger(experiment=run)


def _monitor_name(task_type: str) -> str:
    return "val_AUPRC" if task_type == "binary" else "val_AUPRC_macro"


def _make_trainer(
    *,
    cfg,
    logger,
    stage: str,
    monitor: str,
    mode: str,
    max_epochs: int,
    patience: int,
    out_dir: str,
    run_name: str,
    progress_bar: bool = False,
):
    ckpt_dir = os.path.join(out_dir, stage, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    prefix = run_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{prefix}-{stage}" + "-{epoch}-{" + monitor + ":.4f}",
        monitor=monitor,
        mode=mode,
        save_top_k=1,
    )
    es = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=int(patience),
        check_on_train_epoch_end=False,
    )

    return pl.Trainer(
        max_epochs=int(max_epochs),
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        logger=logger,
        callbacks=[ckpt, es],
        enable_checkpointing=True,
        enable_progress_bar=progress_bar,
        log_every_n_steps=50,
    )


def main():
    p = argparse.ArgumentParser()

    # core
    p.add_argument("--project_name", required=True)
    p.add_argument("--wandb_entity", default="")
    p.add_argument("--log_to_wandb", default="True")
    p.add_argument("--output_directory", default="./outputs")

    # data
    p.add_argument("--dataset", required=True, choices=["P12", "P19"])
    p.add_argument("--base_path", required=True)
    p.add_argument("--split_path_pattern", required=True)  # must contain {} for split index (1..5)
    p.add_argument("--split_type", default="random", choices=["random", "age", "gender"])
    p.add_argument("--reverse", default="False")
    p.add_argument("--baseline", default="False")
    p.add_argument("--predictive_label", default="mortality", choices=["mortality", "LoS"])
    p.add_argument("--max_tokens", default="4096")

    # model
    p.add_argument("--model_name", required=True, choices=["strats", "surprise", "surprise_vt", "surprise_vttg"])
    p.add_argument("--model_cfg_json", required=True)

    # task
    p.add_argument("--task_type", required=True, choices=["binary", "multilabel"])
    p.add_argument("--outcome_cols", required=True)

    # opt
    p.add_argument("--batch_size", default="32")
    p.add_argument("--num_workers", default="0")
    p.add_argument("--lr", default="1e-3")
    p.add_argument("--weight_decay", default="1e-2")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adam"])
    p.add_argument("--precision", default="16-mixed")
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--devices", default="auto")
    p.add_argument("--seed", default="9871")

    # pretrain
    p.add_argument("--do_pretrain", default="True")
    p.add_argument("--max_epochs_pretrain", default="20")
    p.add_argument("--pre_mask_p_train", default="0.15")
    p.add_argument("--earlystop_pretrain_patience", default="7")

    # downstream
    p.add_argument("--max_epochs_train", default="20")
    p.add_argument("--earlystop_train_patience", default="7")

    # CV
    p.add_argument("--num_splits", default="5")

    # progress bar
    p.add_argument("--progress_bar", default="False")

    # additional run name tag
    p.add_argument("--run_name_tag", default="")

    args = p.parse_args()
    progress_bar = args.progress_bar.lower() in ("true", "1", "yes", "y")
    cfg = build_p12p19_cfg_from_args(args)


    os.makedirs(cfg.output_directory, exist_ok=True)

    #for split_id in range(1, cfg.num_splits + 1):
    for split_id in range(6-cfg.num_splits, 6):
        pl.seed_everything(cfg.seed, workers=True)

        split_path = cfg.split_path_pattern.format(split_id)

        dm = P12P19DataModule(
            dataset=cfg.dataset,
            base_path=cfg.base_path,
            split_path=split_path,
            split_type=cfg.split_type,
            reverse=cfg.reverse,
            baseline=cfg.baseline,
            predictive_label=cfg.predictive_label,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            max_tokens=cfg.max_tokens,
            pre_mask_p_train=cfg.pre_mask_p_train,
        )
        dm.setup()
        num_features = dm.num_features
        model_cfg = dict(cfg.model_cfg or {})
        model_cfg["static_dim"] = dm.static_dim

        core = build_model(
            cfg.model_name,
            model_cfg,
            num_features=num_features,
            n_output=cfg.n_output,
        )

        split_out = os.path.join(cfg.output_directory, f"{cfg.project_name}_split{split_id}")
        os.makedirs(split_out, exist_ok=True)

        run_name = f"{cfg.model_name}_{cfg.dataset}_split{split_id}"
        if cfg.run_name_tag:
            run_name += f"-{cfg.run_name_tag}"

        wb_run, logger = _make_logger(cfg, split_id, run_name)
        callbacks = []

        with open(
            os.path.join(split_out, f"config_{cfg.dataset}_split{split_id}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {**asdict(cfg), "split_id": split_id, "model_cfg": model_cfg},
                f,
                indent=2,
                default=str,
            )

        # ---- PRETRAIN (per split) ----
        if cfg.do_pretrain and cfg.max_epochs_pretrain > 0:
            pre_lit = PretrainLit(
                core,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                optimizer=cfg.optimizer,
            )
            trainer_pre = _make_trainer(
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
            )
            trainer_pre.fit(pre_lit, dm.pretrain_train_dataloader(), dm.pretrain_val_dataloader())

            best_pre_path = trainer_pre.checkpoint_callback.best_model_path
            if best_pre_path:
                ckpt = torch.load(best_pre_path, map_location="cpu")
                pre_lit.load_state_dict(ckpt["state_dict"], strict=False)
                core = pre_lit.model

        # ---- DOWNSTREAM (per split) ----
        lit = STRaTSLit(
            model=core,
            task_type=cfg.task_type,
            outcome_cols=cfg.outcome_cols,
            target_indices=cfg.target_indices,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            optimizer=cfg.optimizer,
            scheduler="none",
            scheduler_params=None,
            pretrained_path=None,
        )
        trainer = _make_trainer(
            cfg=cfg,
            logger=logger,
            stage="train",
            monitor=_monitor_name(cfg.task_type),
            mode="max",
            max_epochs=cfg.max_epochs_train,
            patience=cfg.earlystop_train_patience,
            out_dir=split_out,
            run_name=run_name,
            progress_bar=progress_bar,
        )
        trainer.fit(lit, dm.train_dataloader(), dm.val_dataloader())

        # ---- EVAL: source test (per split) ----
        best_train_path = trainer.checkpoint_callback.best_model_path
        if best_train_path:
            trainer.test(lit, dataloaders=dm.test_dataloader(), ckpt_path="best", verbose=False)
        else:
            trainer.test(lit, dataloaders=dm.test_dataloader(), verbose=False)

        if wb_run:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
