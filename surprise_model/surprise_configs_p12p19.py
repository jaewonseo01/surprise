from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence


def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise ValueError(f"Invalid bool: {v}")


@dataclass
class P12P19Config:
    # experiment id
    project_name: str
    wandb_entity: Optional[str] = None

    # data
    dataset: Literal["P12", "P19", "PAM"] = "P12"
    base_path: str = ""
    split_path_pattern: str = ""  # e.g. ".../splits/{}/...json" or ".../split_{}.json"
    split_type: Literal["random", "age", "gender"] = "random"
    reverse: bool = False
    baseline: bool = False
    predictive_label: Literal["mortality", "LoS"] = "mortality"
    max_tokens: int = 4096

    # model
    model_name: Literal["strats", "surprise", "surprise_vt"] = "surprise"
    model_cfg: Dict[str, Any] = None
    n_output: int = 1

    # downstream head/task
    task_type: Literal["binary", "multilabel"] = "binary"
    outcome_cols: Sequence[str] = ("mortality",)
    target_indices: Optional[Sequence[int]] = None

    # optimization
    batch_size: int = 64
    num_workers: int = 8
    lr: float = 5e-4
    weight_decay: float = 1e-2
    optimizer: Literal["adamw", "adam"] = "adamw"
    precision: str = "16-mixed"
    accelerator: str = "auto"
    devices: Any = "auto"
    seed: int = 42

    # pretrain
    do_pretrain: bool = True
    max_epochs_pretrain: int = 30
    pre_mask_p_train: float = 0.15
    earlystop_pretrain_patience: int = 5

    # downstream train
    max_epochs_train: int = 50
    earlystop_train_patience: int = 7  # monitor AUPRC(_macro)

    # CV
    num_splits: int = 5  # 1..5 inclusive

    # logging/outputs
    log_to_wandb: bool = True
    output_directory: str = "./outputs"

    # additional run name tag
    run_name_tag: str = ""


def build_p12p19_cfg_from_args(args) -> P12P19Config:
    model_cfg = json.loads(args.model_cfg_json) if args.model_cfg_json else {}

    outcome_cols = tuple([c for c in args.outcome_cols.split(",") if c.strip() != ""])
    target_indices = None  # keep None by default
    task_type = args.task_type
    n_output = 1 if task_type == "binary" else len(outcome_cols)

    return P12P19Config(
        project_name=args.project_name,
        wandb_entity=(args.wandb_entity or None),

        dataset=args.dataset,
        base_path=args.base_path,
        split_path_pattern=args.split_path_pattern,
        split_type=args.split_type,
        reverse=str2bool(args.reverse),
        baseline=str2bool(args.baseline),
        predictive_label=args.predictive_label,
        max_tokens=int(args.max_tokens),

        model_name=args.model_name,
        model_cfg=model_cfg,
        n_output=n_output,

        task_type=task_type,
        outcome_cols=outcome_cols,
        target_indices=target_indices,

        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        optimizer=args.optimizer,
        precision=args.precision,
        accelerator=args.accelerator,
        devices=args.devices,
        seed=int(args.seed),

        do_pretrain=str2bool(args.do_pretrain),
        max_epochs_pretrain=int(args.max_epochs_pretrain),
        pre_mask_p_train=float(args.pre_mask_p_train),
        earlystop_pretrain_patience=int(args.earlystop_pretrain_patience),

        max_epochs_train=int(args.max_epochs_train),
        earlystop_train_patience=int(args.earlystop_train_patience),

        num_splits=int(args.num_splits),

        log_to_wandb=str2bool(args.log_to_wandb),
        output_directory=args.output_directory,

        run_name_tag=args.run_name_tag,
    )
