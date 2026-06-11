from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional, Sequence


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError(f"Boolean value expected, got: {v}")


@dataclass
class RunConfig:
    # general
    task_name: Literal["p12p19", "domain"]
    source: str
    target_domain: Optional[str]

    model_name: Literal["strats", "surprise", "surprise_vt"]
    model_cfg: Dict[str, Any]

    task_type: Literal["binary", "multilabel"]
    outcome_cols: Sequence[str]
    target_indices: Optional[Sequence[int]]

    batch_size: int = 64
    num_workers: int = 0
    lr: float = 5e-4
    weight_decay: float = 1e-2
    optimizer: Literal["adamw", "adam"] = "adamw"
    scheduler: Literal["none", "cosine"] = "none"
    scheduler_params: Optional[Dict[str, Any]] = None

    max_epochs_pretrain: int = 0
    max_epochs_train: int = 20
    pre_mask_p_train: float = 0.15

    pretrained_path: Optional[str] = None
    save_pretrained_after: bool = False
    downstream_path: Optional[str] = None

    eval_only: bool = False
    log_to_wandb: bool = True
    project_name: str = "Surprise"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    output_directory: str = "./outputs"

    accelerator: str = "auto"
    devices: Any = "auto"
    precision: str | int = "16-mixed"

    # ---- P12/P19/PAM options (only when task_name="p12p19") ----
    p12p19_dataset: Optional[Literal["P12", "P19", "PAM"]] = None
    p12p19_base_path: Optional[str] = None
    p12p19_split_path: Optional[str] = None
    p12p19_num_splits: int = 5
    p12p19_split_path_pattern: Optional[str] = None
    p12p19_split_type: str = "random"
    p12p19_reverse: bool = False
    p12p19_baseline: bool = False
    p12p19_predictive_label: str = "mortality"
    p12p19_max_tokens: int = 4096

    # ---- feather domain options (only when task_name="domain") ----
    raw_directory: str = "./data"
    categorical_itemids: Iterable[int | float] = ()
    aug: bool = False


def build_cfg_from_args(args) -> RunConfig:
    model_cfg = json.loads(args.model_cfg_json)

    outcome_cols_tuple = tuple([c for c in args.outcome_cols.split(",") if c != ""])
    eval_only = str2bool(args.eval_only)
    log_to_wandb = str2bool(args.log_to_wandb)
    aug = str2bool(args.aug)

    # target_indices: 기본 None (전체), 필요하면 CLI로 추가해도 됨
    target_indices = None

    return RunConfig(
        task_name=args.task_name,
        source=args.source,
        target_domain=(args.target_domain if args.target_domain else None),
        model_name=args.model_name,
        model_cfg=model_cfg,
        task_type=args.task_type,
        outcome_cols=outcome_cols_tuple,
        target_indices=target_indices,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        scheduler_params=None,
        max_epochs_pretrain=int(args.max_epochs_pretrain),
        max_epochs_train=int(args.max_epochs_train),
        pre_mask_p_train=float(args.pre_mask_p_train),
        pretrained_path=(args.pretrained_path if args.pretrained_path else None),
        save_pretrained_after=str2bool(args.save_pretrained_after),
        downstream_path=(args.downstream_path if args.downstream_path else None),
        eval_only=eval_only,
        log_to_wandb=log_to_wandb,
        project_name=args.project_name,
        wandb_entity=(args.wandb_entity if args.wandb_entity else None),
        wandb_run_name=(args.wandb_run_name if args.wandb_run_name else None),
        output_directory=args.output_directory,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        # P12/P19/PAM
        p12p19_dataset=(args.p12p19_dataset if args.p12p19_dataset else None),
        p12p19_base_path=(args.p12p19_base_path if args.p12p19_base_path else None),
        p12p19_split_path=(args.p12p19_split_path if args.p12p19_split_path else None),
        p12p19_num_splits=int(args.p12p19_num_splits),
        p12p19_split_path_pattern=(args.p12p19_split_path_pattern if args.p12p19_split_path_pattern else None),
        p12p19_split_type=args.p12p19_split_type,
        p12p19_reverse=str2bool(args.p12p19_reverse),
        p12p19_baseline=str2bool(args.p12p19_baseline),
        p12p19_predictive_label=args.p12p19_predictive_label,
        p12p19_max_tokens=int(args.p12p19_max_tokens),
        # feather domain
        raw_directory=args.raw_directory,
        aug=aug,
    )
