from __future__ import annotations

import json
from dataclasses import dataclass, field
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
class FeatherConfig:
    # ------------------------------------------------------------------
    # experiment / logging
    # ------------------------------------------------------------------
    project_name: str
    wandb_entity: Optional[str] = None
    log_to_wandb: bool = True
    output_directory: str = "./outputs"

    # ------------------------------------------------------------------
    # data layout
    # ------------------------------------------------------------------
    # data_root/
    #   split1/
    #     {domain}_data_{train|valid|test}_{split}.feather
    #     {domain}_data_static_{train|valid|test}_{split}.feather
    #     {domain}_outcomes_{train|valid|test}_{split}.feather
    #
    # augmented data example:
    #     {domain}_data_aug_train_1.feather
    #     {domain}_data_aug_p2_123x3_test_4.feather
    # ------------------------------------------------------------------
    data_root: str = "./data"
    source_domain: Literal["mimic", "eicu"] = "mimic"
    target_domain: Literal["mimic", "eicu"] = "eicu"

    # CV
    num_splits: int = 5
    fixed_target_test_split: int = 1  # target test always from split1

    # augmentation controls
    use_source_aug: bool = False
    source_aug_suffix: str = "aug"
    use_target_aug: bool = False
    target_aug_suffix: str = "aug"

    # ------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------
    model_name: Literal[
        "strats", "surprise", "surprise_vt", "surprise_vttg"
    ] = "surprise"
    model_cfg: Dict[str, Any] = field(default_factory=dict)
    n_output: int = 1

    # ------------------------------------------------------------------
    # task
    # ------------------------------------------------------------------
    task_type: Literal["binary", "multilabel"] = "multilabel"
    outcome_cols: Sequence[str] = ()
    target_indices: Optional[Sequence[int]] = None
    categorical_itemids: Optional[Sequence[int]] = None

    # ------------------------------------------------------------------
    # optimization / trainer
    # ------------------------------------------------------------------
    batch_size: int = 32
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-2
    optimizer: Literal["adamw", "adam"] = "adamw"

    precision: str = "16-mixed"
    accelerator: str = "auto"
    devices: Any = "auto"
    seed: int = 9871

    # ------------------------------------------------------------------
    # pretrain
    # ------------------------------------------------------------------
    do_pretrain: bool = True
    max_epochs_pretrain: int = 50
    pre_mask_p_train: float = 0.15
    earlystop_pretrain_patience: int = 5

    # ------------------------------------------------------------------
    # downstream train
    # ------------------------------------------------------------------
    max_epochs_train: int = 50
    earlystop_train_patience: int = 7

    # additional run name tag
    run_name_tag: str = ""


def build_feather_cfg_from_args(args, ocs) -> FeatherConfig:
    model_cfg = json.loads(args.model_cfg_json) if getattr(args, "model_cfg_json", None) else {}
    outcome_cols = ocs

    return FeatherConfig(
        # experiment / logging
        project_name=args.project_name,
        wandb_entity=(args.wandb_entity or None),
        log_to_wandb=str2bool(args.log_to_wandb),
        output_directory=args.output_directory,

        run_name_tag=args.run_name_tag,

        # data
        data_root=args.data_root,
        source_domain=args.source_domain,
        target_domain=args.target_domain,
        num_splits=int(args.num_splits),
        fixed_target_test_split=int(args.fixed_target_test_split),

        # augmentation
        use_source_aug=str2bool(getattr(args, "use_source_aug", False)),
        source_aug_suffix=str(getattr(args, "source_aug_suffix", "aug")),
        use_target_aug=str2bool(getattr(args, "use_target_aug", False)),
        target_aug_suffix=str(getattr(args, "target_aug_suffix", "aug")),

        # model
        model_name=args.model_name,
        model_cfg=model_cfg,
        n_output=len(ocs),

        # task
        task_type=args.task_type,
        outcome_cols=outcome_cols,
        target_indices=None,
        categorical_itemids=[i for i in range(19, 35)],

        # optimization / trainer
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        optimizer=args.optimizer,
        precision=args.precision,
        accelerator=args.accelerator,
        devices=args.devices,
        seed=int(args.seed),

        # pretrain
        do_pretrain=str2bool(args.do_pretrain),
        max_epochs_pretrain=int(args.max_epochs_pretrain),
        pre_mask_p_train=float(args.pre_mask_p_train),
        earlystop_pretrain_patience=int(args.earlystop_pretrain_patience),

        # downstream
        max_epochs_train=int(args.max_epochs_train),
        earlystop_train_patience=int(args.earlystop_train_patience),

    )