import os
import sys
import json
import subprocess
from itertools import product
from typing import Dict, List, Tuple

# -----------------------------
# Target mapping
# -----------------------------
def default_target_for_source(source_csv: str) -> str:
    """
    Same-frequency domain shift 기본:
      ETTh1 <-> ETTh2
      ETTm1 <-> ETTm2
    """
    name = os.path.basename(source_csv)
    mapping = {
        "ETTh1.csv": "ETTh2.csv",
        "ETTh2.csv": "ETTh1.csv",
        "ETTm1.csv": "ETTm2.csv",
        "ETTm2.csv": "ETTm1.csv",
    }
    tgt = mapping.get(name, "")
    return os.path.join(os.path.dirname(source_csv), tgt) if tgt else ""


def maybe_cross_freq_target(source_csv: str) -> str:
    """
    Cross-frequency domain shift 옵션:
      ETTh1 -> ETTm1, ETTh2 -> ETTm2
      ETTm1 -> ETTh1, ETTm2 -> ETTh2
    """
    name = os.path.basename(source_csv)
    mapping = {
        "ETTh1.csv": "ETTm1.csv",
        "ETTh2.csv": "ETTm2.csv",
        "ETTm1.csv": "ETTh1.csv",
        "ETTm2.csv": "ETTh2.csv",
    }
    tgt = mapping.get(name, "")
    return os.path.join(os.path.dirname(source_csv), tgt) if tgt else ""


# -----------------------------
# Run one job (subprocess)
# -----------------------------
def run_one_ett(
    *,
    source_csv: str,
    target_csv: str = "",          # 없으면 target 평가 스킵
    model_name: str,
    model_cfg: dict,
    pred_len: int,
    lookback: int = 96,
    features: str = "M",           # "M" or "S"
    target_col: str = "OT",
    official_split: bool = True,
    batch_size: int = 32,
    max_epochs_train: int = 20,
    seed: int = 9871,
    num_workers: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    log_to_wandb: bool = True,
    eval_only: bool = False,
    pretrained_path: str = "",     # ETT는 보통 안 쓰지만 arg는 유지
    downstream_path: str = "",
    wandb_run_name: str = "",
    output_directory: str = "./outputs_etth",
):
    # --- fixed for ETT forecast
    task_name = "ett"
    task_type = "forecast"

    # --- force head output dim
    cfg = dict(model_cfg)  # copy
    cfg["n_output"] = int(pred_len)

    # --- tags
    src_tag = os.path.basename(source_csv).replace(".csv", "")
    tgt_tag = os.path.basename(target_csv).replace(".csv", "") if target_csv else "noT"

    if not wandb_run_name:
        wandb_run_name = f"{model_name}_{src_tag}_to_{tgt_tag}_lb{lookback}_pl{pred_len}_{features}_{target_col}"

    if not downstream_path:
        downstream_path = f"./rd_models/{wandb_run_name}.pt"

    # NOTE:
    # - --source 는 'ett'로 고정 (기존 ICU 파이프라인과 의미 충돌 방지)
    # - 실제 csv는 --ett_csv_path / --ett_target_csv_path로 전달
    cmd = [
        sys.executable,
        "model_run_new.py",

        "--source", "ett",
        "--target_domain", (target_csv if target_csv else ""),   # 기존 arg 요구조건 때문에 유지

        "--model_name", model_name,
        "--model_cfg_json", json.dumps(cfg),

        "--task_type", task_type,
        "--task_name", task_name,

        "--outcome_cols", "dummy",           # argparse required placeholder
        "--batch_size", str(batch_size),
        "--max_epochs_pretrain", "0",
        "--max_epochs_train", str(max_epochs_train),

        "--pretrained_path", str(pretrained_path),
        "--downstream_path", downstream_path,
        "--wandb_run_name", wandb_run_name,
        "--seed", str(seed),

        "--num_workers", str(num_workers),
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--log_to_wandb", ("True" if log_to_wandb else "False"),
        "--eval_only", ("True" if eval_only else "False"),
        "--output_directory", output_directory,

        # ---- ETT-specific ----
        "--ett_csv_path", source_csv,
        "--ett_target_csv_path", (target_csv if target_csv else ""),
        "--ett_lookback", str(lookback),
        "--ett_pred_len", str(pred_len),
        "--ett_features", features,
        "--ett_target_col", target_col,
        "--ett_official_split", ("True" if official_split else "False"),
    ]

    print("\n" + "=" * 80)
    print("[RUN]", wandb_run_name)
    print("  source:", source_csv)
    print("  target:", (target_csv if target_csv else "(none)"))
    print("  model :", model_name, cfg)
    print("=" * 80 + "\n")

    subprocess.run(cmd, check=True)


# -----------------------------
# Task settings generator
# -----------------------------
def make_task_settings_for_ett() -> Dict[str, Dict]:
    """
    pred_len에 따른 n_output은 run_one_ett에서 강제 덮어씀.
    여기서는 embed_dim / surprise args 등만 정의.
    """
    settings = {}

    settings["STRaTS"] = dict(
        model_name="strats",
        model_cfg={"embed_dim": 32},
    )

    settings["Surp_t95"] = dict(
        model_name="surprise",
        model_cfg={
            "embed_dim": 32,
            "use_surprise": True,
            "surprise_args": {"sim_threshold": 0.95, "direction": "past"},
        },
    )

    settings["SurpVT_t95"] = dict(
        model_name="surprise_vt",
        model_cfg={
            "embed_dim": 32,
            "use_surprise": True,
            "vt_mask_args": {"sim_threshold": 0.95, "direction": "past"},
        },
    )

    settings["SurpVTTG_t80"] = dict(
        model_name="surprise_vt",
        model_cfg={
            "embed_dim": 32,
            "use_surprise": False,
            "use_timegap_surprise": True,
            "vt_mask_args": {"sim_threshold": 0.80, "direction": "past"},
        },
    )

    return settings


# -----------------------------
# Main grid runner
# -----------------------------
if __name__ == "__main__":
    # ---- datasets ----
    source_csv_list = [
        "./data/ETTh1.csv",
        "./data/ETTh2.csv",
        "./data/ETTm1.csv",
        "./data/ETTm2.csv",
    ]

    # ---- experiment knobs ----
    pred_len_list = [96, 
                     192, 
                     336, 
                     720
                     ]
    lookback_list = [96]
    features_list = ["M"]          # ["M","S"]
    target_col = "OT"

    # ---- runtime ----
    batch_size = 32
    max_epochs_train = 30
    seed = 9871
    num_workers = 0
    lr = 1e-3
    weight_decay = 1e-2
    log_to_wandb = True
    eval_only = False

    # ---- domain shift mode ----
    use_target = True              # False면 target 평가 아예 스킵
    cross_freq = False             # True면 hour<->minute cross-domain

    # ---- tasks/models ----
    task_settings = make_task_settings_for_ett()

    for source_csv, pred_len, lookback, features in product(
        source_csv_list, pred_len_list, lookback_list, features_list
    ):
        # choose target
        if not use_target:
            target_csv = ""
        else:
            target_csv = maybe_cross_freq_target(source_csv) if cross_freq else default_target_for_source(source_csv)

        # sanity: skip if invalid / same file
        if target_csv:
            # normalize paths for comparison
            src_abs = os.path.abspath(source_csv)
            tgt_abs = os.path.abspath(target_csv)
            if src_abs == tgt_abs:
                print(f"[SKIP] source==target: {source_csv}")
                continue
            if not os.path.exists(target_csv):
                print(f"[SKIP] target missing: {target_csv}")
                target_csv = ""  # gracefully skip target

        # run all models for this setting
        src_tag = os.path.basename(source_csv).replace(".csv", "")
        tgt_tag = os.path.basename(target_csv).replace(".csv", "") if target_csv else "noT"

        for run_key, conf in task_settings.items():
            wandb_run_name = f"{run_key}_{src_tag}_to_{tgt_tag}_lb{lookback}_pl{pred_len}_{features}_{target_col}"

            run_one_ett(
                source_csv=source_csv,
                target_csv=target_csv,
                model_name=conf["model_name"],
                model_cfg=conf["model_cfg"],
                pred_len=pred_len,
                lookback=lookback,
                features=features,
                target_col=target_col,
                batch_size=batch_size,
                max_epochs_train=max_epochs_train,
                seed=seed,
                num_workers=num_workers,
                lr=lr,
                weight_decay=weight_decay,
                log_to_wandb=log_to_wandb,
                eval_only=eval_only,
                wandb_run_name=wandb_run_name,
                output_directory="./outputs_etth",
            )
