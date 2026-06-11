import sys
import subprocess
import json
import pandas as pd
from pathlib import Path
from datetime import datetime


def run_one_p12p19(
    *,
    dataset: str,  # "P12" or "P19"
    base_path: str,
    split_path_pattern: str,  # e.g. "/splits/phy19_split{}_new.npy"
    num_splits: int = 5,
    model_name: str = "surprise",
    model_cfg: dict | None = None,
    task_type: str = "binary",
    task_name: str = "p12p19",
    outcome_cols=("mortality",),
    batch_size: int = 128,
    max_epochs_pretrain: int = 20,
    max_epochs_train: int = 20,
    seed: int = 9871,
    pretrained_path: str | None = None,
    wandb_run_name: str | None = None,
    log_to_wandb: bool = True,
    num_workers: int = 0,
    max_tokens: int = 4096,
    split_type: str = "random",
    reverse: bool = False,
    baseline: bool = False,
    predictive_label: str = "mortality",
    accelerator: str = "gpu",
    devices: str = "1",
    precision: str = "16-mixed",
    output_dir: str = "./cv_results",
):
    """
    P12/P19 5-split CV runner.
    - model_run_new.py를 subprocess로 실행
    - 내부에서 5-fold를 돌고 mean/std까지 출력하도록 (네가 수정한 pipeline 기준)
    """

    if model_cfg is None:
        model_cfg = {"embed_dim": 32, "n_output": 1}

    # pretrained_path 기본값 (원하면 네 규칙으로 바꿔도 됨)
    if pretrained_path is None:
        # 여기서는 "없음"으로 두는 게 안전 (P12/P19는 보통 별도 pretrain 안 쓰는 경우 많음)
        pretrained_path = ""

    if wandb_run_name is None:
        wandb_run_name = f"{dataset}_cv{num_splits}_{model_name}_pre"

    # outcome cols -> comma string
    outcome_cols_str = ",".join(outcome_cols)

    # model_cfg -> json string
    model_cfg_json = json.dumps(model_cfg)

    # output dir ensure
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "model_run_new.py",
        # ✅ 기존 required args (너 스크립트가 required=True로 묶어둔 것들 때문에 채워줌)
        "--source", "p12p19",               # 의미는 없지만 required라 넣음
        "--target_domain", "none",          # 의미는 없지만 required라 넣음

        "--model_name", model_name,
        "--model_cfg_json", model_cfg_json,

        "--task_type", task_type,
        "--task_name", task_name,
        "--outcome_cols", outcome_cols_str,

        "--batch_size", str(batch_size),
        "--max_epochs_pretrain", str(max_epochs_pretrain),
        "--max_epochs_train", str(max_epochs_train),

        "--pretrained_path", pretrained_path,
        "--downstream_path", "",            # p12p19 CV는 fold마다 저장이 애매해서 일단 빈값

        "--project_name", "Physionet-0204",
        "--wandb_run_name", wandb_run_name,
        "--seed", str(seed),

        "--eval_only", "False",
        "--aug", "False",
        "--log_to_wandb", "True" if log_to_wandb else "False",
        "--output_directory", output_dir,

        "--num_workers", str(num_workers),
        "--lr", str(model_cfg.get("lr", 1e-3)) if "lr" in model_cfg else "1e-3",
        "--weight_decay", str(model_cfg.get("weight_decay", 1e-2)) if "weight_decay" in model_cfg else "1e-2",

        # ✅ P12/P19 전용 args (네가 pipeline에 추가했다고 가정)
        "--p12p19_dataset", dataset,
        "--p12p19_base_path", base_path,
        "--p12p19_split_path_pattern", split_path_pattern,
        "--p12p19_num_splits", str(num_splits),
        "--p12p19_max_tokens", str(max_tokens),
        "--p12p19_split_type", split_type,
        "--p12p19_reverse", "True" if reverse else "False",
        "--p12p19_baseline", "True" if baseline else "False",
        "--p12p19_predictive_label", predictive_label,

        # (선택) trainer 옵션도 args로 받게 해놨으면 여기에 추가 가능
        # "--accelerator", accelerator, ...
    ]

    print("\n[RUN]", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # ===========================
    # 네 실험 그리드 정의
    # ===========================

    # ✅ 데이터셋별 base_path & split 패턴
    # (너가 쓰는 경로 구조에 맞게 수정)
    DATASET_SETTINGS = {
        "P12": {
            "base_path": "./processed_data/P12data",            # 예시
            "split_path_pattern": "/splits/phy12_split{}.npy",
        },
        "P19": {
            "base_path": "./processed_data/P19data",            # 예시
            "split_path_pattern": "/splits/phy19_split{}_new.npy",
        },
    }

    dataset_list = [
        "P19",
        "P12",
    ]

    # ===========================
    # 모델 설정들 (MIMIC 러너 스타일)
    # ===========================
    task_settings = {}

    # Baseline
    task_settings["Baseline"] = {
        "model_name": "strats",
        "task_type": "binary",
        "outcome_cols": ("mortality",),
        "model_cfg": {"embed_dim": 32, "n_output": 1},
        "pretrained_path": "",   # P12/P19에서 쓸 거면 넣어
        "wandb_run_name": None,
    }

    # Surprise (shared)
    task_settings["Surp_t90"] = {
        "model_name": "surprise",
        "task_type": "binary",
        "outcome_cols": ("mortality",),
        "model_cfg": {
            "embed_dim": 32,
            "n_output": 1,
            "use_surprise": True,
            "surprise_args": {"sim_threshold": 0.90, "direction": "past"},
        },
        "pretrained_path": "",
        "wandb_run_name": None,
    }

    # SurpriseVT
    task_settings["SurpVT_t140"] = {
        "model_name": "surprise_vt",
        "task_type": "binary",
        "outcome_cols": ("mortality",),
        "model_cfg": {
            "embed_dim": 32,
            "n_output": 1,
            "use_surprise": True,
            "vt_mask_args": {"sim_threshold": 1.40, "direction": "past"},
        },
        "pretrained_path": "",
        "wandb_run_name": None,
    }

    # SurpVTTG
    task_settings["SurpVTTG_t80"] = {
        "model_name": "surprise_vt",
        "task_type": "binary",
        "outcome_cols": ("mortality",),
        "model_cfg": {
            "embed_dim": 32,
            "n_output": 1,
            "use_surprise": False,
            "use_timegap_surprise": True,
            "vt_mask_args": {"sim_threshold": 0.80, "direction": "past"},
        },
        "pretrained_path": "",
        "wandb_run_name": None,
    }

    # ===========================
    # 고정 하이퍼
    # ===========================
    num_splits = 5
    batch_size = 32
    max_epochs_pretrain = [20]
    max_epochs_train = 20
    seed = 9871
    max_tokens = 4096
    log_to_wandb = True

    # ===== 전체 루프 =====
    for dataset in dataset_list:
        base_path = DATASET_SETTINGS[dataset]["base_path"]
        split_path_pattern = DATASET_SETTINGS[dataset]["split_path_pattern"]

        for run_type, tconf in task_settings.items():
            for pt_epochs in max_epochs_pretrain:
                run_name = tconf.get("wandb_run_name")

                if run_name is None:
                    if pt_epochs == 0:
                        run_name = f"{dataset}_cv{num_splits}_{run_type}"
                    else:
                        run_name = f"{dataset}_cv{num_splits}_{run_type}_pre"
                    

                run_one_p12p19(
                    dataset=dataset,
                    base_path=base_path,
                    split_path_pattern=split_path_pattern,
                    num_splits=num_splits,
                    model_name=tconf["model_name"],
                    model_cfg=tconf["model_cfg"],
                    task_type=tconf["task_type"],
                    task_name="p12p19",
                    outcome_cols=tconf["outcome_cols"],
                    batch_size=batch_size,
                    max_epochs_pretrain=pt_epochs,
                    max_epochs_train=max_epochs_train,
                    seed=seed,
                    pretrained_path=tconf["pretrained_path"],
                    wandb_run_name=run_name,
                    log_to_wandb=log_to_wandb,
                    max_tokens=max_tokens,
                    split_type="random",
                    reverse=False,
                    baseline=False,
                    predictive_label="mortality",
                    output_dir="./cv_results",
                )
