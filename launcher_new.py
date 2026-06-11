import sys
import subprocess
from itertools import product
import json
import pandas as pd

def run_one(
    source,
    model_name,
    model_cfg,
    task_type,
    task_name,
    outcome_cols,
    batch_size="8",
    max_epochs_pretrain="0",
    max_epochs_train="30",
    seed="9871",
    pretrained_path=None,
    wandb_run_name=None,
    aug="False",
    ep=1
):
    """
    source, model_name, ... 받아서 runner_pipeline.py 를 subprocess로 실행
    """

    # 바인딩 1: target_domain (= other ICU dataset)
    if source == "mimic":
        target_domain = "eicu"
    elif source == "eicu":
        target_domain = "mimic"
    else:
        raise ValueError(f"Unknown source '{source}' for target binding")
    print(f"Pretrain path is {pretrained_path}")
    # 바인딩 2: pretrained_path
    if pretrained_path is None:
        print("No pretrain path")
        pretrain_path_dict={
            'strats': "./rd_models/bestSTraTST2V_model",
            'surprise': "./rd_models/bestSurpSTraTSVT_model",
            'surprise_vt': "./rd_models/bestseparateSurpTGT2V_model"
            }
        pretrained_path = f"{pretrain_path_dict[model_name]}_{source}.pt"


    if wandb_run_name is None:
        # wandb_run_name 자동
        wandb_run_name = f"{model_name}_{source}_{task_name}"

    # 바인딩 3: downstream_path
    # downstream_path = f"./rd_models/ckpts/{wandb_run_name}/{wandb_run_name}_epoch{ep}.ckpt"
    # wandb_run_name = f'{wandb_run_name}_epoch{ep}'
    downstream_path = f"./rd_models/{wandb_run_name}.pt"


    # outcome_cols를 콤마로 이어붙여 문자열로 넘김
    outcome_cols_str = ",".join(outcome_cols)

    # model_cfg dict을 json string으로 넘김
    model_cfg_json = json.dumps(model_cfg)

    cmd = [
        sys.executable,
        "model_run_new.py",                   # <- 아래 (2)번 코드 파일 이름
        "--source", source,
        "--target_domain", target_domain,
        "--model_name", model_name,
        "--model_cfg_json", model_cfg_json,
        "--task_type", task_type,
        "--task_name", task_name,
        "--outcome_cols", outcome_cols_str,
        "--batch_size", str(batch_size),
        "--max_epochs_pretrain", str(max_epochs_pretrain),
        "--max_epochs_train", str(max_epochs_train),
        "--pretrained_path", pretrained_path,
        "--downstream_path", downstream_path,
        "--wandb_run_name", wandb_run_name,
        "--seed", str(seed),
        "--eval_only", "False",           # <- 핵심
        "--aug", aug,
        "--log_to_wandb", "True",
        "--project_name", "Baselines"
    ]

    # 그냥 콘솔에 그대로 흘려보내기
    subprocess.run(cmd, check=True)


if __name__ == "__main__":

    # ============================
    # 너가 실험하고 싶은 설정들 정의
    # ============================
    ocs_df = pd.read_feather('./data/mimic_outcomes_train.feather')
    
    ocs = ocs_df.columns.values.tolist()
    ocs.remove('pid')
    ocs.remove('aki_label')
    ocs.remove('cf_label')
    print(ocs)
    source_list = [
        #"mimic",
        "eicu",
    ]

    # task_settings={
    #     # Baseline STraTS
    #     # "Baseline": {
    #     #     "model_name": "strats",
    #     #     "task_type": "multilabel",
    #     #     "outcome_cols":ocs,
    #     #     "model_cfg": {"embed_dim": 32, "n_output": 28},
    #     #     "pretrained_path": "./rd_models/bestSTraTS_model_mimic.pt",
    #     #     "wandb_run_name": "strats_mimic_multi",                          
    #     # },
    #     # surprise sim 95
    #     "Surp": {
    #         "model_name": "surprise",
    #         "task_type": "multilabel",
    #         "outcome_cols":ocs,
    #         "model_cfg": {"embed_dim": 32, "n_output": 28, "use_surprise": True,
    #                       "surprise_args" : {"sim_threshold" : 0.95, "direction" : "past"}},
    #         "pretrained_path": "./rd_models/bestSurpSTraTS_model_mimic.pt",
    #         "wandb_run_name": "surprise_mimic_multi",                          
    #     },
    #     "SurpVT": {
    #         "model_name": "surprise_vt",
    #         "task_type": "multilabel",
    #         "outcome_cols":ocs,
    #         "model_cfg": {"embed_dim": 32, "n_output": 28, "use_surprise": True,
    #                       "vt_mask_args" : {"sim_threshold" : 0.95, "direction" : "past"}},
    #         "pretrained_path": "./rd_models/bestSurpSTraTSVT_model_mimic.pt",
    #         "wandb_run_name": "surprisevt_mimic_multi",                          
    #     },
    #     "SurpVTTG": {
    #         "model_name": "surprise_vt",           
    #         "task_type": "multilabel",
    #         "outcome_cols":ocs,
    #         "model_cfg": {"embed_dim": 32, "n_output": 28, "use_surprise": False, "use_timegap_surprise" : True,
    #                       "vt_mask_args" : {"sim_threshold" : 0.80, "direction" : "past"}},
    #         "pretrained_path": "./rd_models/bestSurpSTraTSVTTG80_model_mimic.pt",
    #         "wandb_run_name": "surprisevttg80_mimic_multi",                          
    #     },
    #     }

    task_settings = {}

    task_settings['Baseline'] = {
        "model_name": "strats",
        "task_type": "multilabel",
        "outcome_cols": ocs,
        "model_cfg": {
            "embed_dim": 32,
            "n_output": 28,
        },
        "pretrained_path": f"./rd_models/bestSTraTS_model_eicu.pt",
        "wandb_run_name": f"STraTS_eicu_multi",
    }

    # -------------------------
    # threshold grids
    # -------------------------
    surp_thresholds = [0.90]
    surp_vt_thresholds = [1.40]
    surp_vttg_thresholds = [0.80]
    eps = [1, 5, 15]

    # -------------------------
    # Surp (shared value)
    # -------------------------
    for thr in surp_thresholds:
        key = f"Surp_t{thr}"
        task_settings[key] = {
            "model_name": "surprise",
            "task_type": "multilabel",
            "outcome_cols": ocs,
            "model_cfg": {
                "embed_dim": 32,
                "n_output": 28,
                "use_surprise": True,
                "surprise_args": {
                    "sim_threshold": thr,
                    "direction": "past",
                },
            },
            "pretrained_path": f"./rd_models/bestSurpSTraTS_t{int(100*thr)}_model_eicu.pt",
            "wandb_run_name": f"Surprise_eicu_multi_t{int(100*thr)}",
        }

    # # -------------------------
    # # SurpVT (per-variable VT)
    # # -------------------------
    for thr in surp_vt_thresholds:
        key = f"SurpVT_t{thr}"
        task_settings[key] = {
            "model_name": "surprise_vt",
            "task_type": "multilabel",
            "outcome_cols": ocs,
            "model_cfg": {
                "embed_dim": 32,
                "n_output": 28,
                "use_surprise": True,
                "vt_mask_args": {
                    "sim_threshold": thr,
                    "direction": "past",
                },
            },
            "pretrained_path": f"./rd_models/bestSurpSTraTSVT_t{int(100*thr)}_model_eicu.pt",
            "wandb_run_name": f"SurpriseVT_eicu_multi_t{int(100*thr)}",
        }

    # -------------------------
    # SurpVTTG (time-gap based)
    # -------------------------
    for thr in surp_vttg_thresholds:
        key = f"SurpVTTG_t{thr}"
        task_settings[key] = {
            "model_name": "surprise_vt",
            "task_type": "multilabel",
            "outcome_cols": ocs,
            "model_cfg": {
                "embed_dim": 32,
                "n_output": 28,
                "use_surprise": False,
                "use_timegap_surprise": True,
                "vt_mask_args": {
                    "sim_threshold": thr,
                    "direction": "past",
                },
            },
            "pretrained_path": f"./rd_models/bestSurpSTraTSVTTG_t{int(100*thr)}_model_eicu.pt",
            "wandb_run_name": f"SurpriseVTTG_eicu_multi_t{int(100*thr)}",
    }

    # 기타 고정 하이퍼
    batch_size = 8
    max_epochs_pretrain = 50
    max_epochs_train = 50
    seed = 9871

    # ===== 전체 루프 =====
    for source in source_list:
        #for ep in eps:
        for run_type, tconf in task_settings.items():
            run_one(
                source=source,
                model_name=tconf["model_name"],
                model_cfg=tconf["model_cfg"],
                task_type=tconf["task_type"],
                task_name="multi",
                outcome_cols=tconf["outcome_cols"],
                batch_size=batch_size,
                max_epochs_pretrain=max_epochs_pretrain,
                max_epochs_train=max_epochs_train,
                seed=seed,
                pretrained_path=tconf["pretrained_path"],
                wandb_run_name=tconf["wandb_run_name"],
                aug="False",
                # ep=ep
            )
