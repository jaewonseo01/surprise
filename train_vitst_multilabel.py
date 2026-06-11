# train_vitst_multilabel.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import argparse
import time
import numpy as np
from typing import Dict, List, Tuple, Optional

import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from datasets import Dataset, Image as HFImage
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from sklearn.metrics import roc_auc_score, average_precision_score

from modeling_vit import ViTForImageClassification
from modeling_swin import SwinForImageClassification


# -------------------------
# Utils
# -------------------------
def _safe_metric_key(name: str) -> str:
    return str(name).strip().replace(" ", "_").replace("/", "_")


def _ensure_list_str(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, np.ndarray):
        return [str(v) for v in x.tolist()]
    return [str(v) for v in list(x)]


def _source_bundle_dir(
    images_root: str,
    source: str,
    source_split: int,
    *,
    use_source_aug: bool = False,
) -> str:
    """
    ConstructImage.py writes under:
      images_root/split{source_split}/source_<source>_images/
      images_root/split{source_split}/source_<source>_aug_images/
    """
    folder = f"source_{source}_aug_images" if use_source_aug else f"source_{source}_images"
    return os.path.join(images_root, f"split{source_split}", folder)


def _target_meta_split_name(use_target_aug: bool) -> str:
    return "test_aug" if use_target_aug else "test"


def _target_image_subdir(target: str, use_target_aug: bool) -> str:
    return f"{target}_test_aug_images" if use_target_aug else f"{target}_test_images"


def _source_image_subdir(source: str, split_name: str) -> str:
    return f"{source}_{split_name}_images"


def _run_aug_tag(
    use_source_aug: bool,
    source_aug_suffix: str,
    use_target_aug: bool,
    target_aug_suffix: str,
) -> str:
    src = f"srcAug-{source_aug_suffix}" if use_source_aug else "srcOrig"
    tgt = f"tgtAug-{target_aug_suffix}" if use_target_aug else "tgtOrig"
    return f"{src}__{tgt}"


# -------------------------
# Metadata + dataset loading
# -------------------------
def load_metadata(source_bundle_dir: str, domain: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads ImageDict_list_<domain>_<split>.npy from the given bundle dir.
    Examples:
      ImageDict_list_mimic_train.npy
      ImageDict_list_eicu_test.npy
      ImageDict_list_eicu_test_aug.npy
    """
    p_path = os.path.join(source_bundle_dir, f"ImageDict_list_{domain}_{split}.npy")
    if not os.path.exists(p_path):
        raise FileNotFoundError(f"Missing metadata: {p_path}")

    Pdict_list = np.load(p_path, allow_pickle=True)
    if len(Pdict_list) == 0:
        raise ValueError(f"Empty metadata: {p_path}")

    labels = []
    for i, d in enumerate(Pdict_list):
        if "label" not in d:
            raise KeyError(f'Missing key "label" in metadata item {i} from {p_path}')
        labels.append(d["label"])

    if isinstance(labels[0], (list, tuple, np.ndarray)):
        Y = np.asarray(labels, dtype=np.float32)
        if Y.ndim != 2:
            raise ValueError(f"Expected multilabel Y to be 2D [N,C], got shape={Y.shape} from {p_path}")
    else:
        Y = np.asarray(labels, dtype=np.float32)
        if Y.ndim != 1:
            raise ValueError(f"Expected single-label Y to be 1D [N], got shape={Y.shape} from {p_path}")

    return Pdict_list, Y


def load_outcome_cols(source_bundle_dir: str, source_domain: str) -> List[str]:
    """
    ConstructImage.py writes:
      source_bundle_dir/outcome_cols_<domain>.json
    We use SOURCE domain label names for head definition.
    """
    jpath = os.path.join(source_bundle_dir, f"outcome_cols_{source_domain}.json")
    if os.path.exists(jpath):
        with open(jpath, "r", encoding="utf-8") as f:
            cols = json.load(f)
        cols = _ensure_list_str(cols)
        if len(cols) == 0:
            raise ValueError(f"Empty outcome cols in {jpath}")
        return cols

    npypath = os.path.join(source_bundle_dir, f"outcome_cols_{source_domain}.npy")
    if os.path.exists(npypath):
        cols = np.load(npypath, allow_pickle=True)
        cols = _ensure_list_str(cols)
        if len(cols) == 0:
            raise ValueError(f"Empty outcome cols in {npypath}")
        return cols

    raise FileNotFoundError(
        f"Missing outcome column names under source bundle dir.\n"
        f"Tried:\n- {jpath}\n- {npypath}\n"
        f"Fix: ensure ConstructImage.py wrote outcome_cols_<domain>.json"
    )


def make_image_dataset(
    Pdict_list: np.ndarray,
    Y: np.ndarray,
    images_root: str,
    rel_images_subdir: str,
) -> Dataset:
    if len(Pdict_list) != Y.shape[0]:
        raise ValueError(f"N mismatch: Pdict_list={len(Pdict_list)} vs Y={Y.shape[0]}")

    image_paths: List[str] = []
    labels: List[List[float]] = []

    for i, d in enumerate(Pdict_list):
        pid = d["id"]
        img_path = os.path.join(images_root, rel_images_subdir, f"{pid}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")

        image_paths.append(img_path)
        labels.append(np.asarray(Y[i], dtype=np.float32).tolist())

    ds = Dataset.from_dict({"image": image_paths, "labels": labels})
    ds = ds.cast_column("image", HFImage())
    return ds


# -------------------------
# Metrics
# -------------------------
def multilabel_metrics_from_logits(
    logits: np.ndarray,
    y_true: np.ndarray,
    label_names: List[str],
    prefix: str = "",
) -> Dict[str, float]:
    if logits.ndim != 2 or y_true.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got logits={logits.shape}, y_true={y_true.shape}")
    if logits.shape != y_true.shape:
        raise ValueError(f"Shape mismatch logits={logits.shape} vs y_true={y_true.shape}")

    C = y_true.shape[1]
    if len(label_names) != C:
        raise ValueError(f"label_names length {len(label_names)} != #classes {C}")

    probs = 1.0 / (1.0 + np.exp(-logits))

    per_auc = []
    per_ap = []
    out: Dict[str, float] = {}

    for c in range(C):
        yt = y_true[:, c].astype(np.int32)
        yp = probs[:, c].astype(np.float64)

        if yt.min() == yt.max():
            out[f"{prefix}auroc/{label_names[c]}"] = float("nan")
            out[f"{prefix}auprc/{label_names[c]}"] = float("nan")
            continue

        auc = roc_auc_score(yt, yp)
        ap = average_precision_score(yt, yp)

        out[f"{prefix}auroc/{label_names[c]}"] = float(auc)
        out[f"{prefix}auprc/{label_names[c]}"] = float(ap)

        per_auc.append(auc)
        per_ap.append(ap)

    out[f"{prefix}auroc_macro_valid"] = float(np.mean(per_auc)) if len(per_auc) else float("nan")
    out[f"{prefix}auprc_macro_valid"] = float(np.mean(per_ap)) if len(per_ap) else float("nan")
    out[f"{prefix}valid_labels"] = float(len(per_auc))
    return out


# -------------------------
# HF Trainer plumbing
# -------------------------
def build_transforms(image_processor):
    size = image_processor.size
    if isinstance(size, dict):
        h, w = size.get("height", 224), size.get("width", 224)
        resize = Resize((h, w))
    elif isinstance(size, int):
        resize = Resize((size, size))
    else:
        resize = Resize((224, 224))

    mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
    std = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])

    train_tf = Compose([resize, ToTensor(), Normalize(mean=mean, std=std)])
    eval_tf = Compose([resize, ToTensor(), Normalize(mean=mean, std=std)])
    return train_tf, eval_tf


def set_dataset_transforms(ds: Dataset, tfm):
    def _preprocess(batch):
        batch["pixel_values"] = [tfm(img.convert("RGB")) for img in batch["image"]]
        return batch
    return ds.with_transform(_preprocess)


def collate_fn(examples):
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    labels = torch.tensor([ex["labels"] for ex in examples], dtype=torch.float32)
    return {"pixel_values": pixel_values, "labels": labels}


# -------------------------
# Training / evaluation
# -------------------------
def run_train_eval(
    *,
    source: str,
    target: str,
    model_kind: str,
    model_path: str,

    processed_root: str,           # kept for bookkeeping
    source_split: int,
    images_root: str,

    use_source_aug: bool = False,
    source_aug_suffix: str = "aug",
    use_target_aug: bool = False,
    target_aug_suffix: str = "aug",

    output_root: str = "./baseline_models_vitst",
    seed: int = 9871,

    num_train_epochs: int = 20,
    per_device_train_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 4,
    warmup_ratio: float = 0.1,

    logging_steps: int = 50,
    save_steps: int = 200,
    save_total_limit: int = 1,
    eval_steps: int = 200,
    early_stopping_patience: int = 5,

    use_wandb: bool = False,
    wandb_project: str = "vitst_baseline",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
):
    if source == target:
        raise ValueError("source and target must be different")
    if source not in {"mimic", "eicu"} or target not in {"mimic", "eicu"}:
        raise ValueError("source/target must be in {mimic,eicu}")
    if source_split not in {1, 2, 3, 4, 5}:
        raise ValueError("source_split must be in {1..5}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    aug_tag = _run_aug_tag(
        use_source_aug=use_source_aug,
        source_aug_suffix=source_aug_suffix,
        use_target_aug=use_target_aug,
        target_aug_suffix=target_aug_suffix,
    )

    # Source bundle dir contains:
    # - source train/valid/test from split{source_split}
    # - target test metadata/images copied there by ConstructImage.py
    source_bundle_dir = _source_bundle_dir(
        images_root,
        source,
        source_split,
        use_source_aug=use_source_aug,
    )

    label_names = load_outcome_cols(source_bundle_dir, source)

    # metadata names
    target_meta_split = _target_meta_split_name(use_target_aug)

    # source metadata
    Ptr, Ytr = load_metadata(source_bundle_dir, source, "train")
    Pva, Yva = load_metadata(source_bundle_dir, source, "valid")
    Pte_s, Yte_s = load_metadata(source_bundle_dir, source, "test")

    # target test metadata is stored under the same source bundle dir
    Pte_t, Yte_t = load_metadata(source_bundle_dir, target, target_meta_split)

    # image subdirs relative to images_root
    source_bundle_rel = os.path.relpath(source_bundle_dir, images_root)

    ds_train = make_image_dataset(
        Ptr, Ytr, images_root,
        os.path.join(source_bundle_rel, _source_image_subdir(source, "train"))
    )
    ds_val = make_image_dataset(
        Pva, Yva, images_root,
        os.path.join(source_bundle_rel, _source_image_subdir(source, "valid"))
    )
    ds_test_s = make_image_dataset(
        Pte_s, Yte_s, images_root,
        os.path.join(source_bundle_rel, _source_image_subdir(source, "test"))
    )
    ds_test_t = make_image_dataset(
        Pte_t, Yte_t, images_root,
        os.path.join(source_bundle_rel, _target_image_subdir(target, use_target_aug))
    )

    C = Ytr.shape[1]
    if len(label_names) != C:
        raise ValueError(
            f"#labels mismatch: outcome_cols({len(label_names)}) vs Ytr.shape[1]({C}).\n"
            f"Fix: ensure ConstructImage.py wrote outcome_cols_{source}.json matching labels."
        )

    if model_kind == "vit":
        model_loader = ViTForImageClassification
    elif model_kind == "swin":
        model_loader = SwinForImageClassification
    else:
        raise ValueError(f"Unknown model_kind={model_kind} (supported: vit, swin)")

    config = AutoConfig.from_pretrained(model_path, num_labels=C, ignore_mismatched_sizes=True)
    config.problem_type = "multi_label_classification"

    model = model_loader.from_pretrained(
        model_path,
        config=config,
        ignore_mismatched_sizes=True,
    )

    image_processor = AutoImageProcessor.from_pretrained(model_path)
    train_tf, eval_tf = build_transforms(image_processor)

    ds_train = set_dataset_transforms(ds_train, train_tf)
    ds_val = set_dataset_transforms(ds_val, eval_tf)
    ds_test_s = set_dataset_transforms(ds_test_s, eval_tf)
    ds_test_t = set_dataset_transforms(ds_test_t, eval_tf)

    print("N_train =", len(ds_train))
    print("N_val   =", len(ds_val))
    print("N_test_s=", len(ds_test_s))
    print("N_test_t=", len(ds_test_t), f"(target test from split1, target_aug={use_target_aug})")

    default_run_name = f"ViTST_{model_kind}_{source}_to_{target}_split{source_split}__{aug_tag}"
    run_name = wandb_run_name or default_run_name
    out_dir = os.path.join(output_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    wb = None
    if use_wandb:
        import wandb
        wb = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config={
                "model": model_kind,
                "source": source,
                "target": target,
                "source_split": source_split,
                "target_test_split_fixed": 1,
                "seed": seed,
                "epochs": num_train_epochs,
                "train_batch": per_device_train_batch_size,
                "eval_batch": per_device_eval_batch_size,
                "lr": learning_rate,
                "grad_accum": gradient_accumulation_steps,
                "warmup_ratio": warmup_ratio,
                "model_path": model_path,
                "num_labels": C,
                "label_names": _ensure_list_str(label_names),
                "processed_root": processed_root,
                "images_root": images_root,
                "source_bundle_dir": source_bundle_dir,
                "use_source_aug": use_source_aug,
                "source_aug_suffix": source_aug_suffix,
                "use_target_aug": use_target_aug,
                "target_aug_suffix": target_aug_suffix,
                "aug_tag": aug_tag,
                "N_train": len(ds_train),
                "N_val": len(ds_val),
                "N_test_source": len(ds_test_s),
                "N_test_target": len(ds_test_t),
            },
        )
        wandb.watch(model, log="gradients", log_freq=100)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        labels = labels.astype(np.float32)
        metrics = multilabel_metrics_from_logits(
            logits=logits,
            y_true=labels,
            label_names=_ensure_list_str(label_names),
            prefix="eval/",
        )
        metrics["eval/auprc_macro"] = metrics["eval/auprc_macro_valid"]
        metrics["eval/auroc_macro"] = metrics["eval/auroc_macro_valid"]
        return metrics

    training_args = TrainingArguments(
        output_dir=out_dir,
        seed=seed,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,

        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,

        load_best_model_at_end=True,
        metric_for_best_model="eval/auprc_macro",
        greater_is_better=True,

        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        callbacks=callbacks,
    )

    t0 = time.time()
    trainer.train()
    elapsed_min = (time.time() - t0) / 60.0
    print(f"✅ Train done. elapsed={elapsed_min:.2f} min")

    pred_s = trainer.predict(ds_test_s)
    metrics_s = multilabel_metrics_from_logits(
        logits=pred_s.predictions,
        y_true=pred_s.label_ids,
        label_names=_ensure_list_str(label_names),
        prefix="test/source/",
    )

    pred_t = trainer.predict(ds_test_t)
    metrics_t = multilabel_metrics_from_logits(
        logits=pred_t.predictions,
        y_true=pred_t.label_ids,
        label_names=_ensure_list_str(label_names),
        prefix="test/target/",
    )

    print(
        f"[SOURCE TEST] AUROC(macro_valid)={metrics_s['test/source/auroc_macro_valid']:.4f} "
        f"AUPRC(macro_valid)={metrics_s['test/source/auprc_macro_valid']:.4f} "
        f"(valid_labels={int(metrics_s['test/source/valid_labels'])})"
    )
    print(
        f"[TARGET TEST split1] AUROC(macro_valid)={metrics_t['test/target/auroc_macro_valid']:.4f} "
        f"AUPRC(macro_valid)={metrics_t['test/target/auprc_macro_valid']:.4f} "
        f"(valid_labels={int(metrics_t['test/target/valid_labels'])})"
    )

    if wb is not None:
        log_dict = {
            "test/source_auroc": float(metrics_s["test/source/auroc_macro_valid"]),
            "test/source_auprc": float(metrics_s["test/source/auprc_macro_valid"]),
            "test/source_valid_labels": float(metrics_s["test/source/valid_labels"]),
            "test/target_auroc": float(metrics_t["test/target/auroc_macro_valid"]),
            "test/target_auprc": float(metrics_t["test/target/auprc_macro_valid"]),
            "test/target_valid_labels": float(metrics_t["test/target/valid_labels"]),
        }

        for name in _ensure_list_str(label_names):
            key = _safe_metric_key(name)

            s_auc = metrics_s.get(f"test/source/auroc/{name}", float("nan"))
            s_ap = metrics_s.get(f"test/source/auprc/{name}", float("nan"))
            t_auc = metrics_t.get(f"test/target/auroc/{name}", float("nan"))
            t_ap = metrics_t.get(f"test/target/auprc/{name}", float("nan"))

            if np.isfinite(s_auc):
                log_dict[f"test/source_auroc_{key}"] = float(s_auc)
            if np.isfinite(s_ap):
                log_dict[f"test/source_auprc_{key}"] = float(s_ap)
            if np.isfinite(t_auc):
                log_dict[f"test/target_auroc_{key}"] = float(t_auc)
            if np.isfinite(t_ap):
                log_dict[f"test/target_auprc_{key}"] = float(t_ap)

        wb.log(log_dict)
        wb.finish()

    report = {
        "run_name": run_name,
        "source": source,
        "target": target,
        "source_split": int(source_split),
        "target_test_split_fixed": 1,
        "model_kind": model_kind,
        "model_path": model_path,
        "label_names": _ensure_list_str(label_names),
        "use_source_aug": bool(use_source_aug),
        "source_aug_suffix": source_aug_suffix,
        "use_target_aug": bool(use_target_aug),
        "target_aug_suffix": target_aug_suffix,
        "aug_tag": aug_tag,
        "metrics": {**metrics_s, **metrics_t},
        "config": {
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "learning_rate": learning_rate,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_ratio": warmup_ratio,
        },
        "paths": {
            "processed_root": processed_root,
            "images_root": images_root,
            "source_bundle_dir": source_bundle_dir,
        },
    }
    with open(os.path.join(out_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Saved metrics to {os.path.join(out_dir, 'final_metrics.json')}")
    print(f"✅ Checkpoints/models saved under {out_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", type=str, default="mimic", choices=["mimic", "eicu"])
    parser.add_argument("--target", type=str, default=None, choices=["mimic", "eicu"])
    parser.add_argument("--model_kind", type=str, default="vit", choices=["vit", "swin"])

    parser.add_argument("--model_path", type=str, default=None)

    parser.add_argument(
        "--processed_root",
        type=str,
        default="./data_rd_template",
        help="Kept for bookkeeping. Usually same root family as image bundles."
    )
    parser.add_argument("--source_split", type=int, default=1, choices=[1, 2, 3, 4, 5])

    parser.add_argument(
        "--images_root",
        type=str,
        default=None,
        help="Root containing split{1..5}/source_<source>[_aug]_images/"
    )

    # aug
    parser.add_argument("--use_source_aug", action="store_true")
    parser.add_argument("--source_aug_suffix", type=str, default="aug")
    parser.add_argument("--use_target_aug", action="store_true")
    parser.add_argument("--target_aug_suffix", type=str, default="aug")

    # training hyperparams
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train_batch", type=int, default=32)
    parser.add_argument("--eval_batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # logging/saving
    parser.add_argument("--output_root", type=str, default="./baseline_models_vitst")
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--early_stop", type=int, default=7)

    parser.add_argument("--seed", type=int, default=9871)

    # wandb
    parser.add_argument("--use_wandb", action="store_false")
    parser.add_argument("--wandb_project", type=str, default="Baselines")
    parser.add_argument("--wandb_entity", type=str, default="jwseo118-korea-university")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    if args.target is None:
        args.target = "eicu" if args.source == "mimic" else "mimic"
    if args.target == args.source:
        raise ValueError("target must be different from source")

    if args.model_path is None:
        if args.model_kind == "vit":
            args.model_path = "google/vit-base-patch16-224-in21k"
        else:
            args.model_path = "microsoft/swin-tiny-patch4-window7-224"

    if args.images_root is None:
        args.images_root = args.processed_root

    run_train_eval(
        source=args.source,
        target=args.target,
        model_kind=args.model_kind,
        model_path=args.model_path,

        processed_root=args.processed_root,
        source_split=args.source_split,
        images_root=args.images_root,

        use_source_aug=args.use_source_aug,
        source_aug_suffix=args.source_aug_suffix,
        use_target_aug=args.use_target_aug,
        target_aug_suffix=args.target_aug_suffix,

        output_root=args.output_root,
        seed=args.seed,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,

        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        early_stopping_patience=args.early_stop,

        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()