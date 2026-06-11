# train_hymate_models_domainshift.py
#
# - Domain-shift supervised training: train on SOURCE, evaluate on SOURCE test + TARGET test ("target" split)
# - Optional PIPELINE: run pretrain -> finetune in ONE wandb run
# - Save checkpoints to ./baseline_models_hm/{run_name}_pretrain.pt and {run_name}_finetune.pt
# - Final wandb test logging matches Raindrop format:
#   test/source_loss, test/source_auroc, test/source_auprc, test/source_acc,
#   test/target_loss, test/target_auroc, test/target_auprc, test/target_acc,
#   plus per-task test/source_auroc_{task}, test/source_auprc_{task}, etc.
#
# NOTE:
# - For Raindrop-style final logging, runner_hymate.Evaluator MUST return:
#   train_loss (scalar BCE), auroc, auprc, acc, auroc_per, auprc_per, valid_mask
#   (see previous patch to Evaluator).
#
# - PretrainDataset is assumed to be source-only pretraining.
# - Finetune loads from pretrain ckpt using compatible key/shape matching (partial load allowed).

import argparse
import copy
import os
import time
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from runner_hymate import (
    Dataset, PretrainDataset,
    Evaluator, PretrainEvaluator,
    Logger, set_all_seeds, count_parameters
)

# import your models
from train_duett_multilabel import DuETT
# from models import Strats, GRU_TS, ... HyMaTE etc.


# -----------------------------
# small helpers
# -----------------------------


def _safe_metric_key(name: str) -> str:
    return str(name).strip().replace(" ", "_").replace("/", "_")


def set_output_dir(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        # keep original scheme but add stage + domain info
        base = './outputs/' + args.dataset + '/' + args.output_dir_prefix
        if args.pretrain:
            args.output_dir = base + 'pretrain/'
        else:
            if args.load_ckpt_path is not None:
                args.output_dir_prefix = 'finetune_' + args.output_dir_prefix
                base = './outputs/' + args.dataset + '/' + args.output_dir_prefix
            args.output_dir = base + args.model_type
            if args.model_type == 'strats':
                for param in ['num_layers', 'hid_dim', 'num_heads', 'dropout', 'attention_dropout', 'lr']:
                    args.output_dir += ',' + param + ':' + str(getattr(args, param))

        args.output_dir += f"__src:{args.source}__tgt:{args.target}"

    os.makedirs(args.output_dir, exist_ok=True)


def load_state_flexible(path: str) -> dict:
    """
    Supports:
      - raw state_dict (torch.save(model.state_dict()))
      - a dict checkpoint containing 'model_state_dict'
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        return obj["model_state_dict"]
    if isinstance(obj, dict):
        # could still be a state_dict
        # heuristic: state_dict keys often contain '.' and tensors as values
        return obj
    raise ValueError(f"Unsupported checkpoint format at {path}")


def save_model_checkpoint(
    save_dir: str,
    save_name: str,
    model,
    optimizer,
    args,
    extra: dict | None = None,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)

    # only store simple scalars in config to keep ckpt compact
    config_simple = {}
    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, type(None))):
            config_simple[k] = v

    ckpt = {
        "run_name": os.path.splitext(save_name)[0],
        "model_type": args.model_type,
        "source": args.source,
        "target": args.target,
        "seed": args.seed,
        "pretrain": args.pretrain,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (optimizer.state_dict() if optimizer is not None else None),
        "outcome_cols": getattr(args, "outcome_cols", None),
        "config": config_simple,
    }
    if extra:
        ckpt.update(extra)

    torch.save(ckpt, save_path)
    return save_path


def init_wandb_single_run(args):
    if not args.use_wandb:
        return None

    import wandb
    run_name = f"{args.model_type}_{args.source}"
    cfg = {
        "model_type": args.model_type,
        "source": args.source,
        "target": args.target,
        "seed": args.seed,
        "pipeline": args.pipeline,
        "data_dir": args.data_dir,
        "dataset_tag": args.dataset,
        "lr": args.lr,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_obs": getattr(args, "max_obs", None),
        "hid_dim": args.hid_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "attention_dropout": args.attention_dropout,
        "cm_type": getattr(args, "cm_type", None),
        "sr_ratio": getattr(args, "sr_ratio", None),
        "num_blocks": getattr(args, "num_blocks", None),
        "mlp_ratio": getattr(args, "mlp_ratio", None),
        "train_frac": args.train_frac,
        "run": args.run,
    }

    wb = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=cfg,
    )
    return wb



@torch.no_grad()
def eval_macro_log(wb, prefix: str, split: str, res: dict, step: int):
    """
    Lightweight macro logging during training.
    This is NOT required to match Raindrop; only final test keys must match.
    """
    if wb is None:
        return
    wb.log({
        f"{prefix}/{split}_loss": float(res.get("train_loss", np.nan)),
        f"{prefix}/{split}_auroc": float(res.get("auroc", res.get("auroc_macro", np.nan))),
        f"{prefix}/{split}_auprc": float(res.get("auprc", res.get("auprc_macro", np.nan))),
        f"{prefix}/{split}_acc":  float(res.get("acc",  res.get("acc_micro", np.nan))),
        f"{prefix}/{split}_valid_labels": int(res.get("valid_labels", res.get("num_valid_labels", 0))),
        "step": step,
    })


def run_pretrain_stage(args: argparse.Namespace, wb, run_name: str):
    """
    Pretrain stage:
      - Uses PretrainDataset + PretrainEvaluator
      - Model is initialized fresh
      - Saves best state to ./baseline_models_hm/{run_name}_pretrain.pt
    """
    args = copy.deepcopy(args)
    args.pretrain = 1
    args.load_ckpt_path = None
    set_output_dir(args)

    args.logger = Logger(args.output_dir, 'log.txt')
    args.logger.write('\n[PRETRAIN] ' + str(args))

    dataset = PretrainDataset(args)
    evaluator = PretrainEvaluator(args)

    model_class = {
        "duett": DuETT,
        # "hymate": HyMaTE,
        # ...
    }
    if args.model_type not in model_class:
        raise ValueError(f"[PRETRAIN] Model type {args.model_type} not wired in model_class.")

    model = model_class[args.model_type](args).to(args.device)
    count_parameters(args.logger, model)

    # (optional) update wandb watch to this stage model (still single run)
    if wb is not None:
        import wandb
        wandb.watch(model, log="gradients", log_freq=100)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # steps setup
    num_train = len(dataset.splits['train'])
    num_batches_per_epoch = num_train / args.train_batch_size
    args.max_steps = int(round(num_batches_per_epoch) * args.max_epochs)
    if args.validate_every is None:
        args.validate_every = int(np.ceil(num_batches_per_epoch))

    best_val_metric = -np.inf
    best_state = None
    wait = args.patience

    cum_train_loss = 0.0
    num_batches_trained = 0
    num_steps = 0

    model.train()
    train_bar = tqdm(range(args.max_steps), desc="pretrain")

    t0 = time.time()
    for step in train_bar:
        batch = dataset.get_batch()
        batch = {k: v.to(args.device) for k, v in batch.items()}

        loss = model(**batch)  # scalar loss for pretraining

        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        cum_train_loss += float(loss.item())
        num_batches_trained += 1
        num_steps += 1
        train_bar.set_description(f"pretrain_loss={cum_train_loss / max(1, num_batches_trained):.5f}")

        if (num_steps) % args.print_train_loss_every == 0:
            avg = cum_train_loss / max(1, num_batches_trained)
            args.logger.write(f"\n[PRETRAIN] Train-loss at step {num_steps}: {avg}")
            if wb is not None:
                wb.log({"pretrain/train_loss": avg, "step": num_steps})
            cum_train_loss, num_batches_trained = 0.0, 0

        if (num_steps >= args.validate_after) and (num_steps % args.validate_every == 0):
            val_res = evaluator.evaluate(model, dataset, "val", train_step=num_steps)
            # maximize loss_neg
            curr_val_metric = float(val_res.get("loss_neg", -np.inf))
            if wb is not None:
                wb.log({
                    "pretrain/val_loss_neg": curr_val_metric,
                    "step": num_steps
                })

            model.train(True)

            if np.isfinite(curr_val_metric) and curr_val_metric > best_val_metric:
                best_val_metric = curr_val_metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = args.patience
                args.logger.write(f"[PRETRAIN] New best val_loss_neg={best_val_metric:.6f} at step={num_steps}")
            else:
                wait -= 1
                args.logger.write(f"[PRETRAIN] wait -> {wait}")
                if wb is not None:
                    wb.log({"pretrain/earlystop_wait": wait, "step": num_steps})
                if wait == 0:
                    args.logger.write("[PRETRAIN] Patience reached")
                    break

    elapsed_min = (time.time() - t0) / 60.0
    args.logger.write(f"\n[PRETRAIN] Done. elapsed={elapsed_min:.2f} min | best_val_loss_neg={best_val_metric:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        args.logger.write("[PRETRAIN] Restored best model state.")

    # save pretrained model
    save_dir = "./baseline_models_hm"
    save_path = save_model_checkpoint(
        save_dir=save_dir,
        save_name=f"{run_name}_pretrain.pt",
        model=model,
        optimizer=optimizer,
        args=args,
        extra={
            "final": {
                "best_val_loss_neg": float(best_val_metric),
                "elapsed_min": float(elapsed_min),
            }
        },
    )
    args.logger.write(f"✅ Saved pretrained model to {save_path}")

    if wb is not None:
        wb.log({
            "pretrain/final_best_val_loss_neg": float(best_val_metric),
            "pretrain/elapsed_min": float(elapsed_min),
            "pretrain/ckpt_path": save_path,
        })

    return save_path


def run_finetune_stage(args: argparse.Namespace, wb, run_name: str, load_ckpt_path: str | None):
    """
    Finetune stage:
      - Uses Dataset + Evaluator (domain shift)
      - Trains on source train, early stop on source val
      - Final eval on source test + target test (target split)
      - Saves best state to ./baseline_models_hm/{run_name}_finetune.pt
      - Logs final wandb test keys EXACTLY like Raindrop
    """
    args = copy.deepcopy(args)
    args.pretrain = 0
    args.load_ckpt_path = load_ckpt_path
    set_output_dir(args)

    args.logger = Logger(args.output_dir, 'log.txt')
    args.logger.write('\n[FINETUNE] ' + str(args))

    dataset = Dataset(args)
    evaluator = Evaluator(args)

    model_class = {
        "duett": DuETT,
        # "hymate": HyMaTE,
        # ...
    }
    if args.model_type not in model_class:
        raise ValueError(f"[FINETUNE] Model type {args.model_type} not wired in model_class.")

    model = model_class[args.model_type](args).to(args.device)
    count_parameters(args.logger, model)

    # (optional) update wandb watch to finetune model
    if wb is not None:
        import wandb
        wandb.watch(model, log="gradients", log_freq=100)

    # load pretrained weights if provided
    if args.load_ckpt_path is not None:
        curr_state_dict = model.state_dict()
        pt_state_dict = load_state_flexible(args.load_ckpt_path)
        loaded = 0
        for k, v in pt_state_dict.items():
            if k in curr_state_dict and curr_state_dict[k].shape == v.shape:
                curr_state_dict[k] = v
                loaded += 1
        model.load_state_dict(curr_state_dict)
        args.logger.write(f"[FINETUNE] Loaded {loaded} tensors from {args.load_ckpt_path}")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # multilabel BCE with pos_weight from source train (computed in Dataset)
    pos_weight = torch.tensor(args.pos_weight, dtype=torch.float32, device=args.device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # steps setup
    num_train = len(dataset.splits['train'])
    num_batches_per_epoch = num_train / args.train_batch_size
    args.logger.write('\n[FINETUNE] No. of training batches per epoch = ' + str(num_batches_per_epoch))

    args.max_steps = int(round(num_batches_per_epoch) * args.max_epochs)
    if args.validate_every is None:
        args.validate_every = int(np.ceil(num_batches_per_epoch))

    best_val_metric = -np.inf
    best_state = None
    wait = args.patience

    cum_train_loss = 0.0
    num_batches_trained = 0
    num_steps = 0

    model_loss_curve = []
    model_perf_curve = []

    # initial eval
    if args.validate_after < 0:
        val0 = evaluator.evaluate(model, dataset, "val", train_step=-1)
        te0  = evaluator.evaluate(model, dataset, "test", train_step=-1)
        tg0  = evaluator.evaluate(model, dataset, "target", train_step=-1)
        eval_macro_log(wb, "finetune/init", "val", val0, step=-1)
        eval_macro_log(wb, "finetune/init", "test", te0, step=-1)
        eval_macro_log(wb, "finetune/init", "target", tg0, step=-1)

    model.train()
    train_bar = tqdm(range(args.max_steps), desc="finetune")

    t0 = time.time()
    for step in train_bar:
        batch = dataset.get_batch()
        batch = {k: v.to(args.device) for k, v in batch.items()}

        labels = batch["labels"]
        del batch["labels"]
        out = model(**batch)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        loss = criterion(logits, labels)

        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        cum_train_loss += float(loss.item())
        num_batches_trained += 1
        num_steps += 1
        train_bar.set_description(f"finetune_loss={cum_train_loss / max(1, num_batches_trained):.5f}")

        if (num_steps) % args.print_train_loss_every == 0:
            avg = cum_train_loss / max(1, num_batches_trained)
            args.logger.write(f"\n[FINETUNE] Train-loss at step {num_steps}: {avg}")
            if wb is not None:
                wb.log({"finetune/train_loss": avg, "step": num_steps})
            cum_train_loss, num_batches_trained = 0.0, 0

        # validation
        if (num_steps >= args.validate_after) and (num_steps % args.validate_every == 0):
            val_res = evaluator.evaluate(model, dataset, "val", train_step=num_steps)
            # early stop metric: macro AUPRC (like Raindrop early stop)
            curr_val_metric = float(val_res.get("auprc", val_res.get("auprc_macro", -np.inf)))

            # also evaluate test/target for curves (optional)
            test_res = evaluator.evaluate(model, dataset, "test", train_step=num_steps)
            tgt_res  = evaluator.evaluate(model, dataset, "target", train_step=num_steps)

            model_loss_curve.append(float(test_res.get("train_loss", np.nan)))
            model_perf_curve.append(float(test_res.get("auroc", test_res.get("auroc_macro", np.nan))))

            # log macro
            eval_macro_log(wb, "finetune/val", "val", val_res, step=num_steps)
            eval_macro_log(wb, "finetune/val", "test", test_res, step=num_steps)
            eval_macro_log(wb, "finetune/val", "target", tgt_res, step=num_steps)

            model.train(True)

            if np.isfinite(curr_val_metric) and curr_val_metric > best_val_metric:
                best_val_metric = curr_val_metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = args.patience
                args.logger.write(f"[FINETUNE] New best val AUPRC={best_val_metric:.6f} at step={num_steps}")
                if wb is not None:
                    wb.log({"finetune/best_val_auprc": best_val_metric, "step": num_steps})
            else:
                wait -= 1
                args.logger.write(f"[FINETUNE] wait -> {wait}")
                if wb is not None:
                    wb.log({"finetune/earlystop_wait": wait, "step": num_steps})
                if wait == 0:
                    args.logger.write("[FINETUNE] Patience reached")
                    break

    elapsed_min = (time.time() - t0) / 60.0
    args.logger.write(f"\n[FINETUNE] Done. elapsed={elapsed_min:.2f} min | best_val_auprc={best_val_metric:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        args.logger.write("[FINETUNE] Restored best model state.")

    # save finetuned model
    save_dir = "./baseline_models_hm"
    save_path = save_model_checkpoint(
        save_dir=save_dir,
        save_name=f"{run_name}_finetune.pt",
        model=model,
        optimizer=optimizer,
        args=args,
        extra={
            "final": {
                "best_val_auprc": float(best_val_metric),
                "elapsed_min": float(elapsed_min),
            }
        },
    )
    args.logger.write(f"✅ Saved finetuned model to {save_path}")

    # -------------------------
    # final evaluation: source test + target (Raindrop-compatible keys)
    # -------------------------
    src_res = evaluator.evaluate(model, dataset, "test", train_step=num_steps)
    tgt_res = evaluator.evaluate(model, dataset, "target", train_step=num_steps)

    # evaluator must provide these keys for Raindrop-style logging
    src_test_loss = float(src_res["train_loss"])
    src_auroc = float(src_res["auroc"])
    src_auprc = float(src_res["auprc"])
    src_acc = float(src_res["acc"])
    src_auroc_per = src_res["auroc_per"]
    src_auprc_per = src_res["auprc_per"]
    src_valid_mask = src_res["valid_mask"]

    tgt_test_loss = float(tgt_res["train_loss"])
    tgt_auroc = float(tgt_res["auroc"])
    tgt_auprc = float(tgt_res["auprc"])
    tgt_acc = float(tgt_res["acc"])
    tgt_auroc_per = tgt_res["auroc_per"]
    tgt_auprc_per = tgt_res["auprc_per"]
    tgt_valid_mask = tgt_res["valid_mask"]

    args.logger.write(f"[SOURCE TEST] loss={src_test_loss:.4f} AUROC={src_auroc:.4f} AUPRC={src_auprc:.4f} ACC={src_acc:.4f}")
    args.logger.write(f"[TARGET TEST] loss={tgt_test_loss:.4f} AUROC={tgt_auroc:.4f} AUPRC={tgt_auprc:.4f} ACC={tgt_acc:.4f}")

    # ---- FINAL wandb logging EXACTLY like Raindrop
    if wb is not None:
        outcome_cols = getattr(args, "outcome_cols", None)
        if outcome_cols is None:
            outcome_cols = [f"task_{i}" for i in range(len(src_auroc_per))]

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
            key = _safe_metric_key(name)
            if i < len(src_valid_mask) and bool(src_valid_mask[i]):
                log_dict[f"test/source_auroc_{key}"] = float(src_auroc_per[i])
                log_dict[f"test/source_auprc_{key}"] = float(src_auprc_per[i])
            if i < len(tgt_valid_mask) and bool(tgt_valid_mask[i]):
                log_dict[f"test/target_auroc_{key}"] = float(tgt_auroc_per[i])
                log_dict[f"test/target_auprc_{key}"] = float(tgt_auprc_per[i])

        # extra bookkeeping (won't conflict with Raindrop keys)
        log_dict.update({
            "finetune/elapsed_min": float(elapsed_min),
            "finetune/best_val_auprc": float(best_val_metric),
            "finetune/ckpt_path": save_path,
        })

        wb.log(log_dict)

    # optional plots (local)
    if len(model_perf_curve) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(model_perf_curve)
        ax1.set_title("SOURCE test AUROC (per val step)")
        ax2.plot(model_loss_curve)
        ax2.set_title("SOURCE test loss (per val step)")
        plt.tight_layout()
        plt.show()

    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # domain setting
    parser.add_argument("--source", type=str, default="mimic", choices=["mimic", "eicu"])
    parser.add_argument("--target", type=str, default=None, choices=[None, "mimic", "eicu"])
    parser.add_argument("--data_dir", type=str, default="./data")

    # pipeline
    parser.add_argument("--pipeline", action="store_true",
                        help="Run pretrain then finetune in one command (single wandb run).")

    # keep original hyperparams / options
    parser.add_argument('--dataset', type=str, default='clinical')  # used for output_dir naming
    parser.add_argument('--train_frac', type=float, default=0.5)
    parser.add_argument('--run', type=str, default='1o10')

    parser.add_argument('--model_type', type=str, default='duett',
                        choices=['gru', 'tcn', 'sand', 'grud', 'interpnet',
                                 'strats', 'istrats', 'ehrmamba', 'duett', 'hymate'])
    parser.add_argument('--load_ckpt_path', type=str, default=None)

    # strats-ish
    parser.add_argument('--max_obs', type=int, default=880)

    # hymate related
    parser.add_argument('--cm_type', type=str, default='EinFFT')
    parser.add_argument('--sr_ratio', type=int, default=8)
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--mlp_ratio', type=int, default=1)

    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attention_dropout', type=float, default=0.2)

    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--r', type=int, default=24)
    parser.add_argument('--M', type=int, default=12)

    parser.add_argument('--max_timesteps', type=int, default=880)
    parser.add_argument('--hours_look_ahead', type=int, default=24)
    parser.add_argument('--ref_points', type=int, default=24)

    # training/eval
    parser.add_argument('--pretrain', type=int, default=0)  # ignored if --pipeline
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_dir_prefix', type=str, default='')
    parser.add_argument('--seed', type=int, default=9871)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--print_train_loss_every', type=int, default=100)
    parser.add_argument('--validate_after', type=int, default=-1)
    parser.add_argument('--validate_every', type=int, default=None)

    # wandb
    parser.add_argument("--use_wandb", action="store_false")
    parser.add_argument("--wandb_project", type=str, default="Baselines")
    parser.add_argument("--wandb_entity", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # infer target if not provided
    if args.target is None:
        args.target = "eicu" if args.source == "mimic" else "mimic"
    if args.source == args.target:
        raise ValueError("source and target must be different")

    # device + seed (shared for whole pipeline)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device.type != "cuda":
        print("WARNING: CUDA not available; training will be slow.")

    # seed offset uses run index (keep original behavior)
    set_all_seeds(args.seed + int(args.run.split('o')[0]))

    run_name = f"{args.model_type}_{args.source}"
    wb = init_wandb_single_run(args)

    try:
        if args.pipeline:
            # 1) pretrain
            pre_ckpt_path = run_pretrain_stage(args, wb, run_name)
            # 2) finetune (load from pretrain ckpt)
            run_finetune_stage(args, wb, run_name, load_ckpt_path=pre_ckpt_path)
        else:
            # single-stage run (respects args.pretrain)
            if args.pretrain == 1:
                run_pretrain_stage(args, wb, run_name)
            else:
                run_finetune_stage(args, wb, run_name, load_ckpt_path=args.load_ckpt_path)
    finally:
        if wb is not None:
            wb.finish()


if __name__ == "__main__":
    main()
# python train_hymate_models.py --pipeline --source mimic --model_type duett --wandb_project Baselines