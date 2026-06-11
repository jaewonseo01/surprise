import numpy as np
import time
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryAccuracy


class STRaTSLit(pl.LightningModule):
    """
    Downstream (binary / multilabel)
    - metrics/logging: epoch-end only (self.log로 통일)
    - batch signature:
      (times, varis, values, pad_mask, pre_mask, y, pid, static)
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        task_type: str,
        outcome_cols,
        target_indices,
        lr: float,
        weight_decay: float,
        optimizer: str = "adamw",
        scheduler: str = "none",
        scheduler_params=None,
        pretrained_path=None,
    ):
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.outcome_cols = list(outcome_cols)
        self.target_indices = list(range(len(self.outcome_cols))) if target_indices is None else list(target_indices)

        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params or {}
        self.pretrained_path = pretrained_path

        if self.task_type == "binary":
            self.criterion = nn.BCEWithLogitsLoss()
            self.auroc = BinaryAUROC()
            self.auprc = BinaryAveragePrecision()
            self.acc = BinaryAccuracy()
        elif self.task_type == "multilabel":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("task_type must be 'binary' or 'multilabel'")

        self._logits = {"train": [], "val": [], "test": []}
        self._targets = {"train": [], "val": [], "test": []}
        self._last_test = None  # will store dict after each trainer.test()

        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch):
        times, varis, values, pad_mask, _pre_mask, _y, _pid, static = batch
        return self.model(
            times=times,
            varis=varis,
            values=values,
            statics=static,
            padding_mask=pad_mask,
            pretrain=False,
        )

    def _select_targets(self, y: torch.Tensor) -> torch.Tensor:
        return y[:, self.target_indices].float()

    def _shared_step(self, batch, stage: str):
        out = self.forward(batch)
        logits = out["pred"]
        y = batch[5]
        targets = self._select_targets(y)

        if self.task_type == "binary":
            logits = logits.view(-1)
            targets = targets.view(-1)
        else:
            if logits.dim() == 1:
                logits = logits.unsqueeze(-1)

        loss = self.criterion(logits, targets)
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)

        if stage != "train":
            self._logits[stage].append(logits.detach().cpu())
            self._targets[stage].append(targets.detach().cpu())
        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        self._shared_step(batch, "val")

    def test_step(self, batch, _):
        self._shared_step(batch, "test")
    
    def on_train_epoch_start(self):
        self._train_epoch_start_time = time.time()

    def on_train_epoch_end(self):
        # self._logits["train"].clear()
        # self._targets["train"].clear()
        elapsed = time.time() - self._train_epoch_start_time
        print(f"[TRAIN] Epoch {self.current_epoch} took {elapsed:.2f} sec")

    def on_validation_epoch_end(self):
        self._compute_epoch_metrics("val")

    def on_test_epoch_end(self):
        self._compute_epoch_metrics("test")

    def _compute_epoch_metrics(self, stage: str):
        if not self._logits[stage]:
            return
        logits = torch.cat(self._logits[stage], dim=0)
        targets = torch.cat(self._targets[stage], dim=0)

        probs = torch.sigmoid(logits)

        # ---- BINARY
        if self.task_type == "binary":
            t = targets.int()
            p = probs
            auroc_v = float(self.auroc(p, t).item())
            auprc_v = float(self.auprc(p, t).item())
            acc_v   = float(self.acc((p >= 0.5).int(), t).item())

            self.log(f"{stage}_AUROC", auroc_v, on_epoch=True)
            self.log(f"{stage}_AUPRC", auprc_v, on_epoch=True)
            self.log(f"{stage}_ACC", acc_v, on_epoch=True)

            if stage == "test":
                self._last_test = {
                    "auroc_macro": auroc_v,
                    "auprc_macro": auprc_v,
                    "acc_macro": acc_v,
                    "auroc_per": np.array([auroc_v], dtype=np.float32),
                    "auprc_per": np.array([auprc_v], dtype=np.float32),
                    "valid_mask": np.array([True], dtype=bool),
                }

        # ---- MULTILABEL
        else:
            K = probs.size(1)

            auroc_per = np.full((K,), np.nan, dtype=np.float32)
            auprc_per = np.full((K,), np.nan, dtype=np.float32)
            acc_per   = np.full((K,), np.nan, dtype=np.float32)
            valid_mask = np.zeros((K,), dtype=bool)

            for j in range(K):
                p = probs[:, j].detach().cpu()
                t = targets[:, j].detach().cpu().int()

                # undefined if all-0 or all-1
                if int(t.min()) == int(t.max()):
                    continue

                valid_mask[j] = True
                auroc_per[j] = float(BinaryAUROC()(p, t).item())
                auprc_per[j] = float(BinaryAveragePrecision()(p, t).item())
                acc_per[j]   = float(BinaryAccuracy()((p >= 0.5).int(), t).item())

            # macro over valid labels only
            if valid_mask.any():
                self.log(f"{stage}_AUROC_macro", float(np.nanmean(auroc_per)), on_epoch=True)
                self.log(f"{stage}_AUPRC_macro", float(np.nanmean(auprc_per)), on_epoch=True)
                self.log(f"{stage}_ACC_macro",   float(np.nanmean(acc_per)),   on_epoch=True)
            else:
                self.log(f"{stage}_AUROC_macro", float("nan"), on_epoch=True)
                self.log(f"{stage}_AUPRC_macro", float("nan"), on_epoch=True)
                self.log(f"{stage}_ACC_macro",   float("nan"), on_epoch=True)

            # ✅ store only for test (so you can prefix source/target externally)
            if stage == "test":
                self._last_test = {
                    "auroc_macro": float(np.nanmean(auroc_per)) if valid_mask.any() else float("nan"),
                    "auprc_macro": float(np.nanmean(auprc_per)) if valid_mask.any() else float("nan"),
                    "acc_macro":   float(np.nanmean(acc_per))   if valid_mask.any() else float("nan"),
                    "auroc_per": auroc_per,
                    "auprc_per": auprc_per,
                    "valid_mask": valid_mask,
                }

        self._logits[stage].clear()
        self._targets[stage].clear()


    def configure_optimizers(self):
        if self.optimizer_name == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        if self.scheduler == "none":
            return opt
        if self.scheduler == "cosine":
            T_max = self.scheduler_params.get("T_max", 100)
            eta_min = self.scheduler_params.get("eta_min", 1e-6)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
        raise ValueError(f"Unknown scheduler: {self.scheduler}")
