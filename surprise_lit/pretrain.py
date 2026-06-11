import time
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class PretrainLit(pl.LightningModule):
    """
    batch signature must be:
      (times, varis, values, pad_mask, pre_mask, y, pid, static)
    y/pid는 pretrain에서 사용 안 함.
    """

    def __init__(self, model, lr=1e-3, weight_decay=1e-2, optimizer="adamw"):
        super().__init__()
        self.model = model
        self.lr = lr
        self.wd = weight_decay
        self.optimizer = optimizer

    def training_step(self, batch, _):
        times, varis, values, pad, pre, _, _, static = batch
        out = self.model(
            times=times,
            varis=varis,
            values=values,
            statics=static,
            padding_mask=pad,
            pretrain=True,
            pretrain_mask=pre,
        )
        pred_vals = out["pred_vals"]
        mask = out["pretrain_mask"].bool() & (~pad)
        loss = pred_vals.sum() * 0.0 if not mask.any() else F.mse_loss(pred_vals[mask], values[mask])
        self.log("pretrain_train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        times, varis, values, pad, pre, _, _, static = batch
        out = self.model(
            times=times,
            varis=varis,
            values=values,
            statics=static,
            padding_mask=pad,
            pretrain=True,
            pretrain_mask=pre,
        )
        pred_vals = out["pred_vals"]
        mask = out["pretrain_mask"].bool() & (~pad)
        loss = pred_vals.sum() * 0.0 if not mask.any() else F.mse_loss(pred_vals[mask], values[mask])
        self.log("pretrain_val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.optimizer=="adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.optimizer=="adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            raise Exception("Only Adam and AdamW is implemented. For additional optimizer support, modify pretrain.py") 
        
    def on_train_epoch_start(self):
        self._train_epoch_start_time = time.time()

    def on_train_epoch_end(self):
        elapsed = time.time() - self._train_epoch_start_time
        print(f"[TRAIN] Epoch {self.current_epoch} took {elapsed:.2f} sec")