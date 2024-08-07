import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Dict
from pathlib import Path
from dataclasses import dataclass
from ..base import BaseModule


@dataclass(eq=False)  # This avoid lightning trainer try to hash the module
class SourceWrapper(pl.LightningModule):

    name: str
    log_dir: Path
    hp: Dict = None
    lr: float = 1e-3
    wd: float = 1e-3
    loss_fn: str = "CE"

    def __post_init__(self):
        super(SourceWrapper, self).__init__()
        self.save_hyperparameters("hp")

    def set_source_model(self, model: BaseModule):
        """
        This function will set the backbone model to the lightning module.
        and save the model structure to the log directory.
        """
        self.source_model = model
        self.log_dir.mkdir(parents=True, exist_ok=True)

        with open(self.log_dir / "model_structure.txt", "w") as f:
            print(model.summary(), file=f)

        return self

    def forward(self, x: torch.Tensor):
        return self.source_model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.source_model.parameters(), lr=self.lr, weight_decay=self.wd
        )

    def calc_loss(self, img, label, split: str):

        # Calucate the metrics
        logits = self(img)

        if self.loss_fn == "CE":
           loss = F.cross_entropy(logits, label)

        elif self.loss_fn == "MSE":
            label_OH = F.one_hot(label, self.source_model.n_class).float()
            loss = F.mse_loss(logits, label_OH)

        acc = (logits.argmax(1) == label).float().mean()

        # Logging
        self.log(f"{split}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{split}_acc", acc, prog_bar=True, sync_dist=True)
        return {"loss": loss, "acc": acc}

    def training_step(self, batch, batch_idx):
        return self.calc_loss(batch[0], batch[1], "train")

    def validation_step(self, batch, batch_idx):
        return self.calc_loss(batch[0], batch[1], "val")

    def test_step(self, batch, batch_idx):
        return self.calc_loss(batch[0], batch[1], "test")

    def on_save_checkpoint(self, checkpoint):
        save_path = self.log_dir / f"{self.name}.pt"
        self.source_model.save(save_path)
