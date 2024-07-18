import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict
from pathlib import Path
from dataclasses import dataclass

from .prompt import VisualPromptLayer
from ..base import Base


@dataclass(eq=False)  # This avoid lightning trainer try to hash the module
class ReprogrammingWrapper(pl.LightningModule, Base):

    # About Trainer
    name: str
    log_dir: Path
    hp: Dict = None
    lr: float = 1e-3
    wd: float = 1e-3

    # About model
    source_size: int = 112
    target_size: int = 32
    vp: bool = False
    rz: bool = False
    fc: bool = False

    def __post_init__(self):
        super(ReprogrammingWrapper, self).__init__()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_hyperparameters("hp")

        self.vp_layer = VisualPromptLayer(
            self.source_size, 
            self.target_size, 
            prompt=self.vp, 
            resize=self.rz
        )

    def set_source_model(self, model: Base):
        """
        This function will set the backbone model to the lightning module.
        and save the model structure to the log directory.
        """
        self.source_model = model.freeze()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        with open(self.log_dir / "model_structure.txt", "w") as f:
            print(model.summary(), file=f)

        with open(self.log_dir / "prompt_structure.txt", "w") as f:
            print(self.vp_layer.summary(), file=f)

        return self

    def forward(self, x: torch.Tensor):
        x = self.vp_layer(x)
        x = self.source_model(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.vp_layer.parameters(), lr=self.lr, weight_decay=self.wd
        )

    def calc_loss(self, img, label, split: str):

        logits = self(img)
        loss = F.cross_entropy(logits, label)
        pred = logits.argmax(1)
        acc = (pred == label).float().mean()

        # Logging
        self.log(f"{split}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{split}_acc", acc, prog_bar=True, sync_dist=True)

        return {"loss": loss, "acc": acc, "pred": pred, "logits": logits}

    def training_step(self, batch, batch_idx):
        return self.calc_loss(batch[0], batch[1], "train")

    def validation_step(self, batch, batch_idx):
        return self.calc_loss(batch[0], batch[1], "val")

    def test_step(self, batch, batch_idx):
        return self.calc_loss(batch[0], batch[1], "test")

    def on_save_checkpoint(self, checkpoint):
        save_path = self.log_dir / f"{self.name}.pt"
        self.vp_layer.save(save_path)
