import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict
from pathlib import Path
from dataclasses import dataclass
from .backbone import Base
from .rpgLayer import ReprogrammingLayer

@dataclass(eq=False) # This avoid lightning trainer try to hash the module
class ReprogrammingModule(pl.LightningModule, Base):

    # About Trainer
    log_dir: Path
    hp: Dict = None

    #About Optimzer
    lr: float = 1e-3
    wd: float = 1e-3

    # About model
    source_size: int = 224
    target_size: int = 32
    inner_size: int = 156
    fc: bool = False
    vp: bool = False

    def __post_init__(self):
        super(ReprogrammingModule, self).__init__()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_hyperparameters("hp")
        self.rpg_layer = ReprogrammingLayer(
            self.source_size,
            self.target_size,
            self.inner_size,
            self.fc, self.vp)

    def set_source_model(self, source_model: Base):
        self.source_model = source_model.freeze()
        with open(self.log_dir / 'model_structure.txt', 'w') as f:
            f.write(str(source_model))
        return self

    def forward(self, x: torch.Tensor):
        x = self.rpg_layer(x)
        x = self.source_model(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.rpg_layer.parameters(), lr=self.lr, weight_decay=self.wd)

    def calc_loss(self, img, label, split: str):

        logits = self(img)
        loss = F.cross_entropy(logits, label)
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
        save_path = self.log_dir /  f"{self.name}.pt"
        self.rpg_layer.save(save_path)