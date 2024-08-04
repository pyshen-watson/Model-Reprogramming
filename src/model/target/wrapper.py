import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict
from pathlib import Path
from dataclasses import dataclass

from .prompt import VisualPromptLayer
from ..base import Base, BaseModule


@dataclass(eq=False)  # This avoid lightning trainer try to hash the module
class ReprogrammingWrapper(pl.LightningModule, Base):

    # About Trainer
    name: str
    log_dir: Path
    hp: Dict = None
    lr: float = 1e-3
    wd: float = 1e-3
    loss_fn: str = "CE"

    # About model
    source_size: int = 112
    target_size: int = 32
    vp: bool = False
    fc: bool = False
    device: int = 0

    def __post_init__(self):
        super(ReprogrammingWrapper, self).__init__()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_hyperparameters("hp")
        self.automatic_optimization = False

        self.vp_layer = VisualPromptLayer(
            self.source_size, 
            self.target_size, 
            vp=self.vp,
            fc=self.fc,
            device=self.device
        )

    def set_source_model(self, model: BaseModule):
        """
        This function will set the backbone model to the lightning module.
        and save the model structure to the log directory.
        """
        self.source_model = model.requires_grad_(False)
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
        return [
            torch.optim.Adam([self.vp_layer.delta], lr=self.lr*100, weight_decay=self.wd),
            torch.optim.Adam(self.vp_layer.fc_layer.parameters(), lr=self.lr, weight_decay=self.wd),
        ]

    def calc_loss(self, img, label, split: str):

        logits = self(img)

        if self.loss_fn == "CE":
           loss = F.cross_entropy(logits, label)

        elif self.loss_fn == "MSE":
            label_OH = F.one_hot(label, self.source_model.n_class).float()
            loss = F.mse_loss(logits, label_OH)

        acc = (logits.argmax(1) == label).float().mean()

        if split == "train":
            opt1, opt2 = self.optimizers()
            opt1.zero_grad()
            opt2.zero_grad()
            self.manual_backward(loss)
            opt1.step()
            opt2.step()

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
        self.vp_layer.save(save_path)
