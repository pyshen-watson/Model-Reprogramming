import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .backbone import Loadable


class SourceModule(pl.LightningModule):

    def __init__(self, source_model, lr=1e-3, weight_decay=0.0):
        super(SourceModule, self).__init__()
        self.source_model: Loadable = source_model
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters("lr", "weight_decay")

    def forward(self, x: torch.Tensor):
        return self.source_model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

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
        self.source_model.save(f"best_source_w={self.weight_decay}.pt")
