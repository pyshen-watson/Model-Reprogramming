import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class DNN6(nn.Module):

    def __init__(self, n_class=10):
        super(DNN6, self).__init__()

        self.backbone = nn.Sequential(
            nn.Linear(3072, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_class),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        x = x.flatten(1)
        x = self.backbone(x)
        return x


class CNN6(nn.Module):

    def __init__(self, n_class=10):
        super(CNN6, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(),
        )

        self.linear = nn.Linear(8 * 8 * 1024, n_class)

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.linear(x)
        return x


class SourceModule(pl.LightningModule):

    def __init__(self, source_model, lr=1e-3):
        super(SourceModule, self).__init__()
        self.source_model = source_model
        self.lr = lr

    def forward(self, x: torch.Tensor):
        return self.source_model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

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
        torch.save(self.source_model.state_dict(), "best_source.pt")
