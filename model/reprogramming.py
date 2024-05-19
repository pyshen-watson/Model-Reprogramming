import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .backbone import Loadable


class ReprogrammingLayer(Loadable):
    def __init__(self, rescale_size=24, output_size=32):
        super(ReprogrammingLayer, self).__init__()
        self.rescale_size = rescale_size
        self.output_size = output_size

        # Calculate the padding dimensions
        pad_h = (output_size - rescale_size) // 2
        pad_w = (output_size - rescale_size) // 2

        # Initialize the padding with trainable parameters
        self.pad_t = nn.Parameter(torch.randn(3, pad_h, output_size))
        self.pad_b = nn.Parameter(torch.randn(3, pad_h, output_size))
        self.pad_l = nn.Parameter(torch.randn(3, rescale_size, pad_w))
        self.pad_r = nn.Parameter(torch.randn(3, rescale_size, pad_w))

        # One layer of DNN
        # self.fc = nn.Linear(3 * output_size * output_size, 3 * output_size * output_size)

    def forward(self, x: torch.Tensor):

        # Resize the input image
        x = F.interpolate(x, size=self.rescale_size, mode="bilinear", align_corners=False)

        # Add the padding
        batch_size = x.size(0)
        pad_t = self.pad_t.repeat(batch_size, 1, 1, 1)
        pad_b = self.pad_b.repeat(batch_size, 1, 1, 1)
        pad_l = self.pad_l.repeat(batch_size, 1, 1, 1)
        pad_r = self.pad_r.repeat(batch_size, 1, 1, 1)
        x = torch.cat([pad_l, x, pad_r], dim=3)
        x = torch.cat([pad_t, x, pad_b], dim=2)

        # Pass a DNN layer
        # x = x.flatten(1)
        # x = self.fc(x)
        # x = x.reshape(-1, 3, self.output_size, self.output_size)
        return x

class LabelMappingLayer(Loadable):
    
    def __init__(self, n_source_class=10, n_target_class=10):
        super(LabelMappingLayer, self).__init__()
        self.fc = nn.Linear(n_source_class, n_target_class)
        
    def forward(self, x: torch.Tensor):
        return self.fc(x)
    
class ReprogrammingModule(pl.LightningModule, Loadable):
    
    def __init__(self, source_model, rescale_size=24, source_size=32, n_target_class=10, lr=1e-3):
        super(ReprogrammingModule, self).__init__()
        self.rpm_layer = ReprogrammingLayer(rescale_size, source_size)
        self.source_model: nn.Module = source_model
        self.lm_layer = LabelMappingLayer(source_model.n_class, n_target_class)
        self.lr = lr
        
        # Freeze the source model
        for param in self.source_model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        x = self.rpm_layer(x)
        x = self.source_model(x)
        x = self.lm_layer(x)
        return x

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
        self.rpm_layer.save(f"best_rpm_layer_{self.source_model.name}.pt")
        self.lm_layer.save(f"best_lm_layer_{self.source_model.name}.pt")
        
    def load(self, rpm_path: str, lm_path: str):
        self.rpm_layer.load(rpm_path)
        self.lm_layer.load(lm_path)
        return self