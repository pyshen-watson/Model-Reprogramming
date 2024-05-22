import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .backbone import Base
from pathlib import Path


class ReprogrammingLayer(Base):
    def __init__(self, inner_size=24, outter_size=32, visual_prompt=True, fc_layer=True):
        super(ReprogrammingLayer, self).__init__()
        self.inner_size = inner_size
        self.outter_size = outter_size
        self.VP = visual_prompt
        self.FC = fc_layer
        
        if visual_prompt:
            # Calculate the padding dimensions
            pad_h = (outter_size - inner_size) // 2
            pad_w = (outter_size - inner_size) // 2

            # Initialize the padding with trainable parameters
            self.pad_t = nn.Parameter(torch.randn(3, pad_h, outter_size))
            self.pad_b = nn.Parameter(torch.randn(3, pad_h, outter_size))
            self.pad_l = nn.Parameter(torch.randn(3, inner_size, pad_w))
            self.pad_r = nn.Parameter(torch.randn(3, inner_size, pad_w))

        if fc_layer:
            # One layer of DNN
            self.fc = nn.Linear(3 * outter_size * outter_size, 3 * outter_size * outter_size)

    def forward(self, x: torch.Tensor):
        
        
        if self.VP:
            # Resize the input image
            x = F.interpolate(x, size=self.inner_size, mode="bilinear", align_corners=False)
            # Add the padding
            batch_size = x.size(0)
            pad_t = self.pad_t.repeat(batch_size, 1, 1, 1)
            pad_b = self.pad_b.repeat(batch_size, 1, 1, 1)
            pad_l = self.pad_l.repeat(batch_size, 1, 1, 1)
            pad_r = self.pad_r.repeat(batch_size, 1, 1, 1)
            x = torch.cat([pad_l, x, pad_r], dim=3)
            x = torch.cat([pad_t, x, pad_b], dim=2)
        
        if self.FC:
            # Resize the image
            x = F.interpolate(x, size=self.outter_size, mode="bilinear", align_corners=False)
            # Pass a DNN layer
            x = x.flatten(1)
            x = self.fc(x)
            x = x.reshape(-1, 3, self.outter_size, self.outter_size)
        
        # x = F.interpolate(x, size=self.outter_size, mode="bilinear", align_corners=False)
        
        return x

class LabelMappingLayer(Base):
    
    def __init__(self, n_source_class=10, n_target_class=10):
        super(LabelMappingLayer, self).__init__()
        self.fc = nn.Linear(n_source_class, n_target_class)
        
    def forward(self, x: torch.Tensor):
        return self.fc(x)
    
class ReprogrammingModule(pl.LightningModule, Base):
    
    def __init__(self, source_model, inner_size=24, outter_size=32, lr=1e-3, visual_prompt=True, fc_layer=True):
        super(ReprogrammingModule, self).__init__()
        self.rpm_layer = ReprogrammingLayer(inner_size, outter_size, visual_prompt, fc_layer)
        self.source_model: nn.Module = source_model
        self.lr = lr
        
        # Freeze the source model
        for param in self.source_model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        x = self.rpm_layer(x)
        x = self.source_model(x)
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
        ckpt_key = [k for k in checkpoint['callbacks'].keys() if 'ModelCheckpoint' in k ][0]
        ckpt_dir = checkpoint['callbacks'][ckpt_key]["dirpath"]
        best_path = Path(ckpt_dir).parent / f"rpm_{self.source_model.name}.pt"
        self.rpm_layer.save(best_path)