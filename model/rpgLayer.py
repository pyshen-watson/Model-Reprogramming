import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .backbone import Base


class FullyConnectedLayer(Base):

    def __init__(self, target_size: int, enable: bool):
        super(FullyConnectedLayer, self).__init__()
        self.target_size = target_size
        self.enable = enable

        if enable:
            layer_size = 3 * target_size * target_size  # Ex. cifar-10 is 3x32x32
            self.fc_layer = nn.Linear(layer_size, layer_size)
            self._initialize_weights(self.fc_layer)

    def forward(self, x: torch.Tensor):

        if not self.enable:
            return x

        x = x.flatten(1)  # Nx3x32x32 -> Nx3072
        x = self.fc_layer(x)
        x = x.reshape(-1, 3, self.target_size, self.target_size)
        return x


class VisualPromptLayer(Base):

    def __init__(self, inner_size: int, source_size: int, enable: bool):
        super(VisualPromptLayer, self).__init__()
        self.inner_size = inner_size
        self.source_size = source_size
        self.enable = enable
        pad = (source_size-inner_size) // 2

        if enable:
            self.pad_t = nn.Parameter(torch.randn(3, pad, source_size))
            self.pad_b = nn.Parameter(torch.randn(3, pad, source_size))
            self.pad_l = nn.Parameter(torch.randn(3, inner_size, pad))
            self.pad_r = nn.Parameter(torch.randn(3, inner_size, pad))
            self._initialize_weights([self.pad_t, self.pad_b, self.pad_l, self.pad_r])

        self.resize = partial(F.interpolate, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor):

        if not self.enable:
            return self.resize(x, self.source_size)

        # Resize x to inner size before adding the prompt
        x = self.resize(x, self.inner_size) 

        # Prepare the prompt
        batch_size = x.size(0)
        pad_t = self.pad_t.repeat(batch_size, 1, 1, 1)
        pad_b = self.pad_b.repeat(batch_size, 1, 1, 1)
        pad_l = self.pad_l.repeat(batch_size, 1, 1, 1)
        pad_r = self.pad_r.repeat(batch_size, 1, 1, 1)

        # Add the prompt
        x = torch.cat([pad_l, x, pad_r], dim=3)
        x = torch.cat([pad_t, x, pad_b], dim=2)
        return x

class ReprogrammingLayer(Base):

    def __init__(self, source_size=224, target_size=32, inner_size=156, fc=False, vp=False):
        super(ReprogrammingLayer, self).__init__()
        self.fc_layer = FullyConnectedLayer(target_size, enable=fc)
        self.vp_layer = VisualPromptLayer(inner_size, source_size, enable=vp)

    def forward(self, x: torch.Tensor):
        x = self.fc_layer(x)
        x = self.vp_layer(x)
        return x

