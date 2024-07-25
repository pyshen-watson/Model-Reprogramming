import torch.nn as nn
from ..base import Base, BaseModule
from dataclasses import dataclass

class CNNGroup(BaseModule):
    """
    This is a nn.Module structured like: `(CR)xn`.
    - `C` is for Conv2d, `R` is for ReLU and `n` is num_conv.
    """

    def __init__(self, in_ch, out_ch, n_conv):
        super(CNNGroup, self).__init__()
        layers = [Base.Conv(in_ch, out_ch), Base.Act_fn()]

        # From the second layer, the input channel will be the same as the output channel
        for _ in range(n_conv - 1):
            layers += [Base.Conv(out_ch, out_ch), Base.Act_fn()]
        self.layers = nn.ModuleList(layers) # Register the layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

