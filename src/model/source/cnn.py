import torch.nn as nn
from ..base import Base, BaseModule
from dataclasses import dataclass

class CNNBlock(BaseModule):
    """
    This is a nn.Module structured like: `(CR)xn`.
    - `C` is for Conv2d, `R` is for ReLU and `n` is num_conv.
    """

    def __init__(self, in_ch, out_ch, n_conv):
        super(CNNBlock, self).__init__()
        layers = [Base.Conv(in_ch, out_ch), Base.Act_fn()]

        # From the second layer, the input channel will be the same as the output channel
        for _ in range(n_conv - 1):
            layers += [Base.Conv(out_ch, out_ch), Base.Act_fn()]
        self.layers = nn.ModuleList(layers) # Register the layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

@dataclass(eq=False)  # This avoid lightning trainer try to hash the module
class CNN(BaseModule):

    input_size: tuple = (1, 3, 112, 112)
    n_class: int = 10
    level: int = 1  # The number of conv layers per pooling
    width_base: int = 32  # The number of filter of the first convolutional layer

    def __post_init__(self):
        super(CNN, self).__init__()

        layers = [
            CNNBlock(3, self.width_base, self.level),
            Base.GlobalAvgPooling(), 
            Base.Flatten(), 
            Base.Linear(self.width_base , self.n_class)
        ]

        self.layers = nn.ModuleList(layers) # Register the layers

    def forward(self, x):     
        for layer in self.layers:
            x = layer(x)
        return x
