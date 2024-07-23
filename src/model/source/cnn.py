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

@dataclass(eq=False)  # This avoid lightning trainer try to hash the module
class CNN(BaseModule):

    input_size: tuple = (1, 3, 112, 112)
    width: int = 32  # The number of filter of the first convolutional layer
    level: int = 1  # The number of conv
    n_class: int = 10

    def __post_init__(self):
        super(CNN, self).__init__()

        layers = [
            CNNGroup(3, self.width, self.level),
            Base.GlobalAvgPooling(), 
            Base.Flatten(), 
            Base.Linear(self.width , self.n_class)
        ]

        self.layers = nn.ModuleList(layers) # Register the layers

    def forward(self, x):     
        for layer in self.layers:
            x = layer(x)
        return x
