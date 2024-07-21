import torch.nn as nn
from .cnn import CNNBlock
from ..base import Base, BaseModule
from dataclasses import dataclass

@dataclass(eq=False)  # This avoid lightning trainer try to hash the module
class VGG(BaseModule):

    input_size: tuple = (1, 3, 112, 112)
    n_class: int = 10
    pooling: int = 2
    level: int = 1  # The number of conv layers per pooling
    width_base: int = 32  # The number of filter of the first convolutional layer

    def __post_init__(self):
        super(VGG, self).__init__()

        in_ch = 3
        out_ch = self.width_base
        layers = []

        # Convolutional backbone
        for _ in range(self.pooling):
            layers += [CNNBlock(in_ch, out_ch, self.level), Base.AvgPooling()]
            in_ch = out_ch
            out_ch *= 2

        # Fully connected layers
        layers += [
            Base.GlobalAvgPooling(), 
            Base.Flatten(), 
            Base.Linear(out_ch //2 , self.n_class)
        ]

        self.layers = nn.ModuleList(layers)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x