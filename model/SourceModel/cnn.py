import torch
import torch.nn as nn
from ..common import Base
from dataclasses import dataclass

@dataclass(eq=False) # This avoid lightning trainer try to hash the module
class CNN(Base):

    n_class: int = 10
    n_layer: int = 6
    input_size: int = 224
    conv_width: int = 128
    init_std: float = 2 ** 0.5

    def __post_init__(self):
        super(CNN, self).__init__()

        in_ch = 3
        out_ch = self.conv_width
        layers = []

        # Convolutional layers
        for _ in range(self.n_layer):
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding="same"))
            layers.append(nn.ReLU())
            in_ch = out_ch
        
        # Linear layers
        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.input_size * self.input_size * out_ch, self.n_class))

        # Build the net and initialize it
        self.net = nn.Sequential(*layers)
        self.init_weights(self.init_std) # Defined in Base

    def forward(self, x: torch.Tensor):
        return self.net(x)