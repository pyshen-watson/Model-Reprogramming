import torch
import torch.nn as nn
from dataclasses import dataclass
from ..common.base import Base

@dataclass(eq=False) # This avoid lightning trainer try to hash the module
class VGGBlock(Base):
    """
    This block will create a module looks like: (CR)xn+P
    C is for Conv2d, R is for ReLU, n is num_conv and P is Average Pooling
    """
    in_channel: int
    out_channel: int
    num_conv: int

    def __post_init__(self):
        super(VGGBlock, self).__init__()

        in_ch = self.in_channel
        out_ch = self.out_channel
        layers = []

        for _ in range(self.num_conv):
            layers += [Base.Conv(in_ch, out_ch), Base.Act_fn()]
            in_ch = out_ch

        # We use average pooling here because it's hard to compute the kernel of max pooling
        layers += [Base.AvgPooling()]
        self.net = nn.Sequential(*layers)
        self.init_weights(self.net)

    def forward(self, x: torch.Tensor):
        return self.net(x)


@dataclass(eq=False)  # This avoid lightning trainer try to hash the module
class VGG(Base):

    input_size: tuple  # (N, C, H, W)
    n_class: int = 10
    pooling: int = 3  # The number of pooling, i.e. the number of blocks
    level: int = 2  # The number of conv layers per block
    width_base: int = 32  # The number of filter of the first convolutional layer

    def __post_init__(self):
        super(VGG, self).__init__(self.input_size)

        layers = []
        in_ch = 3
        out_ch = self.width_base

        # Convolutional backbone
        for _ in range(self.pooling):
            layers.append(VGGBlock(in_ch, out_ch, self.level))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512) # The paper of VGG suggests that number of channel should not exceed 512

        # Fully connected layers
        out_ch = layers[-1].out_channel # Update to the real number of output channel
        layers += [Base.GlobalAvgPooling(), Base.Flatten(), Base.Linear(out_ch , self.n_class)]

        # Build the net and initialize it
        self.net = nn.Sequential(*layers)
        self.init_weights(self.net)

    def forward(self, x):
        return self.net(x)