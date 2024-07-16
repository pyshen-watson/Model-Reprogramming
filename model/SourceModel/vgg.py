import torch
import torch.nn as nn
from typing import List
from dataclasses import dataclass
from ..common.base import Base

@dataclass(eq=False) # This avoid lightning trainer try to hash the module
class BasicBlock(Base):
    """
    This block will create a module looks like: (CR)xn+P
    C is for Conv2d, R is for ReLU, n is num_conv and P is Average Pooling
    """
    in_channel: int
    out_channel: int
    num_conv: int

    def __post_init__(self):
        super(BasicBlock, self).__init__()

        in_ch = self.in_channel
        out_ch = self.out_channel
        layers = []

        for _ in range(self.num_conv):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_ch = out_ch

        # We use average pooling here because it's hard to compute the kernel of max pooling
        layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)

@dataclass(eq=False) # This avoid lightning trainer try to hash the module
class VGG(Base):

    n_class: int = 10
    level: int = 2 # The number of conv layers per block
    pooling: int = 3 # The number of pooling, i.e. the number of blocks
    init_std: float = 2 ** 0.5

    def __post_init__(self):
        super(VGG, self).__init__()

        # Convolutional layers
        block_config = [self.level] * self.pooling # Ex. [2,2,2] in default setting
        blocks = self.create_convs(3, 32, block_config)
        out_size = blocks[-1].out_channel

        # Fully connected layers
        blocks.append(nn.AdaptiveAvgPool2d((1, 1)))
        blocks.append(nn.Flatten())
        blocks.append(nn.Linear(out_size, self.n_class))

        # Build the net and initialize it
        self.net = nn.Sequential(*blocks)
        self.init_weights(self.init_std)

    def forward(self, x):
        return self.net(x)

    def create_convs(self, in_ch, out_ch, block_config) -> List[BasicBlock]:

        blocks = []

        for num_conv in block_config:
            
            blocks.append(BasicBlock(in_ch, out_ch, num_conv))
            in_ch = out_ch
            
            # According to the paper of VGG, number of channel should not exceed 512
            out_ch = min(out_ch * 2, 512)  

        return blocks


