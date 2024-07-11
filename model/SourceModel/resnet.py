import torch
import torch.nn as nn
from .vgg import Base


class ResidualBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, num_conv: int):
        super(ResidualBlock, self).__init__()

        layers = []
        for _ in range(num_conv):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding="same"))
            layers.append(nn.ReLU())
            in_ch = out_ch

        self.block = nn.Sequential(*layers)
