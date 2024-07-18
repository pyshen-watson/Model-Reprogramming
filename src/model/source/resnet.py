import torch
import torch.nn as nn
from dataclasses import dataclass
from ..base import Base

@dataclass(eq=False) # This avoid lightning trainer try to hash the module
class ResnetBlock(Base):

    in_channel: int
    out_channel: int

    def __post_init__(self):
        super(ResnetBlock, self).__init__()

        in_ch = self.in_channel
        out_ch = self.out_channel

        # The convolutional layers start from relu
        self.convs = nn.Sequential(
            Base.Act_fn(),
            Base.Conv(in_ch, out_ch),
            Base.Act_fn(),
            Base.Conv(out_ch, out_ch),
        )
        self.init_weights(self.convs)

        # Adjust the number of channels if it is mismatch
        self.shortcut = nn.Sequential()
        if in_ch == out_ch:
            self.shortcut = Base.Conv(in_ch, out_ch, kernel_size=1)
            self.init_weights(self.shortcut)

    def forward(self, x: torch.Tensor):
        return self.convs(x) + self.shortcut(x)

@dataclass(eq=False) # This avoid lightning trainer try to hash the module
class ResnetGroup(Base):
    
    n: int
    in_channel: int
    out_channel: int

    def __post_init__(self):
        super(ResnetGroup, self).__init__()

        in_ch = self.in_channel
        out_ch = self.out_channel

        layers = [ResnetBlock(in_ch, out_ch)]
        for _ in range(self.n-1):
            layers.append(ResnetBlock(out_ch, out_ch))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

class Resnet(Base):
    def __init__(self, block_size, k, num_classes):
        super(WideResnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.group1 = WideResnetGroup(block_size, 16, int(16 * k))
        self.group2 = WideResnetGroup(block_size, int(16 * k), int(32 * k))
        self.group3 = WideResnetGroup(block_size, int(32 * k), int(64 * k))
        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(int(64 * k), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x



def WideResnet(block_size, k, num_classes):
  return stax.serial(
      stax.Conv(16, (3, 3), padding='SAME'),
      WideResnetGroup(block_size, int(16 * k)),
      WideResnetGroup(block_size, int(32 * k), (2, 2)),
      WideResnetGroup(block_size, int(64 * k), (2, 2)),
      stax.AvgPool((8, 8)),
      stax.Flatten(),
      stax.Dense(num_classes, 1., 0.))