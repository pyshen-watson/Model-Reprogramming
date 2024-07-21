import torch
import torch.nn as nn
from dataclasses import dataclass
from ..base import BaseModule


@dataclass(eq=False)  # This avoid lightning trainer try to hash the module
class ResnetBlock(BaseModule):

    in_channel: int
    out_channel: int

    def __post_init__(self):
        super(ResnetBlock, self).__init__()

        # aliases
        in_ch = self.in_channel
        out_ch = self.out_channel

        self.main_path = nn.Sequential(
            BaseModule.Conv(in_ch, out_ch),
            BaseModule.Act_fn(),
            BaseModule.Conv(out_ch, out_ch),
            BaseModule.Act_fn(),
        )
        self.init_weights(self.main_path)

        # Adjust the number of channels if it is mismatch
        self.shortcut_path = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut_path = BaseModule.Conv(in_ch, out_ch, ksize=1)
            self.init_weights(self.shortcut_path)

    def forward(self, x):
        return self.main_path(x) + self.shortcut_path(x)


@dataclass(eq=False)  # This avoid lightning trainer try to hash the module
class ResnetGroup(BaseModule):

    n_blocks: int
    in_channel: int
    out_channel: int

    def __post_init__(self):
        super(ResnetGroup, self).__init__()

        # aliases
        in_ch = self.in_channel
        out_ch = self.out_channel
        print(f"ResnetGroup: {in_ch} -> {out_ch}")

        layers = [ResnetBlock(in_ch, out_ch)]
        for _ in range(self.n_blocks - 1):
            layers.append(ResnetBlock(out_ch, out_ch))
        layers.append(BaseModule.AvgPooling())

        self.net = nn.Sequential(*layers)
        self.init_weights(self.net)

    def forward(self, x):
        return self.net(x)


@dataclass(eq=False)  # This avoid lightning trainer try to hash the module
class Resnet(BaseModule):

    input_size: tuple  # (N, C, H, W)
    n_class: int = 10
    pooling: int = 3  # The number of pooling, i.e. the number of blocks
    level: int = 3  # The number of conv layers per block
    width_base: int = 32  # The number of filter of the first convolutional layer

    def __post_init__(self):
        super(Resnet, self).__init__()

        in_ch = 3
        out_ch = self.width_base
        layers = [ BaseModule.Conv(in_ch, out_ch, ksize=7, stride=2), 
                  BaseModule.Act_fn(), 
                  BaseModule.AvgPooling(kernel_size=3) ]
        in_ch = out_ch

        for _ in range(self.pooling):
            layers += [ ResnetGroup(self.level, in_ch, out_ch) ]
            in_ch = out_ch
            out_ch *= 2
        
        layers += [ BaseModule.GlobalAvgPooling(), BaseModule.Flatten(), BaseModule.Linear(in_ch, self.n_class), ]

        self.net = nn.Sequential(*layers)
        self.init_weights(self.net)

    def forward(self, x):
        return self.net(x)
