import torch.nn as nn
from dataclasses import dataclass
from ..base import Base, BaseModule

class ResnetBlock(BaseModule):

    def __init__(self, in_ch, out_ch):
        super(ResnetBlock, self).__init__()

        self.main_path = nn.Sequential(
            Base.Act_fn(),
            Base.Conv(in_ch, out_ch),
            Base.Act_fn(),
            Base.Conv(out_ch, out_ch),
        )

        # Adjust the number of channels if it is mismatch
        self.shortcut = Base.Identity() if in_ch == out_ch else Base.Conv(in_ch, out_ch, ksize=1)

    def forward(self, x):
        return self.main_path(x) + self.shortcut(x)


class ResnetGroup(BaseModule):
    def __init__(self, in_ch, out_ch, n_blocks):
        super(ResnetGroup, self).__init__()

        # The first block inputs the input channel
        layers = [ResnetBlock(in_ch, out_ch)]

        # The rests inputs the output channel
        for _ in range(n_blocks - 1):
            layers.append(ResnetBlock(out_ch, out_ch))

        layers.append(Base.AvgPooling())
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass(eq=False)  # This avoid lightning trainer try to hash the module
class Resnet(BaseModule):

    input_size: tuple = (1, 3, 112, 112)
    width: int = 32  # The number of filter of the first conv layer
    level: int = 1  # The number of conv layers per group
    group: int = 1 # The number of groups
    n_class: int = 10

    def __post_init__(self):
        super(Resnet, self).__init__()
        pool_ksize = 3 if self.input_size[-1] > 32 else 2

        layers = [
            Base.Conv(3, self.width, ksize=7, stride=2),
            Base.AvgPooling(kernel_size=pool_ksize),
        ]

        in_ch = self.width
        out_ch = self.width * 2

        for _ in range(self.group):
            layers += [ResnetGroup(in_ch, out_ch, self.level)]
            in_ch = out_ch
            out_ch *= 2

        layers += [
            Base.GlobalAvgPooling(),
            Base.Flatten(),
            Base.Linear(in_ch, self.n_class),
        ]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
