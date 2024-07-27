import torch.nn as nn
from dataclasses import dataclass
from ..base import Base, BaseModule


@dataclass(eq=False)  # This avoid lightning trainer try to hash the module
class DNN(BaseModule):

    input_size: tuple = (1, 3, 32, 32)
    width: int = 2048  # The number of filter of the first conv layer
    level: int = 1  # The number of conv layers per group
    n_class: int = 10

    def __post_init__(self):
        super(DNN, self).__init__()

        in_feat_size = self.input_size[1] * self.input_size[2] * self.input_size[3]
        feat_list = [in_feat_size] + [self.width] * (self.level - 1) + [self.n_class]
        layers = [Base.Flatten()]

        for i in range(self.level):
            if i == self.level - 1:
                layers += [Base.Linear(feat_list[i], feat_list[i + 1])]
            else:
                layers += [Base.Linear(feat_list[i], feat_list[i + 1]), Base.Act_fn()]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
