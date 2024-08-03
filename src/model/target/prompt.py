import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from dataclasses import dataclass
from ..base import Base, BaseModule

class FullyConnectedLayer(BaseModule):

    def __init__(self, in_size, out_size):
        super(FullyConnectedLayer, self).__init__()

        self.out_size = out_size
        self.fc_layer = nn.Sequential(
            Base.Flatten(),
            Base.Linear(3 * in_size * in_size, 3 * out_size * out_size)
        )

    def forward(self, x: torch.Tensor):
        x = self.fc_layer(x)
        x = x.reshape(-1, 3, self.out_size, self.out_size)
        return x


@dataclass(eq=False)
class VisualPromptLayer(BaseModule):

    source_size: int = 112
    target_size: int = 64
    fc: bool = False
    vp: bool = False
    device: int = 0

    def __post_init__(self):
        super(VisualPromptLayer, self).__init__()
        assert self.target_size <= self.source_size, "Target size must be less than or equal to source size"
        assert (self.target_size - self.source_size) % 2 == 0, "Difference between target and source size must be even"
        self.input_size = (1, 3, self.target_size, self.target_size) # For the profiler

        # ==================== The FC part ====================
        # If vp in enable, fc keep the target data shape. 
        # Otherwise, the output shape of fc will be the same as source data
        fc_out_size = self.target_size if self.vp else self.source_size
        # self.fc_layer = FullyConnectedLayer(self.target_size, fc_out_size)

        # ==================== The VP part ====================
        pad = (self.source_size - self.target_size) // 2
        self.mask = torch.ones(3, self.source_size, self.source_size, requires_grad=False).to(self.device)
        self.mask[:, pad:pad+self.target_size, pad:pad+self.target_size] = 0
        self.delta = nn.Parameter(torch.zeros_like(self.mask), requires_grad=True)
        self.padding = partial(F.pad, pad=(pad, pad, pad, pad), value=0)

    @property
    def norm(self):
        return (F.sigmoid(self.delta) * self.mask).norm(1)

    def forward(self, x: torch.Tensor):
        # x = self.fc_layer(x) if self.fc else x
        x = self.padding(x)  if self.vp else x
        x = x + F.sigmoid(self.delta) * self.mask if self.vp else x
        return x
    


