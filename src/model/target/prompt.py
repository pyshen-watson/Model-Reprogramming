import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from dataclasses import dataclass
from ..base import Base

@dataclass(eq=False)
class ImagePaddingLayer(Base):

    source_size: int = 112
    target_size: int = 32

    def __post_init__(self):
        super(ImagePaddingLayer, self).__init__()
        assert (self.target_size - self.source_size) % 2 == 0, "Difference between target and source size must be even"

        self.pad = (self.source_size - self.target_size) // 2
        self.mask = torch.ones(3, self.source_size, self.source_size, requires_grad=False)
        self.mask[:, self.pad:self.pad+self.target_size, self.pad:self.pad+self.target_size] = 0

    def forward(self, x: torch.Tensor):
        return F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="constant", value=0)    
    

@dataclass(eq=False)
class VisualPromptLayer(Base):

    source_size: int = 112
    target_size: int = 32
    prompt: bool = True
    resize: bool = False

    def __post_init__(self):
        super(VisualPromptLayer, self).__init__()
        assert self.target_size <= self.source_size, "Target size must be less than or equal to source size"
        assert not (self.prompt and self.resize), "Prompt and resize cannot be both enabled."

        self.padding_layer = ImagePaddingLayer(self.source_size, self.target_size)
        mask = self.padding_layer.mask

        self.register_buffer("mask", mask)
        self.delta = nn.Parameter(torch.zeros_like(mask), requires_grad=True)
        self.resize_fn = partial(F.interpolate, size=self.source_size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor):

        if self.resize:
            return self.resize_fn(x)

        x = self.padding_layer(x)
        return x + F.sigmoid(self.delta) * self.mask if self.prompt else x

