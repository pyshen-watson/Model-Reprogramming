import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, normal_
from typing import NoReturn
from colorama import Fore, Style
from collections.abc import Iterable

class Base(nn.Module):

    def __init__(self):
        super(Base, self).__init__()
        self.name = ""

    def forward(self, x: torch.Tensor) -> NoReturn:
        raise NotImplementedError("Forward method is not implemented in Base model")

    def set_name(self, name: str) -> nn.Module:
        self.name = name
        return self

    def load(self, path: str):
        try:
            self.load_state_dict(torch.load(path))
            print(
                f"{Fore.GREEN}✓ Successfully load weight from {path}.{Style.RESET_ALL}"
            )
            return self

        except Exception as e:
            print(f"{Fore.RED}✗ Fail to load weight from {path}: {e}.{Style.RESET_ALL}")
            raise ValueError()

    def save(self, path: str):
        try:
            torch.save(self.state_dict(), path)
            print(f"{Fore.GREEN}✓ Successfully save weight to {path}.{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}✗ Fail to save weight to {path}: {e}.{Style.RESET_ALL}")
            raise ValueError()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        return self

    def _initialize_weights(self, net):

        if not isinstance(net, Iterable):
            net = [net]
        
        for layer in net:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    normal_(layer.bias, 0, 0.1**0.5)
            elif isinstance(layer, nn.Parameter):
                kaiming_normal_(layer.data, mode="fan_out", nonlinearity="relu")

class BasicBlock(nn.Module):
    """
    This block will create a module according to the num_conv.
    It will look like: (CR)xn+P
    C is for Conv2d, R is for ReLU, n is num_conv and P is Average Pooling
    """

    def __init__(self, in_ch: int, out_ch: int, num_conv: int):
        super(BasicBlock, self).__init__()

        layers = []
        for _ in range(num_conv):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding="same"))
            layers.append(nn.ReLU())
            in_ch = out_ch

        # We use average pooling here because it's hard to compute the kernel of max pooling
        layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.backbone = nn.Sequential(*layers)
        self.out_ch = out_ch  # For the convenience of GAP in SourceBackbone

    def forward(self, x: torch.Tensor):
        return self.backbone(x)

class Backbone(Base):

    def __init__(self, n_class=10, level=1, pooling=3):
        super(Backbone, self).__init__()

        # Convolutional layers
        num_blocks = self._get_num_blocks(level, pooling)
        blocks = self._create_convs(3, 64, num_blocks)
        out_size = blocks[-1].out_ch

        # Fully connected layers
        blocks.append(nn.AdaptiveAvgPool2d((1, 1)))
        blocks.append(nn.Flatten())
        blocks.append(nn.Linear(out_size, n_class))

        # Build the net and initialize it
        self.net = nn.Sequential(*blocks)
        self._initialize_weights(self.net)

    def forward(self, x):
        return self.net(x)

    def _get_num_blocks(self, level:int, pooling:int):
        assert level >= 1 and level <= 3, "The level is recommended between 1~3"
        return [level] * pooling

    def _create_convs(self, in_ch, out_ch, num_blocks):

        blocks = []

        for num_conv in num_blocks:
            blocks.append(BasicBlock(in_ch, out_ch, num_conv))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)  # According to the paper of VGG, number of channel should not exceed 512

        return blocks