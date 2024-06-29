import torch
import torch.nn as nn
from typing import NoReturn
from colorama import Fore, Style


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


class VGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_conv):
        super(VGGBlock, self).__init__()

        layers = []
        for _ in range(num_conv):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
            )
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.backbone(x)


class VGG(Base):

    def __init__(
        self, input_size=224, linear_width=4096, n_class=10, num_blocks=[2, 2, 4, 4, 4]
    ):
        super(VGG, self).__init__()

        # Convolutional layers
        blocks = []
        in_channel = 3
        out_channel = 64
        output_size = input_size

        for num_conv in num_blocks:
            blocks.append(VGGBlock(in_channel, out_channel, num_conv))
            in_channel = out_channel
            out_channel = min(out_channel * 2, 512)
            output_size //= 2

        self.conv = nn.Sequential(*blocks)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(out_channel * (output_size * output_size), linear_width),
            nn.ReLU(),
            nn.Linear(linear_width, linear_width),
            nn.ReLU(),
            nn.Linear(linear_width, n_class),
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):

        sigma = 0.1**0.5

        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="relu"
                )
                if layer.bias is not None:
                    nn.init.normal_(layer.bias, 0, sigma)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="relu"
                )
                if layer.bias is not None:
                    nn.init.normal_(layer.bias, 0, sigma)


class DNN(Base):

    def __init__(self, n_class=10, num_layers=6, width=2048):
        super(DNN, self).__init__()

        self.n_class = n_class
        self.num_layers = num_layers

        # Define the backbone
        width_list = [32 * 32 * 3] + [width] * (num_layers - 1)
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(width_list[i], width_list[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width_list[-1], n_class))
        self.backbone = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        x = x.flatten(1)
        x = self.backbone(x)
        return x

    def _initialize_weights(self):
        for m in self.backbone:
            if isinstance(m, nn.Linear):
                std = (2.0 / m.weight.size(1)) ** 0.5
                nn.init.normal_(m.weight, mean=0, std=std)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0, std=0.1**0.5)


class CNN(Base):

    def __init__(
        self,
        n_class=10,
        num_layers=6,
        input_size=32,
        conv_width=64,
        sigma_w=2.0,
        sigma_b=0.1,
    ):
        super(CNN, self).__init__()

        self.n_class = n_class
        self.num_layers = num_layers
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b

        layers = [nn.Conv2d(3, conv_width, 3, padding=1), nn.ReLU()]

        for i in range(1, num_layers):
            layers.append(nn.Conv2d(conv_width, conv_width, 3, padding=1))
            layers.append(nn.ReLU())

        self.backbone = nn.Sequential(*layers)
        self.linear = nn.Linear(input_size * input_size * conv_width, n_class)
        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.linear(x)
        return x

    def _initialize_weights(self):
        for m in self.backbone:
            if isinstance(m, nn.Conv2d):
                mean = 0.0
                std = (self.sigma_w / m.weight.size(1)) ** 0.5
                nn.init.normal_(m.weight, mean, std)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean, self.sigma_b**0.5)

        std = (self.sigma_w / self.linear.weight.size(1)) ** 0.5
        nn.init.normal_(self.linear.weight, mean, std)
        nn.init.normal_(self.linear.bias, mean, self.sigma_b**0.5)
