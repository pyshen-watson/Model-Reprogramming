import torch
import torch.nn as nn


class DNN6(nn.Module):

    def __init__(self, n_class=10):
        super(DNN6, self).__init__()

        self.backbone = nn.Sequential(
            nn.Linear(3072, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_class),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        x = x.flatten(1)
        x = self.backbone(x)
        return x


class CNN6(nn.Module):

    def __init__(self, n_class=10):
        super(CNN6, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(),
        )

        self.linear = nn.Linear(8 * 8 * 1024, n_class)

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.linear(x)
        return x
