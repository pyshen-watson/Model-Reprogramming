import torch
import torch.nn as nn
from torch.nn.init import normal_
from torchinfo import summary
from ..common import log_success, log_fail


class Base:

    @staticmethod
    def Conv(in_ch, out_ch, ksize=3, stride=1, W_std=2.0**0.5, b_std=0.1**0.5):

        padding = (ksize - 1) // 2  # Same size padding
        conv = nn.Conv2d(in_ch, out_ch, ksize, stride, padding)

        # Initialize the weight and bias following NTK papers
        width = in_ch * ksize**2
        normal_(conv.weight, 0, W_std / (width**0.5))
        normal_(conv.bias, 0, b_std)

        return conv

    @staticmethod
    def Linear(in_feat, out_feat, W_std=2.0**0.5, b_std=0.1**0.5):

        linear = nn.Linear(in_feat, out_feat)

        # Initialize the weight and bias following NTK papers
        normal_(linear.weight, 0, W_std / (in_feat**0.5))
        normal_(linear.bias, 0, b_std)
        return linear

    @staticmethod
    def Act_fn():
        return nn.ReLU()

    @staticmethod
    def AvgPooling(kernel_size=2, stride=2):
        return nn.AvgPool2d(kernel_size, stride)

    @staticmethod
    def GlobalAvgPooling():
        return nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def Flatten():
        return nn.Flatten()


class BaseModule(nn.Module):

    def __init__(self):
        super(BaseModule, self).__init__()
        self.input_size = (1, 3, 112, 112)

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("Forward method is not implemented in Base model")

    def load(self, path: str):
        try:
            self.load_state_dict(torch.load(path))
            log_success(f"Successfully load weight from {path}.")
            return self

        except Exception as e:
            log_fail(f"Fail to load weight from {path}: {e}.")

    def save(self, path: str):
        try:
            torch.save(self.state_dict(), path)
            log_success(f"Successfully save weight to {path}.")

        except Exception as e:
            log_fail(f"Fail to save weight to {path}: {e}.")

    def freeze(self):
        try:
            for param in self.parameters():
                param.requires_grad = False
            log_success("Freezed the gradient of the model.")
            return self
        except:
            log_fail("Fail to freeze the gradient of the model.")

    def summary(self):

        try:
            model_summary = summary(self, self.input_size, depth=100)
            log_success("Export the model structure.")
            return model_summary

        except:
            log_fail("Fail to export the model structure.")
