import torch
import torch.nn as nn
from torch.nn.init import normal_
from torchinfo import summary
from .log import log_success, log_fail

class Base(nn.Module):

    def __init__(self):
        super(Base, self).__init__()
        self.net: nn.Sequential = None
    
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

    def summary(self, input_size):

        if self.net is None:
            log_fail("The model is not initialized.")
            
        try:
            model_summary = summary(self.net, 
                                    input_size=input_size, # NCHW 
                                    verbose=0, 
                                    depth=100)

            log_success("Export the model structure.")
            return model_summary

        except:
            log_fail("Fail to export the model structure. Please check the input size again, it should content the batch size.")

    def init_weights(self, std_w=2.0**0.5, std_b=0.1**0.5):

        for layer in self.net:

            width = None # this value is a flag

            # The width of convolutional layer is in_channels * kernel_size^2
            if isinstance(layer, nn.Conv2d):
                shape = layer.weight.shape
                width = shape[1] * shape[2] * shape[3]
            
            # The width of linear layer is the number of input features
            elif isinstance(layer, nn.Linear):
                width = layer.weight.shape[1]
            
            if width is not None:
                normal_(layer.weight, 0, std_w / (width**0.5))
                if layer.bias is not None:
                    normal_(layer.bias, 0, std_b)