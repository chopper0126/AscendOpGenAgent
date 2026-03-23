import torch
import torch.nn as nn
import torch_npu
import custom_ops_lib

def module_fn(x: torch.Tensor) -> torch.Tensor:
    return custom_ops_lib.gelu_custom(x)

class ModelNew(nn.Module):

    def __init__(self):
        """
        Initializes the GELU model.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies GELU activation to the input tensor.

        Args:
        x (torch.Tensor): Input tensor of any shape.
        fn: Function to apply (defaults to module_fn)

        Returns:
        torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return fn(x)