import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs a GELU activation.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, approximate='none') -> torch.Tensor:
        """
        Applies GELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            approximate (str, optional): The gelu approximation algorithm to use: 'none'|'tanh'.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return torch.nn.functional.gelu(x, approximate=approximate)


def get_inputs():
    """返回 forward() 的输入参数列表"""
    batch_size = 32
    seq_len = 1024
    return [torch.randn(batch_size, seq_len), 'none']


def get_init_inputs():
    """返回 __init__() 的初始化参数列表"""
    return []
