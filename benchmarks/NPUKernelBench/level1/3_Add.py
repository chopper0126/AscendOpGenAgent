import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs element-wise addition with broadcasting support.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Applies element-wise addition to the input tensors with broadcasting support.

        Args:
            x (torch.Tensor): First input tensor of any shape.
            y (torch.Tensor): Second input tensor, broadcastable with x.
            alpha (float, optional): The multiplier for y.

        Returns:
            torch.Tensor: Output tensor x + alpha * y, shape follows broadcasting rules.
        """
        return torch.add(x, y, alpha=alpha)

def get_inputs():
    """返回 forward() 的输入参数列表"""
    batch_size = 128
    seq_len = 128
    return [torch.randn(batch_size, seq_len), torch.randn(batch_size, seq_len)]


def get_init_inputs():
    """返回 __init__() 的初始化参数列表"""
    return []
