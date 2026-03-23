import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Simple model that performs GELU activation.
    """
    def __init__(self):
        """
        Initializes the GELU model.
        """
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return F.gelu(x)

batch_size = 16
dim = 4096

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No initialization parameters needed