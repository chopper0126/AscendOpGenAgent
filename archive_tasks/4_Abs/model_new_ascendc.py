import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"

try:
    import abs_custom_ext  # noqa: F401
except ImportError:
    import glob as _glob
    _libs = _glob.glob(str(_KERNEL_BUILD / "abs_custom_ext*.so"))
    if _libs:
        torch.ops.load_library(_libs[0])


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.npu.abs_custom(x)
