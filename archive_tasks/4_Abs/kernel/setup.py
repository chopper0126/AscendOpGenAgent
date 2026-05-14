import os
import glob
import subprocess
from pathlib import Path

import setuptools
from setuptools.command.build_ext import build_ext
from torch_npu.utils.cpp_extension import NpuExtension

HERE = Path(__file__).resolve().parent
BUILD_DIR = HERE / "build"


class BuildExt(build_ext):
    def run(self):
        so_files = glob.glob(str(BUILD_DIR / "abs_custom_ext*.so"))
        if not so_files:
            os.makedirs(BUILD_DIR, exist_ok=True)
            soc_ver = os.environ.get("SOC_VERSION", "Ascend910B2")
            ascend = os.environ.get("ASCEND_HOME_PATH", "")
            subprocess.check_call(
                ["cmake", str(HERE),
                 f"-DSOC_VERSION={soc_ver}",
                 f"-DASCEND_CANN_PACKAGE_PATH={ascend}",
                 "-DCMAKE_BUILD_TYPE=Release"],
                cwd=BUILD_DIR)
            subprocess.check_call(
                ["make", f"-j{os.cpu_count()}"], cwd=BUILD_DIR)

        self.build_lib = str(BUILD_DIR)
        super().run()


setuptools.setup(
    name="abs_custom",
    version="0.1.0",
    description="Abs AscendC kernel — registers torch.ops.npu.abs_custom",
    ext_modules=[NpuExtension("abs_custom_ext", sources=[])],
    cmdclass={"build_ext": BuildExt},
    license="BSD 3-Clause",
    python_requires=">=3.8",
)
