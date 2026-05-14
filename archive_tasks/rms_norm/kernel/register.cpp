// Copyright (c) 2026 Huawei Technologies Co., Ltd
// Licensed under the BSD 3-Clause License

#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("rms_norm(Tensor x, Tensor gamma, float eps=1e-5) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("rms_norm", TORCH_FN(ascend_kernel::rms_norm));
}

}  // namespace
