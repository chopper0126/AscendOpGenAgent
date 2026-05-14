// Copyright (c) 2026 Huawei Technologies Co., Ltd
// Licensed under the BSD 3-Clause License

#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("abs_custom(Tensor self) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("abs_custom", TORCH_FN(ascend_kernel::abs_custom));
}

}  // namespace
