#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor gelu_custom_impl_npu(const at::Tensor& input) {
    at::Tensor result = at::empty_like(input);
    EXEC_NPU_CMD(aclnnGeluCustom, input, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gelu_custom", &gelu_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu_custom", &gelu_custom_impl_npu, "Gaussian Error Linear Unit");
}