#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

std::vector<at::Tensor> rms_norm(const at::Tensor &x, const at::Tensor &gamma, double eps);

} // namespace ascend_kernel

#endif // OPS_H
