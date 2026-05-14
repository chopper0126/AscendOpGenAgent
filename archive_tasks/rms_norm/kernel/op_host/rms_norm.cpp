// RMSNorm op_host — tiling calculation + strategy dispatch + kernel launch

#include <algorithm>
#include <cstdint>

#include <torch/extension.h>
#include <torch/library.h>

#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

// ---------------------------------------------------------------------------
// Tiling constants & struct (shared with kernel via OP_KERNEL_SRC)
// ---------------------------------------------------------------------------

constexpr int32_t DEFAULT_BLOCK_M = 64;
constexpr int32_t DEFAULT_ROW_FACTOR = 8;
constexpr int32_t DEFAULT_NUM_PHYSICAL_CORES = 20;

struct RmsNormTilingData {
    int32_t M;
    int32_t N;
    int32_t blockM;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t rowFactor;
    float eps;
    float invN;
};

// ---------------------------------------------------------------------------
// Forward declarations of kernel entry points (defined in op_kernel/rms_norm.cpp)
// ---------------------------------------------------------------------------

// merge_n strategy
extern "C" void rms_norm_MergeN_do_fp32(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);
extern "C" void rms_norm_MergeN_do_fp16(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);
extern "C" void rms_norm_MergeN_do_bf16(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

// single_row strategy
extern "C" void rms_norm_SingleRow_do_fp32(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);
extern "C" void rms_norm_SingleRow_do_fp16(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);
extern "C" void rms_norm_SingleRow_do_bf16(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

// splitd strategy
extern "C" void rms_norm_SplitD_do_fp32(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);
extern "C" void rms_norm_SplitD_do_fp16(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);
extern "C" void rms_norm_SplitD_do_bf16(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

using LaunchFn = void (*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);

// ---------------------------------------------------------------------------
// Host function
// ---------------------------------------------------------------------------

namespace ascend_kernel {

std::vector<at::Tensor> rms_norm(const at::Tensor &x, const at::Tensor &gamma, double eps)
{
    TORCH_CHECK(x.dim() == 2, "x must be [M, N]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be [N]");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
            x.scalar_type() == at::kBFloat16,
        "x must be float16, float32, or bfloat16");
    TORCH_CHECK(gamma.scalar_type() == x.scalar_type(),
        "gamma must have the same dtype as x");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(x.sizes()[1] == gamma.sizes()[0], "gamma shape mismatch");

    const auto m = static_cast<int32_t>(x.sizes()[0]);
    const auto n = static_cast<int32_t>(x.sizes()[1]);

    const int32_t mNum = (m + DEFAULT_BLOCK_M - 1) / DEFAULT_BLOCK_M;
    const int32_t usedCoreNum = std::min<int32_t>(DEFAULT_NUM_PHYSICAL_CORES, mNum);
    const int32_t tasksPerCore = (mNum + usedCoreNum - 1) / usedCoreNum;

    at::Tensor y = at::empty_like(x);
    at::Tensor invRms = at::empty({m}, x.options());

    // Pack tiling data
    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(RmsNormTilingData))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<RmsNormTilingData *>(tilingCpu.data_ptr());
    tiling->M = m;
    tiling->N = n;
    tiling->blockM = DEFAULT_BLOCK_M;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->rowFactor = DEFAULT_ROW_FACTOR;
    tiling->eps = static_cast<float>(eps);
    tiling->invN = 1.0f / static_cast<float>(n);
    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    // Strategy + dtype dispatch
    LaunchFn launch = nullptr;
    if (x.scalar_type() == at::kFloat) {
        if (n <= 1024) {
            launch = rms_norm_MergeN_do_fp32;
        } else if (n > 8192) {
            launch = rms_norm_SplitD_do_fp32;
        } else {
            launch = rms_norm_SingleRow_do_fp32;
        }
    } else if (x.scalar_type() == at::kHalf) {
        if (n <= 1024) {
            launch = rms_norm_MergeN_do_fp16;
        } else if (n > 8192) {
            launch = rms_norm_SplitD_do_fp16;
        } else {
            launch = rms_norm_SingleRow_do_fp16;
        }
    } else if (x.scalar_type() == at::kBFloat16) {
        if (n <= 1024) {
            launch = rms_norm_MergeN_do_bf16;
        } else if (n > 8192) {
            launch = rms_norm_SplitD_do_bf16;
        } else {
            launch = rms_norm_SingleRow_do_bf16;
        }
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }

    launch(
        usedCoreNum,
        aclStream,
        static_cast<uint8_t *>(const_cast<void *>(x.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(gamma.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(y.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(invRms.storage().data())),
        static_cast<uint8_t *>(const_cast<void *>(tilingNpu.storage().data())));

    return {y, invRms};
}

}  // namespace ascend_kernel
