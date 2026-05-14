// AscendC Abs operator — host-side tiling

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_abs_custom.h"

namespace ascend_kernel {

constexpr int64_t CACHE_LINE_BYTE_LENGTH = 512;

at::Tensor abs_custom(const at::Tensor &self)
{
    TORCH_CHECK(self.scalar_type() == at::kHalf || self.scalar_type() == at::kFloat ||
                self.scalar_type() == at::kBFloat16,
                "abs_custom: only float16, bfloat16 and float32 are supported, got ", self.scalar_type());

    at::Tensor output = at::empty_like(self);

    int64_t totalLength = self.numel();
    if (totalLength == 0) {
        return output;
    }

    int64_t dtypeSize = self.element_size();

    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = static_cast<int64_t>(ascendc_platform->GetCoreNumAiv());
    uint64_t ubSize;
    ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t ubSizeLimit = static_cast<int64_t>(ubSize);

    // Block-level tiling (inter-core, Cache Line aligned)
    int64_t totalLengthCore = (totalLength + coreNum - 1) / coreNum;
    int64_t totalLengthCoreAlign = (totalLengthCore + CACHE_LINE_BYTE_LENGTH - 1)
                                    / CACHE_LINE_BYTE_LENGTH * CACHE_LINE_BYTE_LENGTH;

    int64_t usedCoreNum = (totalLength + totalLengthCoreAlign - 1) / totalLengthCoreAlign;
    int64_t formerNum = usedCoreNum - 1;
    int64_t formerLength = totalLengthCoreAlign;
    int64_t tailLength = totalLength - formerNum * formerLength;

    // UB-level tiling (intra-core)
    // bufferCoefficient: fp32=20, fp16/bf16=16
    int64_t bufferCoefficient = (dtypeSize == 4) ? 20 : 16;
    int64_t maxTileElements = ubSizeLimit / bufferCoefficient;
    int64_t alignElements = 32 / static_cast<int64_t>(dtypeSize);
    int64_t tileLength = (maxTileElements / alignElements) * alignElements;

    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);

    EXEC_KERNEL_CMD(abs_custom, blockDim,
                    self, output,
                    formerNum, formerLength, tailLength, tileLength, dtypeSize);

    return output;
}

}  // namespace ascend_kernel
