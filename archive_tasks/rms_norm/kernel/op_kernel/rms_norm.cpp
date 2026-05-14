// AscendC RMSNorm kernel — merged strategies: merge_n, single_row, splitd
// Supports float32, float16, bfloat16

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"

// ---------------------------------------------------------------------------
// Tiling & constants
// ---------------------------------------------------------------------------

constexpr uint32_t DEFAULT_BLOCK_M = 64;
constexpr uint32_t DEFAULT_ROW_FACTOR = 8;
constexpr uint32_t DEFAULT_NUM_PHYSICAL_CORES = 20;

struct RmsNormKernelTiling {
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
// Inline helpers (from kernel_common.h / vector_tile.h)
// ---------------------------------------------------------------------------

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    return (a + b - 1U) / b;
}

template <typename T>
__aicore__ inline void CopyTiling(T *tiling, GM_ADDR tilingGM)
{
    int32_t *dst = reinterpret_cast<int32_t *>(tiling);
    auto *src = reinterpret_cast<__gm__ int32_t *>(tilingGM);
    for (size_t i = 0; i < sizeof(T) / sizeof(int32_t); ++i) {
        dst[i] = src[i];
    }
}

template <typename T>
__aicore__ inline void LoadGmToUb(
    AscendC::LocalTensor<T> &dst,
    AscendC::GlobalTensor<T> src,
    uint32_t count)
{
    AscendC::DataCopyExtParams copyParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, static_cast<T>(0)};
    AscendC::DataCopyPad(dst, src, copyParams, padParams);
}

template <typename T>
__aicore__ inline void StoreUbToGm(
    AscendC::GlobalTensor<T> dst,
    AscendC::LocalTensor<T> &src,
    uint32_t count)
{
    AscendC::DataCopyExtParams copyParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams);
}

// =========================================================================
// Strategy 1: merge_n — preferred when N <= 1024
// =========================================================================

template <typename dataType>
class RmsNormMergeNKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR invRms, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(x), tiling_.M * tiling_.N);
        gammaGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gamma), tiling_.N);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(y), tiling_.M * tiling_.N);
        invRmsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(invRms), tiling_.M);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            subBlockRows_ = tiling_.blockM / AscendC::GetSubBlockNum();
            rowLoops_ = subBlockRows_ / tiling_.rowFactor;
            pipe_->InitBuffer(gammaBuf_, tiling_.N * sizeof(dataType));
            pipe_->InitBuffer(xInQueue_, 1, kMergeRowFactor * tiling_.N * sizeof(dataType));
            pipe_->InitBuffer(yOutQueue_, 1, kMergeRowFactor * tiling_.N * sizeof(dataType));
            pipe_->InitBuffer(invRmsOutQueue_, 1, kMergeRowFactor * sizeof(dataType));
            pipe_->InitBuffer(scaleBuf_, kMergeRowFactor * tiling_.N * sizeof(float));
            pipe_->InitBuffer(gammaTileBuf_, kMergeRowFactor * tiling_.N * sizeof(float));
            pipe_->InitBuffer(gammaBroadcastTmpBuf_, 2 * kMergeRowFactor * tiling_.N * sizeof(uint8_t));
            pipe_->InitBuffer(scaleBroadcastTmpBuf_, 2 * kMergeRowFactor * tiling_.N * sizeof(uint8_t));
            pipe_->InitBuffer(reduceBuf_, 2 * kMergeRowFactor * tiling_.N * sizeof(uint8_t));
            pipe_->InitBuffer(sumBuf_, 16 * sizeof(float));
            pipe_->InitBuffer(invRmsBuf_, kMergeRowFactor * sizeof(float));
            if constexpr (!std::is_same_v<dataType, float>) {
                pipe_->InitBuffer(xCastBuf_, kMergeRowFactor * tiling_.N * sizeof(float));
                pipe_->InitBuffer(gammaCastBuf_, tiling_.N * sizeof(float));
                pipe_->InitBuffer(yCastBuf_, kMergeRowFactor * tiling_.N * sizeof(float));
            }

            const uint32_t gammaSrcShape[2] = {1U, static_cast<uint32_t>(tiling_.N)};
            const uint32_t gammaDstShape[2] = {
                static_cast<uint32_t>(kMergeRowFactor),
                static_cast<uint32_t>(tiling_.N),
            };

            gammaInLocal_ = gammaBuf_.Get<dataType>();
            LoadGmToUb(gammaInLocal_, gammaGM_, static_cast<uint32_t>(tiling_.N));
            AscendC::PipeBarrier<PIPE_MTE2>();
            AscendC::PipeBarrier<PIPE_ALL>();
            PrepareInputTensor(gammaLocal_, gammaInLocal_, gammaCastBuf_, tiling_.N);

            gammaTileLocal_ = gammaTileBuf_.Get<float>();
            gammaBroadcastTmpLocal_ = gammaBroadcastTmpBuf_.Get<uint8_t>();
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Broadcast<float, 2, 0>(
                gammaTileLocal_,
                gammaLocal_,
                gammaDstShape,
                gammaSrcShape,
                gammaBroadcastTmpLocal_);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int subBlockIdx = AscendC::GetSubBlockIdx();

            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                const int bx = coreIdx * tiling_.tasksPerCore + localIdx;
                if (bx >= BlockCount()) {
                    continue;
                }

                for (int r = 0; r < rowLoops_; ++r) {
                    const int rowBase = bx * tiling_.blockM + subBlockIdx * subBlockRows_ + r * tiling_.rowFactor;
                    const int validRows = tiling_.M - rowBase;
                    if (validRows > 0) {
                        ProcessRows(rowBase, validRows >= kMergeRowFactor ? kMergeRowFactor : validRows);
                    }
                }
            }
        }
    }

private:
    static constexpr int kMergeRowFactor = 8;

    __aicore__ inline int32_t BlockCount() const
    {
        return (tiling_.M + tiling_.blockM - 1) / tiling_.blockM;
    }

    template <typename T>
    __aicore__ inline AscendC::RoundMode OutputRoundMode() const
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            return AscendC::RoundMode::CAST_ROUND;
        }
        return AscendC::RoundMode::CAST_NONE;
    }

    __aicore__ inline void PrepareInputTensor(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &src,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &castBuf,
        int32_t count)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            dst = src.template ReinterpretCast<float>();
        } else {
            dst = castBuf.Get<float>();
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PrepareOutputTensor(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &out,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &castBuf)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            dst = out.template ReinterpretCast<float>();
        } else {
            dst = castBuf.Get<float>();
        }
    }

    __aicore__ inline void PrepareInvRmsTensor(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &out)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            dst = out.template ReinterpretCast<float>();
        } else {
            dst = invRmsBuf_.Get<float>();
        }
    }

    __aicore__ inline void FinalizeOutputTensor(
        AscendC::LocalTensor<dataType> &out,
        AscendC::LocalTensor<float> &src,
        int32_t count)
    {
        if constexpr (!std::is_same_v<dataType, float>) {
            AscendC::Cast(out, src, OutputRoundMode<dataType>(), count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyInX(int32_t rowBase, int32_t count)
    {
        xInQueue_.AllocTensor<dataType>(xInLocal_);
        LoadGmToUb(xInLocal_, xGM_[rowBase * tiling_.N], static_cast<uint32_t>(count));
        xInQueue_.EnQue(xInLocal_);
    }

    __aicore__ inline void CopyOutY(int32_t rowBase, int32_t count)
    {
        yOutQueue_.DeQue<dataType>(yOutLocal_);
        StoreUbToGm(yGM_[rowBase * tiling_.N], yOutLocal_, static_cast<uint32_t>(count));
        yOutQueue_.FreeTensor(yOutLocal_);
    }

    __aicore__ inline void CopyOutInvRms(int32_t rowBase, int32_t count)
    {
        invRmsOutQueue_.DeQue<dataType>(invRmsOutLocal_);
        StoreUbToGm(invRmsGM_[rowBase], invRmsOutLocal_, static_cast<uint32_t>(count));
        invRmsOutQueue_.FreeTensor(invRmsOutLocal_);
    }

    __aicore__ inline void ComputeSingleRow()
    {
        const uint32_t reduceShape[2] = {1U, static_cast<uint32_t>(tiling_.N)};
        const uint32_t scaleSrcShape[2] = {1U, 1U};
        const uint32_t scaleDstShape[2] = {1U, static_cast<uint32_t>(tiling_.N)};

        yOutQueue_.AllocTensor<dataType>(yOutLocal_);
        scaleLocal_ = scaleBuf_.Get<float>();
        reduceTmpLocal_ = reduceBuf_.Get<uint8_t>();
        sumLocal_ = sumBuf_.Get<float>();
        invRmsOutQueue_.AllocTensor<dataType>(invRmsOutLocal_);
        PrepareInvRmsTensor(invRmsLocal_, invRmsOutLocal_);
        scaleBroadcastTmpLocal_ = scaleBroadcastTmpBuf_.Get<uint8_t>();

        xInQueue_.DeQue<dataType>(xInLocal_);
        PrepareInputTensor(xLocal_, xInLocal_, xCastBuf_, tiling_.N);
        PrepareOutputTensor(yLocal_, yOutLocal_, yCastBuf_);

        AscendC::Mul(yLocal_, xLocal_, xLocal_, tiling_.N);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
            sumLocal_, yLocal_, reduceTmpLocal_, reduceShape, true);
        AscendC::Muls(sumLocal_, sumLocal_, tiling_.invN, 1);
        AscendC::Adds(sumLocal_, sumLocal_, tiling_.eps, 1);
        AscendC::Rsqrt(invRmsLocal_, sumLocal_, 1);
        FinalizeOutputTensor(invRmsOutLocal_, invRmsLocal_, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Broadcast<float, 2, 1>(
            scaleLocal_, invRmsLocal_, scaleDstShape, scaleSrcShape, scaleBroadcastTmpLocal_);

        AscendC::Mul(yLocal_, xLocal_, scaleLocal_, tiling_.N);
        AscendC::Mul(yLocal_, yLocal_, gammaLocal_, tiling_.N);
        FinalizeOutputTensor(yOutLocal_, yLocal_, tiling_.N);

        xInQueue_.FreeTensor(xInLocal_);
        yOutQueue_.EnQue(yOutLocal_);
        invRmsOutQueue_.EnQue(invRmsOutLocal_);
    }

    __aicore__ inline void ComputeRows(int32_t tileSize)
    {
        const uint32_t reduceShape[2] = {
            static_cast<uint32_t>(kMergeRowFactor),
            static_cast<uint32_t>(tiling_.N),
        };
        const uint32_t scaleSrcShape[2] = {
            static_cast<uint32_t>(kMergeRowFactor),
            1U,
        };
        const uint32_t scaleDstShape[2] = {
            static_cast<uint32_t>(kMergeRowFactor),
            static_cast<uint32_t>(tiling_.N),
        };

        yOutQueue_.AllocTensor<dataType>(yOutLocal_);
        scaleLocal_ = scaleBuf_.Get<float>();
        reduceTmpLocal_ = reduceBuf_.Get<uint8_t>();
        sumLocal_ = sumBuf_.Get<float>();
        invRmsOutQueue_.AllocTensor<dataType>(invRmsOutLocal_);
        PrepareInvRmsTensor(invRmsLocal_, invRmsOutLocal_);
        scaleBroadcastTmpLocal_ = scaleBroadcastTmpBuf_.Get<uint8_t>();

        xInQueue_.DeQue<dataType>(xInLocal_);
        PrepareInputTensor(xLocal_, xInLocal_, xCastBuf_, tileSize);
        PrepareOutputTensor(yLocal_, yOutLocal_, yCastBuf_);

        AscendC::Mul(yLocal_, xLocal_, xLocal_, tileSize);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
            sumLocal_, yLocal_, reduceTmpLocal_, reduceShape, true);
        AscendC::Muls(sumLocal_, sumLocal_, tiling_.invN, kMergeRowFactor);
        AscendC::Adds(sumLocal_, sumLocal_, tiling_.eps, kMergeRowFactor);
        AscendC::Rsqrt(invRmsLocal_, sumLocal_, kMergeRowFactor);
        FinalizeOutputTensor(invRmsOutLocal_, invRmsLocal_, kMergeRowFactor);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Broadcast<float, 2, 1>(
            scaleLocal_, invRmsLocal_, scaleDstShape, scaleSrcShape, scaleBroadcastTmpLocal_);

        AscendC::Mul(yLocal_, xLocal_, scaleLocal_, tileSize);
        AscendC::Mul(yLocal_, yLocal_, gammaTileLocal_, tileSize);
        FinalizeOutputTensor(yOutLocal_, yLocal_, tileSize);

        xInQueue_.FreeTensor(xInLocal_);
        yOutQueue_.EnQue(yOutLocal_);
        invRmsOutQueue_.EnQue(invRmsOutLocal_);
    }

    __aicore__ inline void ProcessSingleRow(int rowIdx)
    {
        CopyInX(rowIdx, tiling_.N);
        ComputeSingleRow();
        CopyOutInvRms(rowIdx, 1);
        CopyOutY(rowIdx, tiling_.N);
    }

    __aicore__ inline void ProcessRows(int rowBase, int validRows)
    {
        if (validRows < kMergeRowFactor) {
            for (int r = 0; r < validRows; ++r) {
                ProcessSingleRow(rowBase + r);
            }
            return;
        }

        const int tileSize = kMergeRowFactor * tiling_.N;
        CopyInX(rowBase, tileSize);
        ComputeRows(tileSize);
        CopyOutInvRms(rowBase, validRows);
        CopyOutY(rowBase, tileSize);
    }

private:
    RmsNormKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int subBlockRows_{0};
    int rowLoops_{0};

    AscendC::GlobalTensor<dataType> xGM_;
    AscendC::GlobalTensor<dataType> gammaGM_;
    AscendC::GlobalTensor<dataType> yGM_;
    AscendC::GlobalTensor<dataType> invRmsGM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaBuf_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> yOutQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> invRmsOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scaleBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaTileBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaBroadcastTmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scaleBroadcastTmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> invRmsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yCastBuf_;

    AscendC::LocalTensor<dataType> gammaInLocal_;
    AscendC::LocalTensor<dataType> xInLocal_;
    AscendC::LocalTensor<dataType> yOutLocal_;
    AscendC::LocalTensor<dataType> invRmsOutLocal_;
    AscendC::LocalTensor<float> gammaLocal_;
    AscendC::LocalTensor<float> xLocal_;
    AscendC::LocalTensor<float> yLocal_;
    AscendC::LocalTensor<float> scaleLocal_;
    AscendC::LocalTensor<float> gammaTileLocal_;
    AscendC::LocalTensor<float> invRmsLocal_;
    AscendC::LocalTensor<uint8_t> gammaBroadcastTmpLocal_;
    AscendC::LocalTensor<uint8_t> scaleBroadcastTmpLocal_;
    AscendC::LocalTensor<uint8_t> reduceTmpLocal_;
    AscendC::LocalTensor<float> sumLocal_;
};

// =========================================================================
// Strategy 2: single_row — preferred when 1024 < N <= 8192
// =========================================================================

template <typename dataType>
class RmsNormSingleRowKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR invRms, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(x), tiling_.M * tiling_.N);
        gammaGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gamma), tiling_.N);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(y), tiling_.M * tiling_.N);
        invRmsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(invRms), tiling_.M);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            subBlockRows_ = tiling_.blockM / AscendC::GetSubBlockNum();
            pipe_->InitBuffer(gammaBuf_, tiling_.N * sizeof(dataType));
            pipe_->InitBuffer(xInQueue_, 1, tiling_.N * sizeof(dataType));
            pipe_->InitBuffer(yOutQueue_, 1, tiling_.N * sizeof(dataType));
            pipe_->InitBuffer(invRmsOutQueue_, 1, sizeof(dataType));
            pipe_->InitBuffer(reduceBuf_, tiling_.N * sizeof(float));
            pipe_->InitBuffer(sumBuf_, 16 * sizeof(float));
            pipe_->InitBuffer(invRmsBuf_, sizeof(float));
            if constexpr (!std::is_same_v<dataType, float>) {
                pipe_->InitBuffer(xCastBuf_, tiling_.N * sizeof(float));
                pipe_->InitBuffer(gammaCastBuf_, tiling_.N * sizeof(float));
                pipe_->InitBuffer(yCastBuf_, tiling_.N * sizeof(float));
            }

            gammaInLocal_ = gammaBuf_.Get<dataType>();
            LoadGmToUb(gammaInLocal_, gammaGM_, static_cast<uint32_t>(tiling_.N));
            AscendC::PipeBarrier<PIPE_MTE2>();
            AscendC::PipeBarrier<PIPE_ALL>();
            PrepareInputTensor(gammaLocal_, gammaInLocal_, gammaCastBuf_, tiling_.N);
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int subBlockIdx = AscendC::GetSubBlockIdx();

            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                const int bx = coreIdx * tiling_.tasksPerCore + localIdx;
                if (bx >= BlockCount()) {
                    continue;
                }

                for (int row = 0; row < subBlockRows_; ++row) {
                    const int rowIdx = bx * tiling_.blockM + subBlockIdx * subBlockRows_ + row;
                    if (rowIdx < tiling_.M) {
                        ProcessRow(rowIdx);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline int32_t BlockCount() const
    {
        return (tiling_.M + tiling_.blockM - 1) / tiling_.blockM;
    }

    __aicore__ inline AscendC::RoundMode OutputRoundMode() const
    {
        if constexpr (std::is_same_v<dataType, bfloat16_t>) {
            return AscendC::RoundMode::CAST_ROUND;
        }
        return AscendC::RoundMode::CAST_NONE;
    }

    __aicore__ inline void PrepareInputTensor(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &src,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &castBuf,
        int32_t count)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            dst = src.template ReinterpretCast<float>();
        } else {
            dst = castBuf.Get<float>();
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PrepareOutputTensor(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &out,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &castBuf,
        int32_t count)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            dst = out.template ReinterpretCast<float>();
        } else {
            dst = castBuf.Get<float>();
            AscendC::Duplicate(out, static_cast<dataType>(0), count);
        }
    }

    __aicore__ inline void FinalizeOutputTensor(
        AscendC::LocalTensor<dataType> &out,
        AscendC::LocalTensor<float> &src,
        int32_t count)
    {
        if constexpr (!std::is_same_v<dataType, float>) {
            AscendC::Cast(out, src, OutputRoundMode(), count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PrepareInvRmsTensor(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &out)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            dst = out.template ReinterpretCast<float>();
        } else {
            dst = invRmsBuf_.Get<float>();
        }
    }

    __aicore__ inline void CopyInX(int32_t rowIdx)
    {
        xInQueue_.AllocTensor<dataType>(xInLocal_);
        LoadGmToUb(xInLocal_, xGM_[rowIdx * tiling_.N], static_cast<uint32_t>(tiling_.N));
        xInQueue_.EnQue(xInLocal_);
    }

    __aicore__ inline void CopyOutY(int32_t rowIdx)
    {
        yOutQueue_.DeQue<dataType>(yOutLocal_);
        StoreUbToGm(yGM_[rowIdx * tiling_.N], yOutLocal_, static_cast<uint32_t>(tiling_.N));
        yOutQueue_.FreeTensor(yOutLocal_);
    }

    __aicore__ inline void CopyOutInvRms(int32_t rowIdx)
    {
        invRmsOutQueue_.DeQue<dataType>(invRmsOutLocal_);
        StoreUbToGm(invRmsGM_[rowIdx], invRmsOutLocal_, 1);
        invRmsOutQueue_.FreeTensor(invRmsOutLocal_);
    }

    __aicore__ inline void ComputeRow()
    {
        yOutQueue_.AllocTensor<dataType>(yOutLocal_);
        invRmsOutQueue_.AllocTensor<dataType>(invRmsOutLocal_);
        PrepareInvRmsTensor(invRmsLocal_, invRmsOutLocal_);
        reduceLocal_ = reduceBuf_.Get<float>();
        sumLocal_ = sumBuf_.Get<float>();

        xInQueue_.DeQue<dataType>(xInLocal_);
        PrepareInputTensor(xLocal_, xInLocal_, xCastBuf_, tiling_.N);
        PrepareOutputTensor(yLocal_, yOutLocal_, yCastBuf_, tiling_.N);

        AscendC::Mul(yLocal_, xLocal_, xLocal_, tiling_.N);
        AscendC::ReduceSum<float>(sumLocal_, yLocal_, reduceLocal_, tiling_.N);

        float meanSq = sumLocal_.GetValue(0) * tiling_.invN + tiling_.eps;
        AscendC::Duplicate(sumLocal_, meanSq, 1);
        AscendC::Rsqrt(invRmsLocal_, sumLocal_, 1);
        AscendC::PipeBarrier<PIPE_ALL>();
        float invRms = invRmsLocal_.GetValue(0);
        AscendC::PipeBarrier<PIPE_ALL>();
        FinalizeOutputTensor(invRmsOutLocal_, invRmsLocal_, 1);

        AscendC::Muls(yLocal_, xLocal_, invRms, tiling_.N);
        AscendC::Mul(yLocal_, yLocal_, gammaLocal_, tiling_.N);
        FinalizeOutputTensor(yOutLocal_, yLocal_, tiling_.N);

        xInQueue_.FreeTensor(xInLocal_);
        yOutQueue_.EnQue(yOutLocal_);
        invRmsOutQueue_.EnQue(invRmsOutLocal_);
    }

    __aicore__ inline void ProcessRow(int rowIdx)
    {
        CopyInX(rowIdx);
        ComputeRow();
        CopyOutInvRms(rowIdx);
        CopyOutY(rowIdx);
    }

private:
    RmsNormKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int subBlockRows_{0};

    AscendC::GlobalTensor<dataType> xGM_;
    AscendC::GlobalTensor<dataType> gammaGM_;
    AscendC::GlobalTensor<dataType> yGM_;
    AscendC::GlobalTensor<dataType> invRmsGM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaBuf_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> yOutQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> invRmsOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> invRmsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yCastBuf_;

    AscendC::LocalTensor<dataType> gammaInLocal_;
    AscendC::LocalTensor<dataType> xInLocal_;
    AscendC::LocalTensor<dataType> yOutLocal_;
    AscendC::LocalTensor<dataType> invRmsOutLocal_;
    AscendC::LocalTensor<float> gammaLocal_;
    AscendC::LocalTensor<float> xLocal_;
    AscendC::LocalTensor<float> yLocal_;
    AscendC::LocalTensor<float> invRmsLocal_;
    AscendC::LocalTensor<float> reduceLocal_;
    AscendC::LocalTensor<float> sumLocal_;
};

// =========================================================================
// Strategy 3: splitd — preferred when N > 8192
// =========================================================================

template <typename dataType>
class RmsNormSplitDKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR invRms, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(x), tiling_.M * tiling_.N);
        gammaGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gamma), tiling_.N);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(y), tiling_.M * tiling_.N);
        invRmsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(invRms), tiling_.M);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            subBlockRows_ = tiling_.blockM / AscendC::GetSubBlockNum();
            pipe_->InitBuffer(xInQueue_, 1, kBlockN * sizeof(dataType));
            pipe_->InitBuffer(gammaInQueue_, 1, kBlockN * sizeof(dataType));
            pipe_->InitBuffer(yOutQueue_, 1, kBlockN * sizeof(dataType));
            pipe_->InitBuffer(invRmsOutQueue_, 1, sizeof(dataType));
            pipe_->InitBuffer(reduceBuf_, kBlockN * sizeof(float));
            pipe_->InitBuffer(sumBuf_, 16 * sizeof(float));
            pipe_->InitBuffer(tempBuf_, kTileFloatBytes);
            pipe_->InitBuffer(invRmsBuf_, sizeof(float));
            if constexpr (!std::is_same_v<dataType, float>) {
                pipe_->InitBuffer(xCastBuf_, kTileFloatBytes);
                pipe_->InitBuffer(gammaCastBuf_, kTileFloatBytes);
                pipe_->InitBuffer(yCastBuf_, kTileFloatBytes);
            }
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int subBlockIdx = AscendC::GetSubBlockIdx();

            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                const int bx = coreIdx * tiling_.tasksPerCore + localIdx;
                if (bx >= BlockCount()) {
                    continue;
                }

                for (int row = 0; row < subBlockRows_; ++row) {
                    const int rowIdx = bx * tiling_.blockM + subBlockIdx * subBlockRows_ + row;
                    if (rowIdx < tiling_.M) {
                        ProcessRow(rowIdx);
                    }
                }
            }
        }
    }

private:
    static constexpr int kBlockN = 1024;
    static constexpr uint32_t kTileFloatBytes = kBlockN * sizeof(float);

    __aicore__ inline int32_t BlockCount() const
    {
        return (tiling_.M + tiling_.blockM - 1) / tiling_.blockM;
    }

    __aicore__ inline int32_t NumTiles() const
    {
        return (tiling_.N + kBlockN - 1) / kBlockN;
    }

    __aicore__ inline int32_t GetValidN(int32_t colBase) const
    {
        return (colBase + kBlockN <= tiling_.N) ? kBlockN : (tiling_.N - colBase);
    }

    __aicore__ inline AscendC::RoundMode OutputRoundMode() const
    {
        if constexpr (std::is_same_v<dataType, bfloat16_t>) {
            return AscendC::RoundMode::CAST_ROUND;
        }
        return AscendC::RoundMode::CAST_NONE;
    }

    __aicore__ inline void PrepareInputTensor(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &src,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &castBuf,
        int32_t count)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            dst = src.template ReinterpretCast<float>();
        } else {
            dst = castBuf.Get<float>();
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PrepareOutputTensor(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &out,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &castBuf)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            dst = out.template ReinterpretCast<float>();
        } else {
            dst = castBuf.Get<float>();
        }
    }

    __aicore__ inline void FinalizeOutputTensor(
        AscendC::LocalTensor<dataType> &out,
        AscendC::LocalTensor<float> &src,
        int32_t count)
    {
        if constexpr (!std::is_same_v<dataType, float>) {
            AscendC::Cast(out, src, OutputRoundMode(), count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PrepareInvRmsTensor(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &out)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            dst = out.template ReinterpretCast<float>();
        } else {
            dst = invRmsBuf_.Get<float>();
        }
    }

    __aicore__ inline void CopyInX(int32_t rowIdx, int32_t colBase, int32_t validN)
    {
        xInQueue_.AllocTensor<dataType>(xInLocal_);
        LoadGmToUb(xInLocal_, xGM_[rowIdx * tiling_.N + colBase], static_cast<uint32_t>(validN));
        xInQueue_.EnQue(xInLocal_);
    }

    __aicore__ inline void CopyInGamma(int32_t colBase, int32_t validN)
    {
        gammaInQueue_.AllocTensor<dataType>(gammaInLocal_);
        LoadGmToUb(gammaInLocal_, gammaGM_[colBase], static_cast<uint32_t>(validN));
        gammaInQueue_.EnQue(gammaInLocal_);
    }

    __aicore__ inline void CopyOutY(int32_t rowIdx, int32_t colBase, int32_t validN)
    {
        yOutQueue_.DeQue<dataType>(yOutLocal_);
        StoreUbToGm(yGM_[rowIdx * tiling_.N + colBase], yOutLocal_, static_cast<uint32_t>(validN));
        yOutQueue_.FreeTensor(yOutLocal_);
    }

    __aicore__ inline void CopyOutInvRms(int32_t rowIdx)
    {
        invRmsOutQueue_.DeQue<dataType>(invRmsOutLocal_);
        StoreUbToGm(invRmsGM_[rowIdx], invRmsOutLocal_, 1);
        invRmsOutQueue_.FreeTensor(invRmsOutLocal_);
    }

    __aicore__ inline float ComputeInvRms(int32_t rowIdx)
    {
        reduceLocal_ = reduceBuf_.Get<float>();
        sumLocal_ = sumBuf_.Get<float>();
        tempLocal_ = tempBuf_.Get<float>();
        invRmsOutQueue_.AllocTensor<dataType>(invRmsOutLocal_);
        PrepareInvRmsTensor(invRmsLocal_, invRmsOutLocal_);
        float sumSq = 0.0f;

        for (int by = 0; by < NumTiles(); ++by) {
            const int colBase = by * kBlockN;
            const int validN = GetValidN(colBase);

            CopyInX(rowIdx, colBase, validN);

            xInQueue_.DeQue<dataType>(xInLocal_);
            PrepareInputTensor(xLocal_, xInLocal_, xCastBuf_, validN);
            AscendC::Mul(tempLocal_, xLocal_, xLocal_, validN);
            AscendC::ReduceSum<float>(sumLocal_, tempLocal_, reduceLocal_, validN);
            sumSq += sumLocal_.GetValue(0);
            xInQueue_.FreeTensor(xInLocal_);
        }

        AscendC::Duplicate(sumLocal_, sumSq * tiling_.invN + tiling_.eps, 1);
        AscendC::Rsqrt(invRmsLocal_, sumLocal_, 1);
        AscendC::PipeBarrier<PIPE_ALL>();
        float invRms = invRmsLocal_.GetValue(0);
        AscendC::PipeBarrier<PIPE_ALL>();
        FinalizeOutputTensor(invRmsOutLocal_, invRmsLocal_, 1);

        invRmsOutQueue_.EnQue(invRmsOutLocal_);
        return invRms;
    }

    __aicore__ inline void ComputeTile(float invRms, int32_t validN)
    {
        yOutQueue_.AllocTensor<dataType>(yOutLocal_);
        xInQueue_.DeQue<dataType>(xInLocal_);
        gammaInQueue_.DeQue<dataType>(gammaInLocal_);
        PrepareInputTensor(xLocal_, xInLocal_, xCastBuf_, validN);
        PrepareInputTensor(gammaLocal_, gammaInLocal_, gammaCastBuf_, validN);
        PrepareOutputTensor(yLocal_, yOutLocal_, yCastBuf_);
        AscendC::Muls(yLocal_, xLocal_, invRms, validN);
        AscendC::Mul(yLocal_, yLocal_, gammaLocal_, validN);
        FinalizeOutputTensor(yOutLocal_, yLocal_, validN);
        xInQueue_.FreeTensor(xInLocal_);
        gammaInQueue_.FreeTensor(gammaInLocal_);
        yOutQueue_.EnQue(yOutLocal_);
    }

    __aicore__ inline void ProcessRow(int rowIdx)
    {
        float invRms = ComputeInvRms(rowIdx);
        CopyOutInvRms(rowIdx);

        for (int by = 0; by < NumTiles(); ++by) {
            const int colBase = by * kBlockN;
            const int validN = GetValidN(colBase);

            CopyInX(rowIdx, colBase, validN);
            CopyInGamma(colBase, validN);
            ComputeTile(invRms, validN);
            CopyOutY(rowIdx, colBase, validN);
        }
    }

private:
    RmsNormKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int subBlockRows_{0};

    AscendC::GlobalTensor<dataType> xGM_;
    AscendC::GlobalTensor<dataType> gammaGM_;
    AscendC::GlobalTensor<dataType> yGM_;
    AscendC::GlobalTensor<dataType> invRmsGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> gammaInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> yOutQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> invRmsOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> invRmsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yCastBuf_;

    AscendC::LocalTensor<dataType> xInLocal_;
    AscendC::LocalTensor<dataType> gammaInLocal_;
    AscendC::LocalTensor<dataType> yOutLocal_;
    AscendC::LocalTensor<dataType> invRmsOutLocal_;
    AscendC::LocalTensor<float> invRmsLocal_;
    AscendC::LocalTensor<float> xLocal_;
    AscendC::LocalTensor<float> gammaLocal_;
    AscendC::LocalTensor<float> yLocal_;
    AscendC::LocalTensor<float> reduceLocal_;
    AscendC::LocalTensor<float> sumLocal_;
    AscendC::LocalTensor<float> tempLocal_;
};

// =========================================================================
// Entry point macros — generate 9 (strategy × dtype) kernel entry functions
// =========================================================================

#define DEF_RMS_NORM_ENTRY(StrategyName, dtype, dtype_suffix)                          \
    extern "C" __global__ __aicore__ void rms_norm_##StrategyName##_custom_##dtype_suffix( \
        GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR invRms, GM_ADDR tiling)           \
    {                                                                                   \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);                              \
        AscendC::TPipe pipe;                                                            \
        RmsNorm##StrategyName##Kernel<dtype> kernel;                                    \
        kernel.Init(x, gamma, y, invRms, tiling, &pipe);                               \
        kernel.Process();                                                               \
    }                                                                                   \
                                                                                        \
    extern "C" void rms_norm_##StrategyName##_do_##dtype_suffix(                        \
        uint32_t blockDim,                                                              \
        void *stream,                                                                   \
        uint8_t *x,                                                                     \
        uint8_t *gamma,                                                                 \
        uint8_t *y,                                                                     \
        uint8_t *invRms,                                                                \
        uint8_t *tiling)                                                                \
    {                                                                                   \
        rms_norm_##StrategyName##_custom_##dtype_suffix<<<blockDim, nullptr, stream>>>( \
            x, gamma, y, invRms, tiling);                                               \
    }

DEF_RMS_NORM_ENTRY(MergeN, float, fp32)
DEF_RMS_NORM_ENTRY(MergeN, half, fp16)
DEF_RMS_NORM_ENTRY(MergeN, bfloat16_t, bf16)
DEF_RMS_NORM_ENTRY(SingleRow, float, fp32)
DEF_RMS_NORM_ENTRY(SingleRow, half, fp16)
DEF_RMS_NORM_ENTRY(SingleRow, bfloat16_t, bf16)
DEF_RMS_NORM_ENTRY(SplitD, float, fp32)
DEF_RMS_NORM_ENTRY(SplitD, half, fp16)
DEF_RMS_NORM_ENTRY(SplitD, bfloat16_t, bf16)

#undef DEF_RMS_NORM_ENTRY
