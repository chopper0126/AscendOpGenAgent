// AscendC Abs kernel — supports float32, float16, bfloat16

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelAbs {
public:
    __aicore__ inline KernelAbs() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                int64_t formerNum, int64_t formerLength,
                                int64_t tailLength, int64_t tileLength)
    {
        int64_t blockIdx = AscendC::GetBlockIdx();

        if (blockIdx < formerNum) {
            this->blockLength = formerLength;
            int64_t offset = formerLength * blockIdx;
            xGm.SetGlobalBuffer((__gm__ T *)x + offset, formerLength);
            yGm.SetGlobalBuffer((__gm__ T *)y + offset, formerLength);
        } else {
            this->blockLength = tailLength;
            int64_t tailIdx = blockIdx - formerNum;
            int64_t offset = formerLength * formerNum + tailLength * tailIdx;
            xGm.SetGlobalBuffer((__gm__ T *)x + offset, tailLength);
            yGm.SetGlobalBuffer((__gm__ T *)y + offset, tailLength);
        }

        this->tileLength = tileLength;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(T));
        if constexpr (sizeof(T) == 2) {
            pipe.InitBuffer(tmpBuf0, tileLength * sizeof(float));
            pipe.InitBuffer(tmpBuf1, tileLength * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        int64_t tileNum = (this->blockLength + this->tileLength - 1) / this->tileLength;
        int64_t tailTileLength = this->blockLength - (tileNum - 1) * this->tileLength;

        int64_t alignNum = 32 / static_cast<int64_t>(sizeof(T));
        int64_t alignedTailLen = ((tailTileLength + alignNum - 1) / alignNum) * alignNum;

        for (int64_t i = 0; i < tileNum - 1; ++i) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        if (tileNum > 0) {
            CopyIn(tileNum - 1, alignedTailLen);
            Compute(tileNum - 1, alignedTailLen);
            CopyOut(tileNum - 1, alignedTailLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t curTileLength)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(xLocal, xGm[progress * this->tileLength], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t progress, int64_t curTileLength)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        if constexpr (sizeof(T) == sizeof(float)) {
            AscendC::Abs(yLocal, xLocal, curTileLength);
        } else {
            AscendC::LocalTensor<float> tmp0 = tmpBuf0.Get<float>();
            AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
            AscendC::Cast(tmp0, xLocal, AscendC::RoundMode::CAST_NONE, curTileLength);
            AscendC::Abs(tmp1, tmp0, curTileLength);
            AscendC::Cast(yLocal, tmp1, AscendC::RoundMode::CAST_RINT, curTileLength);
        }

        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t curTileLength)
    {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[progress * this->tileLength], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    int64_t blockLength;
    int64_t tileLength;
};

extern "C" __global__ __aicore__ void abs_custom(GM_ADDR x, GM_ADDR y,
                                                 int64_t formerNum, int64_t formerLength,
                                                 int64_t tailLength, int64_t tileLength,
                                                 int64_t dtypeSize)
{
    if (dtypeSize == 2) {
        KernelAbs<half> op;
        op.Init(x, y, formerNum, formerLength, tailLength, tileLength);
        op.Process();
    } else {
        KernelAbs<float> op;
        op.Init(x, y, formerNum, formerLength, tailLength, tileLength);
        op.Process();
    }
}
