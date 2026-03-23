#include "kernel_operator.h"

class KernelGelu {
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x3Buf, innerBuf, scaleBuf, tanhBuf, onePlusTanhBuf, halfXBuf;
    AscendC::GlobalTensor<float> inputGm;
    AscendC::GlobalTensor<float> outputGm;
    uint32_t elementsPerCore;
    uint32_t tileSize;
    uint32_t innerLoops;
    float sqrtTwoOverPi;
    float coeff;

public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input_ptr, GM_ADDR output_ptr, uint32_t elementsPerCore, uint32_t tileSize, uint32_t innerLoops, float sqrtTwoOverPi, float coeff)
    {
        this->elementsPerCore = elementsPerCore;
        this->tileSize = tileSize;
        this->innerLoops = innerLoops;
        this->sqrtTwoOverPi = sqrtTwoOverPi;
        this->coeff = coeff;

        // Set global memory buffer. Offset is calculated based on block index
        uint32_t start = elementsPerCore * AscendC::GetBlockIdx();
        uint32_t totalElements = elementsPerCore;
        
        inputGm.SetGlobalBuffer((__gm__ float *)input_ptr + start, totalElements);
        outputGm.SetGlobalBuffer((__gm__ float *)output_ptr + start, totalElements);

        // Initialize pipe buffer queues with one slot, each holding tileSize floats
        pipe.InitBuffer(inQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(outQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(x3Buf, this->tileSize * sizeof(float));
        pipe.InitBuffer(innerBuf, this->tileSize * sizeof(float));
        pipe.InitBuffer(scaleBuf, this->tileSize * sizeof(float));
        pipe.InitBuffer(tanhBuf, this->tileSize * sizeof(float));
        pipe.InitBuffer(onePlusTanhBuf, this->tileSize * sizeof(float));
        pipe.InitBuffer(halfXBuf, this->tileSize * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->innerLoops; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t idx)
    {
        AscendC::LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
        AscendC::DataCopyPad(inputLocal, inputGm[idx * this->tileSize],
            {1, static_cast<uint16_t>(this->tileSize * sizeof(float)), 0, 0},
            {false, 0, 0, 0});
        inQueue.EnQue(inputLocal);
    }
    
    __aicore__ inline void Compute(uint32_t idx)
    {
        AscendC::LocalTensor<float> inputLocal = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> outputLocal = outQueue.AllocTensor<float>();
        
        AscendC::LocalTensor<float> x3Local = x3Buf.Get<float>();
        AscendC::LocalTensor<float> innerLocal = innerBuf.Get<float>();
        AscendC::LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
        AscendC::LocalTensor<float> tanhLocal = tanhBuf.Get<float>();
        AscendC::LocalTensor<float> onePlusTanhLocal = onePlusTanhBuf.Get<float>();
        AscendC::LocalTensor<float> halfXLocal = halfXBuf.Get<float>();
        
        // x3 = x * x * x
        AscendC::Mul(x3Local, inputLocal, inputLocal, this->tileSize);
        AscendC::Mul(x3Local, x3Local, inputLocal, this->tileSize);
        
        // inner = coeff * x3 + x
        AscendC::Muls(innerLocal, x3Local, this->coeff, this->tileSize);
        AscendC::Add(innerLocal, innerLocal, inputLocal, this->tileSize);
        
        // scale = sqrtTwoOverPi * inner
        AscendC::Muls(scaleLocal, innerLocal, this->sqrtTwoOverPi, this->tileSize);
        
        // tanh_val = tanh(scale)
        AscendC::Tanh(tanhLocal, scaleLocal, this->tileSize);
        
        // one_plus_tanh = 1 + tanh_val
        AscendC::Adds(onePlusTanhLocal, tanhLocal, 1.0f, this->tileSize);
        
        // half_x = 0.5 * x
        AscendC::Muls(halfXLocal, inputLocal, 0.5f, this->tileSize);
        
        // output = half_x * one_plus_tanh
        AscendC::Mul(outputLocal, halfXLocal, onePlusTanhLocal, this->tileSize);
        
        outQueue.EnQue<float>(outputLocal);
        inQueue.FreeTensor(inputLocal);
    }
    
    __aicore__ inline void CopyOut(uint32_t idx)
    {
        AscendC::LocalTensor<float> outputLocal = outQueue.DeQue<float>();
        AscendC::DataCopyPad(outputGm[idx * this->tileSize], outputLocal,
            {1, static_cast<uint16_t>(this->tileSize * sizeof(float)), 0, 0});
        outQueue.FreeTensor(outputLocal);
    }
};

extern "C" __global__ __aicore__ void gelu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGelu op;
    op.Init(x, y, tiling_data.elementsPerCore, tiling_data.tileSize, tiling_data.innerLoops, tiling_data.sqrtTwoOverPi, tiling_data.coeff);
    op.Process();
}