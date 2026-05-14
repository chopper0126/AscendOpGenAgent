# Abs 设计文档

## 1. 算子接口

### 1.1 函数签名
```cpp
at::Tensor abs(
    const at::Tensor &x
);
```

### 1.2 参数说明
| 参数名 | 类型 | 输入/输出 | 支持的数据类型 | 描述 | 约束条件 |
|--------|------|-------|-------|--------|------|
| x | at::Tensor | 输入 | bfloat16/float16/float32 | 输入tensor | 支持ND |
| output | at::Tensor | 输出 | bfloat16/float16/float32 | 输出tensor，与输入同dtype同shape | 支持ND |

### 1.3 支持的数据类型
- [x] bfloat16
- [x] float16
- [x] float32

---

## 2. 计算逻辑

### 2.1 算法描述
对输入 tensor 的每个元素计算绝对值：`output[i] = |input[i]|`

### 2.2 AscendC API 伪代码

**float32 路径**（直接计算）:
```
CopyIn:  DataCopy(xLocal, xGlobal[offset], tileLength)
Compute: Abs(yLocal, xLocal, tileLength)
CopyOut: DataCopy(yGlobal[offset], yLocal, tileLength)
```

**float16/bfloat16 路径**（升精度计算）:
```
CopyIn:  DataCopy(xLocalHalf, xGlobal[offset], tileLength)
Compute: Cast(xLocalFp32, xLocalHalf, CAST_NONE, tileLength)    // 升精度
         Abs(yLocalFp32, xLocalFp32, tileLength)                // FP32 计算
         Cast(yLocalHalf, yLocalFp32, CAST_ROUND, tileLength)   // 降精度
CopyOut: DataCopy(yGlobal[offset], yLocalHalf, tileLength)
```

### 2.3 实现路径选择
- [x] AscendC Kernel（纯vector实现）
- [ ] CATLASS模板库（矩阵乘法类）
- [ ] ACLNN封装（CANN内置算子）

**选择理由**: Abs 是典型的逐元素操作，每个元素独立计算，使用 AscendC Vector API 即可高效实现。

---

## 3. Tiling策略

AscendC 算子采用**两级 Tiling 策略**来充分利用硬件并行能力。

### 两级 Tiling 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    全局内存 (GM)                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              totalLength 元素数据                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │  Core 0  │     │  Core 1  │ ... │ Core 39  │   ← Block级Tiling (核间切分)
    │ formerLen│     │ formerLen│     │ tailLen  │
    └──────────┘     └──────────┘     └──────────┘
          │                │                │
          ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │   UB 0   │     │   UB 1   │     │  UB 39   │   ← UB级Tiling (核内切分)
    │ tileLen  │     │ tileLen  │     │ tileLen  │
    │ tileLen  │     │ tileLen  │     │ tileLen  │
    │   ...    │     │   ...    │     │   ...    │
    └──────────┘     └──────────┘     └──────────┘
```

### 核心原则

- **Block级Tiling (核间切分)**: 确保每个 Core 处理的计算量相对均衡，使用整核/尾核策略
- **UB级Tiling (核内切分)**: 根据 UB 分配表确定 buffer 需求，计算单次循环处理量

**算子类型**: elementwise

### 3.1 Tiling参数结构体定义

```cpp
struct AbsTilingData {
    int64_t totalLength;        // 总数据长度
    int64_t usedCoreNum;        // 实际使用的核数

    int64_t formerNum;          // 整核数量
    int64_t formerLength;       // 整核数据长度
    int64_t tailNum;            // 尾核数量
    int64_t tailLength;         // 尾核数据长度

    int64_t tileLength;         // UB单次处理长度
};
```

### 3.2 Block级Tiling（核间切分）

**策略要点**:
1. **Cache Line对齐**: 每个核处理的数据块 512 字节对齐
2. **负载均衡**: 整核/尾核策略

| 参数 | 计算公式 | 说明 |
|------|----------|------|
| totalLengthCore | (totalLength + CORE_NUM - 1) / CORE_NUM | 每核平均数据量 |
| totalLengthCoreAlign | (totalLengthCore + 512 - 1) / 512 * 512 | Cache Line 对齐 |
| usedCoreNum | (totalLength + totalLengthCoreAlign - 1) / totalLengthCoreAlign | 实际使用核数 |
| formerNum | usedCoreNum - 1 | 整核数量 |
| tailNum | 1 | 尾核数量 |
| formerLength | totalLengthCoreAlign | 整核数据长度 |
| tailLength | totalLength - (usedCoreNum - 1) * formerLength | 尾核数据长度 |

### 3.3 UB级Tiling（核内切分）

#### 精度处理说明

**重要**: NPU 计算单元不支持 float16/bfloat16 数据类型的直接计算，必须升精度到 float32 后再进行计算。

| 输入数据类型 | 处理方式 | 计算精度 | UB 影响 |
|------------|---------|---------|--------|
| float16 | 升精度到 float32 | float32 | 需要额外 float32 buffer |
| bfloat16 | 升精度到 float32 | float32 | 需要额外 float32 buffer |
| float32 | 直接计算 | float32 | 无额外开销 |

#### UB 分配表

**float32 输入**:

| Buffer名称 | 大小(字节) | 用途 | 数量 | 总大小 |
|-----------|-----------|------|------|--------|
| inQueueX | tileLength * 4 | 输入数据缓冲 (fp32) | BUFFER_NUM(2) | tileLength * 8 |
| outQueueY | tileLength * 4 | 输出数据缓冲 (fp32) | BUFFER_NUM(2) | tileLength * 8 |
| tempBuffer | tileLength * 4 | 计算缓冲 | 1 | tileLength * 4 |
| **总计** | - | - | - | **tileLength * 20** |

**float16/bfloat16 输入（需要升精度到 float32）**:

| Buffer名称 | 大小(字节) | 用途 | 数量 | 总大小 |
|-----------|-----------|------|------|--------|
| inQueueX | tileLength * 2 | 输入数据缓冲 (fp16/bf16) | BUFFER_NUM(2) | tileLength * 4 |
| outQueueY | tileLength * 2 | 输出数据缓冲 (fp16/bf16) | BUFFER_NUM(2) | tileLength * 4 |
| tempBuffer | tileLength * 4 | float32 计算缓冲 | 2 | tileLength * 8 |
| **总计** | - | - | - | **tileLength * 16** |

#### bufferCoefficient

| 数据类型 | bufferCoefficient |
|----------|-------------------|
| float32 | **20** |
| float16/bf16 | **16** |

#### tileLength 计算

| 参数 | 计算公式 | 说明 |
|------|----------|------|
| bufferCoefficient | float32: 20, fp16/bf16: 16 | 根据UB分配表确定 |
| maxTileElements | UB_SIZE_LIMIT / bufferCoefficient | UB_SIZE_LIMIT 通过平台API获取 |
| alignElements | 32 / dtypeSize | 32字节对齐 |
| tileLength | (maxTileElements / alignElements) * alignElements | 对齐后的tile长度 |

---

## 4. Workspace需求

### 4.1 Workspace 大小计算
elementwise 类算子使用系统 workspace。

| 算子类别 | workspace size | 说明 |
|----------|---------------|------|
| elementwise 类 | SYSTEM_WORKSPACE_SIZE | 16MB |

### 4.2 Workspace 分配示例
```cpp
constexpr int64_t SYSTEM_WORKSPACE_SIZE = 16 * 1024 * 1024;  // 16MB
size_t workspaceSize = SYSTEM_WORKSPACE_SIZE;
auto workspace = at::empty({static_cast<int64_t>(workspaceSize)},
                           at::TensorOptions().dtype(at::kByte).device(x.device()));
```

---

## 5. 性能优化

### 5.1 关键优化点
1. Double Buffer 隐藏内存延迟，实现数据搬运与计算流水线
2. Cache Line 对齐 (512B) 优化核间负载均衡
3. 向量化 DataCopy 替代逐元素访问
4. FP16/BF16 升精度到 FP32 后使用 Vector Abs 计算

### 5.2 算子特性
- **计算模式**: memory-bound
- **访存模式**: 顺序访问
- **并行性**: 高

---

## 6. Kernel端实现要点

### 6.1 执行流程（核内循环）

```cpp
__aicore__ inline void Process() {
    int64_t coreLength = AscendC::GetBlockIdx() == tiling->usedCoreNum - 1
                         ? this->tailLength : this->formerLength;
    int64_t tileNum = (coreLength + this->tileLength - 1) / this->tileLength;
    int64_t tailTileLength = coreLength - (tileNum - 1) * this->tileLength;

    for (int64_t i = 0; i < tileNum - 1; ++i) {
        CopyIn(i, this->tileLength);
        Compute(i, this->tileLength);
        CopyOut(i, this->tileLength);
    }
    // 处理尾块
    CopyIn(tileNum - 1, tailTileLength);
    Compute(tileNum - 1, tailTileLength);
    CopyOut(tileNum - 1, tailTileLength);
}
```

### 6.2 分 dtype Kernel 入口

需要分别实现 float32 和 float16/bfloat16 两条路径：
- **AbsKernelFloat32**: 直接 Abs 计算
- **AbsKernelFloat16**: Cast 升精度 → Abs → Cast 降精度

---

## 7. 实现检查清单

### 7.1 文件结构
- [ ] `kernel/CMakeLists.txt`
- [ ] `kernel/setup.py`
- [ ] `kernel/ops.h` (添加声明)
- [ ] `kernel/register.cpp` (添加注册)
- [ ] `kernel/op_host/abs.cpp`
- [ ] `kernel/op_kernel/abs.cpp`

### 7.2 Host端实现
- [ ] 定义 AbsTilingData 结构体
- [ ] 实现 Block 级 Tiling 参数计算（Cache Line 对齐）
- [ ] 根据 dtype 确定 bufferCoefficient
- [ ] 实现 UB 级 tileLength 计算（32 字节对齐）
- [ ] 分配 workspace (SYSTEM_WORKSPACE_SIZE)
- [ ] 按 dtype 分发调用不同 kernel 入口

### 7.3 Kernel端实现
- [ ] 实现 Init 函数（整核/尾核偏移计算）
- [ ] 实现 CopyIn 函数（GM → UB, DataCopy）
- [ ] 实现 Compute 函数（fp16/bf16 时 Cast→Abs→Cast, fp32 时直接 Abs）
- [ ] 实现 CopyOut 函数（UB → GM, DataCopy）
- [ ] 实现 Process 主循环（Double Buffer + 尾块处理）

---

## 8. 参考实现

- **PyTorch参考**: `torch.abs`
- **AscendC API**: `Abs(dst, src, len)`
- **相似算子**: `csrc/ops/relu/` — 单输入 elementwise 算子参考
