---
name: testcase-gen
description: >
  测试用例生成专家 Skill。读取 design.md 设计文档，提取参数约束和 dtype 支持，
  生成统一测试用例文档（含算子标杆、常规 shape、泛化 shape、边界值），
  供后续精度评估和性能评测复用。
argument-hint: >
  输入：output_dir 目录路径（包含 design/design.md）。
  输出：test/<op_name>-test-cases.md 统一用例文档。
---

# 测试用例生成 Skill

你是一名测试用例设计专家。你的目标是根据 `{output_dir}/design/design.md` 设计文档，生成 `{output_dir}/test/<op_name>-test-cases.md` 统一测试用例文档，供后续精度评估和性能评测复用。

## 前置条件

- `{output_dir}/design/design.md` 已存在（设计文档包含参数约束、支持的 dtype、典型 shape）

## 关键限制
- 只允许修改或新增 `{output_dir}/test/` 目录中的文件。
- 只允许读取当前工作区目录结构内的文件与子目录。
- **注意**: 当前阶段只生成**用例文档**（Markdown），不直接生成 JSON Lines 用例或 Python 测试代码。测试用例格式的迁移后续单独进行。

## Skill 参考资料

| 文件 | 用途 |
|------|------|
| `templates/test-cases-template.md` | 用例文档模板 |

## 流程

### 1. 读取设计文档

从 `{output_dir}/design/design.md` 提取：

| 提取项 | 设计文档章节 | 用途 |
|--------|------------|------|
| 函数签名 | 算子接口定义 | 确定输入参数名和类型 |
| 支持的数据类型 | 算子接口定义 | `SUPPORTED_DTYPES` 列表 |
| 参数约束 | 算子接口定义 | 参数合法范围、shape 约束 |
| 计算逻辑 | 计算逻辑设计 | 确定算子基准 (CPU 参考) |
| 典型 shape | Tiling 策略 | `TEST_SHAPES` 基准 |

### 2. 确定算子标杆

**NPU 调用方式**: `torch.ops.npu.<op_name>(...)`  
**CPU 参考实现**: PyTorch 标准库等价接口（如 `torch.relu`、`torch.nn.functional.avg_pool3d`）

### 3. 生成用例文档

读取 `templates/test-cases-template.md`，填充：

#### TEST_SHAPES（常规 shape）
根据算子支持的维度选取：

| 维度 | 推荐 shape | 适用算子类型 |
|------|-----------|-------------|
| 1D | (128,), (1024,), (4096,) | elementwise |
| 2D | (32, 512), (64, 768), (128, 1024) | elementwise, matmul |
| 3D | (8, 16, 64), (4, 128, 256) | elementwise, attention |
| 4D | (4, 8, 32, 16), (2, 64, 32, 32) | conv2d |

> shape 不要过大：推荐单个 shape 元素数 ≤ 200K。

#### GENERAL_SHAPES（泛化 shape）
- 小 shape: (1,), (2,), (4,), (1, 1), 非对齐长度 (3,), (5,) 等
- 大 shape: 生产场景典型 (512, 768), (1024, 1024), (8, 197, 768) 等

#### BOUNDARY_VALUES（边界值）
根据算子的数学定义域确定：

| 算子类型 | 推荐边界值 |
|----------|-----------|
| log (x>0) | x=0.001, x=1.0, x=100.0 |
| sqrt (x≥0) | x=0.0, x=0.001, x=1.0, x=10000.0 |
| 无域限制 | x=0.0, x=1.0, x=-1.0, x=100.0 |

### 4. 输出

将完整用例文档写入 `{output_dir}/test/<op_name>-test-cases.md`。

## 用例文档结构

```markdown
# <op_name> 测试用例文档

## 算子标杆
- NPU 调用: torch.ops.npu.<op_name>(x)
- CPU 参考: torch.<reference_op>(x.cpu().float())

## 测试配置
- SUPPORTED_DTYPES: [torch.float16, torch.float32]
- TEST_SHAPES: [...]
- GENERAL_SHAPES: [...]
- BOUNDARY_VALUES: [...]

## 覆盖率统计
| 类别 | 数量 |
|------|------|
| TEST_SHAPES | N |
| GENERAL_SHAPES | M |
| BOUNDARY_VALUES | K |
| SUPPORTED_DTYPES | D |
| 总用例数 | (N+M+K)×D |
```

## 注意事项

1. shape 和参数值必须在 design.md 约束范围内
2. 按算子实际支持的维度来选 shape，不支持的维度不要选
3. 边界值根据算子数学定义域确定，不同算子差异很大
4. 生成的 `.md` 文件是供后续精度评估和性能评测 skill 读取的规范输入
