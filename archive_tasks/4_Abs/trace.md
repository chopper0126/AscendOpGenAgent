# Trace: Abs

- 时间: 2026-05-14 11:05
- 算子: Abs (torch.abs)
- 设计路径: design.md (简单算子)
- 最终结果: SKIP (tilelang) | PASS (ascendc)

## 阶段零: Case 精简

- 结果: 通过
- 原始 case 数: 50
- 精简后 case 数: 10
- 备注: 覆盖 float32/float16/bfloat16、1D-4D、极端小 [100] 到极端大 [4096, 24576]

## 阶段〇.五: 设计文档 (仅简单算子路径)

- 结果: 通过
- 说明: 简单算子路径，生成 design/design.md（单输入单输出 elementwise，bufferCoefficient: fp32=20, fp16/bf16=16）
- 备注: 一次生成通过

## 阶段一: TileLang (仅复杂算子路径)

- 结果: 跳过
- 说明: Abs 为简单逐元素算子，走 design.md 路径，不经过 TileLang

## 阶段二: AscendC

- 结果: 通过
- 设计路径: design.md (简单算子)
- 产物: kernel/op_host/abs.cpp, kernel/op_kernel/abs.cpp, kernel/ops.h, kernel/register.cpp, kernel/CMakeLists.txt, kernel/setup.py, model_new_ascendc.py
- 编译: evaluate_ascendc.sh (cmake + make + whl)
- evaluate_ascendc.sh 执行次数: 2
- 关键错误信息:
  - 第 1 轮编译错误: `DataCopyPad` GM→UB 缺少第 4 个参数 `DataCopyPadExtParams`:
    ```
    error: no matching function for call to 'DataCopyPad'
    candidate function template not viable: requires 4 arguments, but 3 were provided
    ```
- Agent 行为记录:
  - 第 1 轮: 按 elementwise 模板生成全部 kernel 文件 + model_new_ascendc.py，退化检测通过，编译失败 — CopyIn 中 DataCopyPad 缺少 padParams 参数
  - 第 2 轮: 修改 CopyIn 函数，添加 `DataCopyPadExtParams<T> padParams{true, 0, 0, static_cast<T>(0)}` 作为第 4 参数 → 编译+验证通过
- 走偏点: 初次实现时直接使用 DataCopyPad(dst, src, copyParams) 三参数形式，但 GM→UB 方向必须传四参数（含 padParams），参考 rms_norm kernel 修复

## 阶段三: 性能分析

- 结果: 完成
- performance-analyzer 执行详情:
  - 测试配置: device=npu, warmup=5, repeats=50, seed=0
  - 测试的实现: reference / ascendc
  - 总体统计:
    - reference: mean=0.075ms (10 cases)
    - ascendc: mean=0.078ms (10 cases)
  - 性能结论:
    - AscendC 慢于 reference，overall speedup 0.59x
    - 大 shape case (case 2: [4096,18432]) speedup 0.98x，接近持平
    - 小 shape case 受 kernel launch 开销影响，speedup 仅 0.36x-0.64x
    - 对逐元素 Abs 操作，CANN 内置 aclnnAbs 已高度优化，自定义 kernel 难以超越

## 阶段四: 全量用例验证

- 结果: 通过
- 说明: 恢复 4_Abs.json.bak → 4_Abs.json (50 cases)，一次通过

## 汇总表报告

| Level | Problem ID | 算子名称 | 算子类型 | 编译通过 | 精度正确 | PyTorch 参考延迟 | 生成AscendC代码延迟 | 加速比 | 最终状态 | 精度正确 | 性能0.6x pytorch | 性能0.8x pytorch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | Abs | elementwise | ✅ | ✅ | 0.075 | 0.078 | 0.59 | 成功 | 是 | 否 | 否 |

## 评测输出摘要

```
PASS: AscendC 验证通过
```
