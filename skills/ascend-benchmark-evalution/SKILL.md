---
name: ascend-benchmark-evalution
description: >
  Ascend Benchmark Evaluation Skill — 专门用于测评 lingxi-code agent 生成算子的能力，包括精度和性能。
  支持遍历数据集、调用 lingxi-code agent 生成算子、测评精度和性能，并生成详细报告。
argument-hint: >
  必需：benchmark_path、agent_workspace、level_problems。。
  可选：output_dir、arch、npu_id、resume、timeout_per_task、warmup、repeats。
  level_problems 格式：{1: [1,2], 2: [1,3], 3: None}，None 表示该 level 全选。
  benchmark_path 支持：1) 绝对路径；2) 相对路径（基于当前工作目录）；3) 数据集名称（如 "NPUKernelBench"）。
---

# Ascend Benchmark Evaluation Skill

<role>
你是一个自动化评测框架执行器，专门用于测评 lingxi-code agent 生成 AscendC 算子的能力。你的任务是遍历指定的算子数据集，调用 lingxi-code agent 生成算子代码，验证其正确性，测试性能，并生成详细的评测报告。
</role>

## 输入参数

### 必需参数

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `benchmark_path` | str | 算子数据集路径 | `"/home/lvyou/y00889327/AscendOpGenAgent/benchmarks/NPUKernelBench"` |
| `agent_workspace` | str | Agent 工作区路径（包含 agents/ 和 skills/） | `"/path/to/.opencode"` |
| `level_problems` | dict | 每个 level 的 problem 选择 | `{1: [1,2], 2: None}` |

### level_problems 格式说明

```python
# 选择 Level 1 的 problem 1,2
# 选择 Level 2 的 problem 1,3
# Level 3 全选
# Level 4 不选（不传入）
{
    1: [1, 2],        # 只选指定 problems
    2: [1, 3],        
    3: None,          # None 表示该 level 全部
    # 4: ...          # 不传入表示不评测该 level
}
```

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_dir` | str | `"benchmark_results"` | 输出目录名 |
| `arch` | str | 首次运行时询问 | 目标硬件架构（ascend910b2 等） |
| `npu_id` | int | 首次运行时询问 | 目标 NPU 设备序号（如 0, 1, 2...） |
| `resume` | bool | `true` | 是否断点续跑 |
| `timeout_per_task` | int | 2400 | 单任务超时（秒）|
| `warmup` | int | 5 | 性能测试 warmup 次数 |
| `repeats` | int | 50 | 性能测试重复次数 |

## 工作流程

```
Phase 1: 初始化
  ├── 解析用户输入（自然语言 → 结构化参数）
  ├── 加载配置（默认值 ← 用户输入）
  ├── 检查/询问目标硬件架构（arch）
  │   └── 保存 arch 到状态文件供后续任务使用
  ├── 检查/询问目标 NPU 序号（npu_id）
  │   └── 保存 npu_id 到状态文件供后续任务使用
  │   └── 设置环境变量 ASCEND_RT_VISIBLE_DEVICES={npu_id}
  ├── 创建输出目录结构
  └── 恢复断点状态（resume=true 时）

Phase 2: 任务扫描
  ├── 遍历 benchmark_path 中的算子数据集
  ├── 解析每个算子的任务文件，提取元数据
  ├── 过滤已完成的任务
  └── 构建任务队列

Phase 3: 串行评测
  └── 串行执行每个任务：
      ├── 调用 lingxi-code agent 生成算子代码
      ├── 正确性验证（使用 ascendc_evalution skill）
      ├── 性能测试（调用 benchmark.py）
      ├── 保存结果
      └── 增量更新评测报告

Phase 4: 报告完成
  └── 所有任务完成后，生成最终汇总报告
```

## 输出目录结构

```
{output_dir}/
└── lingxi-code_{timestamp}_{run_id}/
    ├── tmp/                            # 中间文件目录
    │   ├── temp_code/                  # 临时代码文件
    │   ├── temp_logs/                  # 临时日志文件
    │   └── cache/                      # 缓存文件
    ├── level_{n}/                      # 按 level 组织的结果
    │   ├── {problem_id}_{op_name}/
    │   │   ├── generated_code/         # 生成的代码
    │   │   ├── verify_result.json      # 验证结果
    │   │   └── perf_result.json        # 性能结果
    │   └── ...
    ├── .benchmark_state.json           # 运行状态（含 arch, npu_id 等）
    └── agent_report.md                 # ⚠️ 始终生成评测报告（增量更新）
```

## 报告内容

### agent_report.md 结构

**增量更新机制：**
- 每个任务完成后立即更新报告，追加新结果
- 报告头部始终显示最新汇总统计
- 支持断点续跑时恢复和继续更新

**报告包含内容：**

1. **执行摘要** - 时间、硬件、评测范围
2. **总体统计** - 表格形式展示各 Level 和总体的任务数、编译成功率、正确率、优于PyTorch比例、平均加速比
3. **按算子类型统计** - 分类统计
4. **编译失败列表** - 按 Level 组织的编译失败详情
5. **数值验证失败列表** - 按 Level 组织的数值错误详情（含 Max Diff）
6. **性能劣化列表** - 按 Level 组织的性能低于 PyTorch 的算子（含劣化倍数）
7. **详细结果表** - 每个 problem 的完整结果

## 使用示例

### 示例 1: 基础使用

```
测评 NPUKernelBench 数据集
参数：
- benchmark_path: /home/lvyou/y00889327/AscendOpGenAgent/benchmarks/NPUKernelBench
```

**执行流程：**
1. 首次运行会询问目标硬件架构和 NPU 设备序号
2. 结果保存到 `./benchmark_results/lingxi-code_20260323_162216_run001/`
3. 报告增量更新

### 示例 2: 自定义输出目录

```
测评 NPUKernelBench 数据集，自定义输出目录
参数：
- benchmark_path: NPUKernelBench
- output_dir: my_benchmark_results
- arch: ascend910b2
- npu_id: 0
```

**输出路径：** `./my_benchmark_results/lingxi-code_20260323_162216_run001/`

## 依赖

- Python 3.8+
- opencode Agent 调用机制
- lingxi-code agent（用于生成 AscendC 算子）
- ascendc_evalution skill（用于验证和性能测试）
- NPU 设备（用于验证和性能测试）

## 注意事项

1. **串行执行**: 任务按顺序逐个执行，不进行并行化
2. **始终生成报告**: 无论评测成功与否，都会生成 `agent_report.md`
3. **增量报告**: 每个任务完成后立即更新报告，支持实时查看进度
4. **断点续跑**: 基于 `(level, problem_id)` 去重，支持中断后恢复
5. **超时处理**: 单任务超时不影响整体流程，记录为失败
6. **错误隔离**: 单任务失败会记录但继续执行其他任务
7. **架构选择**: 首次运行需选择目标硬件架构，选择后保存到状态文件
8. **NPU 选择**: 首次运行需选择目标 NPU 设备序号，选择后保存到状态文件
