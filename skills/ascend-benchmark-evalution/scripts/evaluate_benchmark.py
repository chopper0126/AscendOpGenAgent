#!/usr/bin/env python3
"""
Ascend Benchmark Evaluation Script
专门用于测评 lingxi-code agent 生成算子的能力，包括精度和性能
"""

import os
import sys
import json
import time
import re
import glob
import logging
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import subprocess
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """任务配置"""
    benchmark_path: str
    output_dir: str = "./benchmark_results"
    arch: str = "ascend910b1"
    npu_id: int = 0
    resume: bool = True
    timeout_per_task: int = 2400
    warmup: int = 5
    repeats: int = 50


@dataclass
class GenerationResult:
    """代码生成结果"""
    success: bool = False
    code: str = ""
    generation_time: float = 0.0
    error_message: str = ""


@dataclass
class VerifyResult:
    """验证结果"""
    success: bool = False
    compiled: bool = False
    correctness: bool = False
    max_diff: Optional[float] = None
    error_message: str = ""


@dataclass
class PerformanceResult:
    """性能测试结果"""
    success: bool = False
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    peak_memory_mb: float = 0.0
    speedup_vs_torch: float = 0.0
    error_message: str = ""


@dataclass
class EvaluationResult:
    """完整评测结果"""
    level: int
    problem_id: int
    op_name: str
    op_type: str
    timestamp: str = ""
    generation: Optional[GenerationResult] = None
    verification: Optional[VerifyResult] = None
    performance: Optional[PerformanceResult] = None
    output_dir: str = ""


class TaskScanner:
    """任务扫描器"""
    
    @staticmethod
    def scan_tasks(benchmark_path: str) -> List[Dict[str, Any]]:
        """扫描任务
        
        Args:
            benchmark_path: Benchmark 根目录
        
        Returns:
            任务列表
        """
        tasks = []
        
        # 遍历 benchmark_path 下的所有 level 目录
        level_dirs = glob.glob(os.path.join(benchmark_path, "level*"))
        
        for level_dir in level_dirs:
            # 提取 level 编号
            match = re.search(r"level(\d+)", os.path.basename(level_dir))
            if not match:
                continue
            
            level = int(match.group(1))
            
            # 扫描任务文件
            task_files = glob.glob(os.path.join(level_dir, "*.py"))
            
            for task_file in task_files:
                # 提取 problem_id
                match = re.match(r"(\d+)_.*\.py", os.path.basename(task_file))
                if not match:
                    continue
                
                problem_id = int(match.group(1))
                
                # 提取算子名
                op_name = match.group(0).replace(".py", "").split("_", 1)[1]
                
                tasks.append({
                    "level": level,
                    "problem_id": problem_id,
                    "task_file": task_file,
                    "op_name": op_name
                })
        
        # 排序
        tasks.sort(key=lambda x: (x["level"], x["problem_id"]))
        
        return tasks
    
    @staticmethod
    def classify_op_type(op_name: str) -> str:
        """分类算子类型"""
        op_name_lower = op_name.lower()
        
        if any(kw in op_name_lower for kw in ["matmul", "bmm", "linear", "gemm"]):
            return "matmul"
        elif any(kw in op_name_lower for kw in ["conv"]):
            return "conv"
        elif any(kw in op_name_lower for kw in ["softmax", "layernorm", "batchnorm", "sum", "mean", "max", "min"]):
            return "reduce"
        elif any(kw in op_name_lower for kw in ["attention", "mha", "sdpa"]):
            return "attention"
        elif any(kw in op_name_lower for kw in ["add", "mul", "sub", "div", "relu", "sigmoid", "tanh", "gelu", "silu"]):
            return "elementwise"
        else:
            return "other"


class StateManager:
    """状态管理器（断点续跑）"""
    
    def __init__(self, output_dir: str):
        self.state_file = os.path.join(output_dir, ".benchmark_state.json")
        self.completed_tasks = set()
        self.arch = "ascend910b1"
        self.npu_id = 0
        self.load_state()
    
    def load_state(self):
        """加载状态"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.completed_tasks = set(tuple(x) for x in data.get("completed", []))
                    self.arch = data.get("arch", "ascend910b1")
                    self.npu_id = data.get("npu_id", 0)
                logger.info(f"已加载 {len(self.completed_tasks)} 个已完成任务")
            except Exception as e:
                logger.warning(f"加载状态失败: {e}")
    
    def save_state(self):
        """保存状态"""
        try:
            data = {
                "completed": [list(x) for x in self.completed_tasks],
                "arch": self.arch,
                "npu_id": self.npu_id
            }
            with open(self.state_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"保存状态失败: {e}")
    
    def is_completed(self, level: int, problem_id: int) -> bool:
        """检查任务是否已完成"""
        return (level, problem_id) in self.completed_tasks
    
    def mark_completed(self, level: int, problem_id: int):
        """标记任务完成"""
        self.completed_tasks.add((level, problem_id))
        self.save_state()
    
    def update_arch(self, arch: str):
        """更新架构"""
        self.arch = arch
        self.save_state()
    
    def update_npu_id(self, npu_id: int):
        """更新 NPU ID"""
        self.npu_id = npu_id
        self.save_state()


class LingxiCodeCaller:
    """Lingxi Code Agent 调用器"""
    
    def __init__(self, workspace: str):
        self.workspace = workspace
    
    def generate_operator(self, op_name: str, task_file: str, output_dir: str, timeout: int = 2400) -> GenerationResult:
        """调用 lingxi-code agent 生成算子代码"""
        result = GenerationResult()
        start_time = time.time()
        
        try:
            # 读取任务文件内容，提取算子描述
            with open(task_file, "r", encoding="utf-8") as f:
                task_desc = f.read()
            
            # 构建调用 lingxi-code agent 的 prompt
            prompt = f"生成{op_name}算子。{task_desc}使用 AscendC 实现。"
            
            logger.info(f"调用 lingxi-code agent 生成 {op_name} 算子...")
            logger.info(f"Task file: {task_file}")
            logger.info(f"Output dir: {output_dir}")
            
            # 使用 opencode run 直接调用 lingxi-code
            cmd = [
                "opencode", "run",
                "--agent", "lingxi-code",
                prompt
            ]
            
            logger.info(f"执行命令: {' '.join(cmd[:4])}...")
            
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workspace
            )
            
            # 记录输出
            if proc.stdout:
                logger.info(f"Agent stdout:\n{proc.stdout[:1000]}...")  # 截断长输出
            if proc.stderr:
                logger.warning(f"Agent stderr:\n{proc.stderr[:1000]}...")
            
            # 检查返回码
            if proc.returncode != 0:
                result.success = False
                result.error_message = f"Agent 执行失败，返回码: {proc.returncode}\nstderr: {proc.stderr}\nstdout: {proc.stdout}"
                logger.error(result.error_message)
                return result
            
            # 检查生成的代码文件
            generated_code_dir = os.path.join(self.workspace, "output", op_name)
            if os.path.exists(generated_code_dir):
                # 查找生成的代码文件
                code_files = glob.glob(os.path.join(generated_code_dir, "*.py"))
                if code_files:
                    with open(code_files[0], "r", encoding="utf-8") as f:
                        result.code = f.read()
                    result.success = True
                    logger.info(f"代码生成成功，文件: {code_files[0]}")
                else:
                    result.success = False
                    result.error_message = f"代码生成失败，未找到输出文件"
                    logger.error(result.error_message)
            else:
                result.success = False
                result.error_message = f"代码生成失败，未找到输出目录: {generated_code_dir}"
                logger.error(result.error_message)
                
        except subprocess.TimeoutExpired:
            result.success = False
            result.error_message = f"代码生成超时（{timeout}秒）"
            logger.error(result.error_message)
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"代码生成失败: {e}")
        
        result.generation_time = time.time() - start_time
        return result


class VerifierCaller:
    """验证器调用器"""
    
    def __init__(self, workspace: str):
        self.workspace = workspace
        self.scripts_dir = os.path.join(self.workspace, "skills", "ascendc_evalution", "scripts")
    
    def verify(self, op_name: str, timeout: int = 300) -> VerifyResult:
        """调用验证脚本"""
        result = VerifyResult()
        
        try:
            verify_script = os.path.join(self.scripts_dir, "evaluate.py")
            
            if not os.path.exists(verify_script):
                logger.warning(f"验证脚本不存在: {verify_script}")
                result.success = False
                result.error_message = "验证脚本不存在"
                return result
            
            cmd = [
                "python3", verify_script,
                op_name
            ]
            
            logger.info(f"执行验证: {' '.join(cmd)}")
            
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workspace
            )
            
            if proc.returncode == 0 and "验证成功" in proc.stdout:
                result.success = True
                result.compiled = True
                result.correctness = True
            else:
                result.success = False
                result.error_message = proc.stderr or proc.stdout
                
        except subprocess.TimeoutExpired:
            result.success = False
            result.error_message = f"验证超时（{timeout}秒）"
        except Exception as e:
            result.success = False
            result.error_message = str(e)
        
        return result
    
    def benchmark(self, op_name: str, output_file: str, warmup: int = 5, repeats: int = 50, timeout: int = 300) -> PerformanceResult:
        """调用性能测试脚本"""
        result = PerformanceResult()
        
        try:
            benchmark_script = os.path.join(self.scripts_dir, "benchmark.py")
            
            if not os.path.exists(benchmark_script):
                logger.warning(f"性能测试脚本不存在: {benchmark_script}")
                result.success = False
                result.error_message = "性能测试脚本不存在"
                return result
            
            cmd = [
                "python3", benchmark_script,
                "--op_name", op_name,
                "--warmup", str(warmup),
                "--repeats", str(repeats),
                "--output", output_file
            ]
            
            logger.info(f"执行性能测试: {' '.join(cmd)}")
            
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workspace
            )
            
            if proc.returncode == 0 and os.path.exists(output_file):
                # 读取性能结果
                with open(output_file, "r") as f:
                    perf_data = json.load(f)
                
                result.success = True
                result.avg_latency_ms = perf_data.get("avg_latency_ms", 0)
                result.p50_latency_ms = perf_data.get("p50_latency_ms", 0)
                result.p99_latency_ms = perf_data.get("p99_latency_ms", 0)
                result.peak_memory_mb = perf_data.get("peak_memory_mb", 0)
                result.speedup_vs_torch = perf_data.get("speedup_vs_torch", 0)
            else:
                result.success = False
                result.error_message = proc.stderr or proc.stdout
                
        except subprocess.TimeoutExpired:
            result.success = False
            result.error_message = f"性能测试超时（{timeout}秒）"
        except Exception as e:
            result.success = False
            result.error_message = str(e)
        
        return result


class ReportGenerator:
    """报告生成器"""
    
    @staticmethod
    def generate_agent_report(agent_name: str, results: List[EvaluationResult], output_file: str):
        """生成单个 Agent 报告"""
        
        # 统计计算
        total = len(results)
        compiled = sum(1 for r in results if r.verification and r.verification.compiled)
        correct = sum(1 for r in results if r.verification and r.verification.correctness)
        
        speedups = [
            r.performance.speedup_vs_torch 
            for r in results 
            if r.performance and r.performance.success
        ]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0
        
        # 按 Level 统计（包含优于PyTorch的比例）
        level_stats = {}
        for level in sorted(set(r.level for r in results)):
            level_results = [r for r in results if r.level == level]
            if level_results:
                level_compiled = sum(1 for r in level_results if r.verification and r.verification.compiled)
                level_correct = sum(1 for r in level_results if r.verification and r.verification.correctness)
                level_speedups = [
                    r.performance.speedup_vs_torch 
                    for r in level_results 
                    if r.performance and r.performance.success
                ]
                level_avg_speedup = sum(level_speedups) / len(level_speedups) if level_speedups else 0
                # 统计优于PyTorch的数量（speedup > 1.0）
                better_than_torch = sum(1 for s in level_speedups if s > 1.0)
                better_ratio = better_than_torch / len(level_speedups) * 100 if level_speedups else 0
                
                level_stats[level] = {
                    "total": len(level_results),
                    "compiled": level_compiled,
                    "correct": level_correct,
                    "speedups": level_speedups,
                    "avg_speedup": level_avg_speedup,
                    "better_than_torch": better_than_torch,
                    "better_ratio": better_ratio
                }
        
        # 按算子类型统计
        type_stats = {}
        for result in results:
            op_type = result.op_type
            if op_type not in type_stats:
                type_stats[op_type] = {"total": 0, "compiled": 0, "correct": 0, "speedups": []}
            type_stats[op_type]["total"] += 1
            if result.verification:
                if result.verification.compiled:
                    type_stats[op_type]["compiled"] += 1
                if result.verification.correctness:
                    type_stats[op_type]["correct"] += 1
            if result.performance and result.performance.success:
                type_stats[op_type]["speedups"].append(result.performance.speedup_vs_torch)
        
        # 收集失败案例
        compile_failed = [r for r in results if r.verification and not r.verification.compiled]
        verify_failed = [r for r in results if r.verification and r.verification.compiled and not r.verification.correctness]
        perf_degraded = [r for r in results if r.performance and r.performance.success and r.performance.speedup_vs_torch < 1.0]
        
        # 生成 Markdown 报告
        report = f"""# Agent: {agent_name} 评测报告

## 执行摘要
- 执行时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 总任务数: {total}

## 总体统计
| 指标 | Level1 | Level2 | Level3 | Level4 | 总体 |
|------|--------|--------|--------|--------|------|
| 任务数 | {level_stats.get(1, {}).get("total", 0)} | {level_stats.get(2, {}).get("total", 0)} | {level_stats.get(3, {}).get("total", 0)} | {level_stats.get(4, {}).get("total", 0)} | {total} |
| 编译成功 | {level_stats.get(1, {}).get("compiled", 0)} ({level_stats.get(1, {}).get("compiled", 0)/level_stats.get(1, {}).get("total", 1)*100:.0f}%) | {level_stats.get(2, {}).get("compiled", 0)} ({level_stats.get(2, {}).get("compiled", 0)/level_stats.get(2, {}).get("total", 1)*100:.0f}%) | {level_stats.get(3, {}).get("compiled", 0)} ({level_stats.get(3, {}).get("compiled", 0)/level_stats.get(3, {}).get("total", 1)*100:.0f}%) | {level_stats.get(4, {}).get("compiled", 0)} ({level_stats.get(4, {}).get("compiled", 0)/level_stats.get(4, {}).get("total", 1)*100:.0f}%) | {compiled} ({compiled/total*100:.0f}%) |
| 数值正确 | {level_stats.get(1, {}).get("correct", 0)} ({level_stats.get(1, {}).get("correct", 0)/level_stats.get(1, {}).get("total", 1)*100:.0f}%) | {level_stats.get(2, {}).get("correct", 0)} ({level_stats.get(2, {}).get("correct", 0)/level_stats.get(2, {}).get("total", 1)*100:.0f}%) | {level_stats.get(3, {}).get("correct", 0)} ({level_stats.get(3, {}).get("correct", 0)/level_stats.get(3, {}).get("total", 1)*100:.0f}%) | {level_stats.get(4, {}).get("correct", 0)} ({level_stats.get(4, {}).get("correct", 0)/level_stats.get(4, {}).get("total", 1)*100:.0f}%) | {correct} ({correct/total*100:.0f}%) |
| 优于PyTorch | {level_stats.get(1, {}).get("better_than_torch", 0)} ({level_stats.get(1, {}).get("better_ratio", 0):.0f}%) | {level_stats.get(2, {}).get("better_than_torch", 0)} ({level_stats.get(2, {}).get("better_ratio", 0):.0f}%) | {level_stats.get(3, {}).get("better_than_torch", 0)} ({level_stats.get(3, {}).get("better_ratio", 0):.0f}%) | {level_stats.get(4, {}).get("better_than_torch", 0)} ({level_stats.get(4, {}).get("better_ratio", 0):.0f}%) | {sum(s > 1.0 for s in speedups)} ({sum(s > 1.0 for s in speedups)/len(speedups)*100 if speedups else 0:.0f}%) |
| 平均加速比 | {level_stats.get(1, {}).get("avg_speedup", 0):.2f}x | {level_stats.get(2, {}).get("avg_speedup", 0):.2f}x | {level_stats.get(3, {}).get("avg_speedup", 0):.2f}x | {level_stats.get(4, {}).get("avg_speedup", 0):.2f}x | {avg_speedup:.2f}x |

## 按算子类型统计
| 类型 | 任务数 | 编译成功 | 数值正确 | 平均加速比 |
|------|--------|----------|----------|------------|
"""
        
        for op_type in sorted(type_stats.keys()):
            stats = type_stats[op_type]
            avg_spd = sum(stats["speedups"]) / len(stats["speedups"]) if stats["speedups"] else 0
            report += f"| {op_type} | {stats['total']} | {stats['compiled']} ({stats['compiled']/stats['total']*100:.0f}%) | {stats['correct']} ({stats['correct']/stats['total']*100:.0f}%) | {avg_spd:.2f}x |\n"
        
        # 编译失败列表
        if compile_failed:
            report += """

## 编译失败列表

"""
            for level in sorted(set(r.level for r in compile_failed)):
                level_failures = [r for r in compile_failed if r.level == level]
                if level_failures:
                    report += f"""
### Level {level}
| Problem ID | 算子名 | 错误信息 |
|------------|--------|----------|
"""
                    for result in sorted(level_failures, key=lambda x: x.problem_id):
                        error = result.generation.error_message[:80] if result.generation and result.generation.error_message else (result.verification.error_message[:80] if result.verification else "未知错误")
                        report += f"| {result.problem_id} | {result.op_name} | {error}... |\n"
                    report += "\n"
        
        # 数值验证失败列表
        if verify_failed:
            report += """
## 数值验证失败列表

"""
            for level in sorted(set(r.level for r in verify_failed)):
                level_failures = [r for r in verify_failed if r.level == level]
                if level_failures:
                    report += f"""
### Level {level}
| Problem ID | 算子名 | Max Diff | 错误类型 |
|------------|--------|----------|----------|
"""
                    for result in sorted(level_failures, key=lambda x: x.problem_id):
                        max_diff = result.verification.max_diff if result.verification and result.verification.max_diff else "N/A"
                        error_type = "精度超标" if max_diff != "N/A" else "数值异常"
                        report += f"| {result.problem_id} | {result.op_name} | {max_diff} | {error_type} |\n"
                    report += "\n"
        
        # 性能劣化列表
        if perf_degraded:
            report += """
## 性能劣化列表（相比 PyTorch）

"""
            for level in sorted(set(r.level for r in perf_degraded)):
                level_degraded = [r for r in perf_degraded if r.level == level]
                if level_degraded:
                    report += f"""
### Level {level}
| Problem ID | 算子名 | 加速比 | 劣化倍数 |
|------------|--------|--------|----------|
"""
                    for result in sorted(level_degraded, key=lambda x: x.problem_id):
                        speedup = result.performance.speedup_vs_torch if result.performance else 0
                        degradation = 1.0 / speedup if speedup > 0 else float('inf')
                        report += f"| {result.problem_id} | {result.op_name} | {speedup:.2f}x | {degradation:.2f}x |\n"
                    report += "\n"
        
        report += """

## 详细结果表
| Problem | 算子名 | 类型 | 编译 | 正确 | 加速比 | 状态 |
|---------|--------|------|------|------|--------|------|
"""
        
        for result in sorted(results, key=lambda x: (x.level, x.problem_id)):
            compiled = "✓" if result.verification and result.verification.compiled else "✗"
            correct = "✓" if result.verification and result.verification.correctness else "✗"
            speedup = f"{result.performance.speedup_vs_torch:.2f}x" if result.performance and result.performance.success else "-"
            status = "成功" if result.verification and result.verification.correctness else "失败"
            report += f"| {result.problem_id} | {result.op_name} | {result.op_type} | {compiled} | {correct} | {speedup} | {status} |\n"
        
        # 保存报告
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info(f"Agent 报告已生成: {output_file}")


class BenchmarkEvaluator:
    """Benchmark 评测器主类"""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run{self.timestamp[-6:]}"
        self.output_dir = os.path.join(
            config.output_dir, 
            f"lingxi-code_{self.timestamp}_{self.run_id}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化状态管理器
        self.state_manager = StateManager(self.output_dir)
        
        # 如果用户指定了 arch 或 npu_id，更新状态
        if config.arch:
            self.state_manager.update_arch(config.arch)
        if config.npu_id is not None:
            self.state_manager.update_npu_id(config.npu_id)
        
        # 设置环境变量
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(self.state_manager.npu_id)
        
        # 记录所有结果
        self.results: List[EvaluationResult] = []
        
        # 工作目录
        self.workspace = os.getcwd()
    
    def run(self):
        """执行评测"""
        logger.info("=" * 60)
        logger.info("开始 Ascend Benchmark 评测")
        logger.info(f"Benchmark 路径: {self.config.benchmark_path}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"目标架构: {self.state_manager.arch}")
        logger.info(f"NPU ID: {self.state_manager.npu_id}")
        logger.info("=" * 60)
        
        # 1. 扫描任务
        logger.info("步骤 1: 扫描任务...")
        tasks = TaskScanner.scan_tasks(self.config.benchmark_path)
        logger.info(f"发现 {len(tasks)} 个任务")
        
        # 2. 过滤已完成的任务
        pending_tasks = []
        for task in tasks:
            if self.config.resume and self.state_manager.is_completed(
                task["level"], task["problem_id"]
            ):
                logger.debug(f"跳过已完成任务: Level {task['level']} Problem {task['problem_id']}")
                continue
            pending_tasks.append(task)
        
        logger.info(f"需要评测 {len(pending_tasks)} 个任务 (已完成 {len(tasks) - len(pending_tasks)} 个)")
        
        # 3. 串行执行评测
        logger.info("步骤 2: 开始评测（串行执行）...")
        
        for i, task in enumerate(pending_tasks):
            logger.info(f"进度: {i+1}/{len(pending_tasks)} - Level {task['level']} Problem {task['problem_id']} - {task['op_name']}")
            
            try:
                result = self.evaluate_single_task(task)
                
                # 保存结果
                self.results.append(result)
                
                # 标记完成
                self.state_manager.mark_completed(
                    result.level,
                    result.problem_id
                )
                
                # 增量更新报告
                report_file = os.path.join(self.output_dir, "agent_report.md")
                ReportGenerator.generate_agent_report(
                    "lingxi-code",
                    self.results,
                    report_file
                )
                
            except Exception as e:
                logger.error(f"任务执行失败: {e}")
                logger.error(traceback.format_exc())
        
        # 4. 生成最终报告
        logger.info("步骤 3: 生成最终报告...")
        
        report_file = os.path.join(self.output_dir, "agent_report.md")
        ReportGenerator.generate_agent_report(
            "lingxi-code",
            self.results,
            report_file
        )
        
        # 保存原始数据
        all_results_json = os.path.join(self.output_dir, "all_results.json")
        with open(all_results_json, "w") as f:
            json.dump(
                [asdict(r) for r in self.results],
                f,
                indent=2,
                default=str
            )
        
        logger.info("=" * 60)
        logger.info("评测完成!")
        logger.info(f"结果保存在: {self.output_dir}")
        logger.info(f"报告文件: {report_file}")
        logger.info("=" * 60)
    
    def evaluate_single_task(self, task_info: Dict[str, Any]) -> EvaluationResult:
        """评测单个任务"""
        
        level = task_info["level"]
        problem_id = task_info["problem_id"]
        task_file = task_info["task_file"]
        op_name = task_info["op_name"]
        
        # 分类算子类型
        op_type = TaskScanner.classify_op_type(op_name)
        
        # 创建输出目录
        task_output_dir = os.path.join(
            self.output_dir,
            f"level_{level}",
            f"{problem_id}_{op_name}"
        )
        os.makedirs(task_output_dir, exist_ok=True)
        
        # 初始化结果
        result = EvaluationResult(
            level=level,
            problem_id=problem_id,
            op_name=op_name,
            op_type=op_type,
            timestamp=datetime.now().isoformat(),
            output_dir=task_output_dir
        )
        
        try:
            # 1. 代码生成
            logger.info(f"[Level {level} Problem {problem_id}]: 生成代码...")
            agent_caller = LingxiCodeCaller(self.workspace)
            
            gen_result = agent_caller.generate_operator(
                op_name=op_name,
                task_file=task_file,
                output_dir=task_output_dir,
                timeout=self.config.timeout_per_task
            )
            result.generation = gen_result
            
            if not gen_result.success:
                logger.error(f"代码生成失败: {gen_result.error_message}")
                return result
            
            # 2. 正确性验证
            logger.info(f"[Level {level} Problem {problem_id}]: 验证正确性...")
            verifier = VerifierCaller(self.workspace)
            verify_result = verifier.verify(
                op_name=op_name,
                timeout=self.config.timeout_per_task
            )
            result.verification = verify_result
            
            # 保存验证结果
            verify_result_file = os.path.join(task_output_dir, "verify_result.json")
            with open(verify_result_file, "w") as f:
                json.dump(asdict(verify_result), f, indent=2)
            
            if not verify_result.success:
                logger.warning(f"验证失败: {verify_result.error_message}")
                return result
            
            # 3. 性能测试
            logger.info(f"[Level {level} Problem {problem_id}]: 性能测试...")
            perf_output_file = os.path.join(task_output_dir, "perf_result.json")
            perf_result = verifier.benchmark(
                op_name=op_name,
                output_file=perf_output_file,
                warmup=self.config.warmup,
                repeats=self.config.repeats,
                timeout=self.config.timeout_per_task
            )
            result.performance = perf_result
            
            logger.info(
                f"[Level {level} Problem {problem_id}]: "
                f"完成 (编译: {verify_result.compiled}, "
                f"正确: {verify_result.correctness}, "
                f"加速比: {perf_result.speedup_vs_torch:.2f}x)"
            )
            
        except Exception as e:
            logger.error(f"评测异常: {e}")
            logger.error(traceback.format_exc())
            result.generation = result.generation or GenerationResult()
            result.generation.error_message = str(e)
        
        return result


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ascend Benchmark Evaluation")
    parser.add_argument("--benchmark_path", required=True, help="Benchmark 数据集路径")
    parser.add_argument("--output_dir", default="./benchmark_results", help="输出目录")
    parser.add_argument("--arch", default="", help="目标硬件架构")
    parser.add_argument("--npu_id", type=int, default=None, help="NPU 设备序号")
    parser.add_argument("--resume", type=bool, default=True, help="是否断点续跑")
    parser.add_argument("--timeout_per_task", type=int, default=2400, help="单任务超时（秒）")
    parser.add_argument("--warmup", type=int, default=5, help="性能测试 warmup 次数")
    parser.add_argument("--repeats", type=int, default=50, help="性能测试重复次数")
    
    args = parser.parse_args()
    
    # 解析 benchmark_path
    benchmark_path = args.benchmark_path
    if not os.path.isabs(benchmark_path):
        if "/" in benchmark_path:
            # 相对路径
            benchmark_path = os.path.join(os.getcwd(), benchmark_path)
        else:
            # 数据集名称
            benchmark_path = os.path.join(os.getcwd(), "benchmarks", benchmark_path)
    
    config = TaskConfig(
        benchmark_path=benchmark_path,
        output_dir=args.output_dir,
        arch=args.arch if args.arch else "ascend910b1",
        npu_id=args.npu_id if args.npu_id is not None else 0,
        resume=args.resume,
        timeout_per_task=args.timeout_per_task,
        warmup=args.warmup,
        repeats=args.repeats
    )
    
    evaluator = BenchmarkEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
