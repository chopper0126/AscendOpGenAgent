#!/usr/bin/env bash
# 包装器: 执行 performance.py，输出重定向到文件，
# 仅向 stdout 打印摘要（不占用上下文）
#
# 用法:
#   bash wrap-performance.sh <output_dir>

set -euo pipefail

OUTPUT_DIR="${1:-}"

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "用法: $0 <output_dir>"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PERF_SCRIPT="${PROJECT_ROOT}/skills/ascendc/performance-analyzer/references/performance.py"

if [[ ! -f "${PERF_SCRIPT}" ]]; then
  echo "性能测试脚本不存在: ${PERF_SCRIPT}"
  exit 1
fi

if [[ ! -d "${OUTPUT_DIR}" ]]; then
  echo "输出目录不存在: ${OUTPUT_DIR}"
  exit 1
fi

LOG_FILE="${OUTPUT_DIR}/.last_performance.log"
SUMMARY_FILE="${OUTPUT_DIR}/.last_performance.summary"
REPORT_FILE="${OUTPUT_DIR}/preformance.json"

echo "===== 性能测试开始 $(date) =====" > "${LOG_FILE}"

set +e
python "${PERF_SCRIPT}" --output_dir "${OUTPUT_DIR}" --output "${REPORT_FILE}" >> "${LOG_FILE}" 2>&1
EXIT_CODE=$?
set -e

echo "===== 性能测试结束 $(date) exit=${EXIT_CODE} =====" >> "${LOG_FILE}"

# 提取摘要: Overall speedup 和 per-case 加速比
if [[ ${EXIT_CODE} -eq 0 ]]; then
  echo "PASS: 性能测试完成" > "${SUMMARY_FILE}"
  # 提取加速比行
  grep -E "(Overall speedup|Speedup|speedup|加速)" "${LOG_FILE}" >> "${SUMMARY_FILE}" 2>/dev/null || true
else
  echo "FAIL (exit=${EXIT_CODE}): 性能测试失败" > "${SUMMARY_FILE}"
  tail -20 "${LOG_FILE}" >> "${SUMMARY_FILE}" 2>/dev/null || true
fi

# 向 stdout 输出摘要
cat "${SUMMARY_FILE}"
echo ""
echo "[完整日志: ${LOG_FILE}]"
echo "[性能报告: ${REPORT_FILE}]"
