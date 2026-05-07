#!/usr/bin/env bash
# 包装器: 执行 evaluate_ascendc.sh / evaluate_tilelang.sh，输出重定向到文件，
# 仅向 stdout 打印摘要（不占用上下文）
#
# 用法:
#   bash wrap-evaluate.sh <script_type> <output_dir>
#   script_type: ascendc | tilelang

set -euo pipefail

SCRIPT_TYPE="${1:-}"
OUTPUT_DIR="${2:-}"

if [[ -z "${SCRIPT_TYPE}" || -z "${OUTPUT_DIR}" ]]; then
  echo "用法: $0 <ascendc|tilelang> <output_dir>"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SKILLS_DIR="${PROJECT_ROOT}/skills/ascendc"

case "${SCRIPT_TYPE}" in
  ascendc)
    EVAL_SCRIPT="${SKILLS_DIR}/ascendc-translator/references/evaluate_ascendc.sh"
    LABEL="AscendC"
    ;;
  tilelang)
    EVAL_SCRIPT="${SKILLS_DIR}/tilelang-designer/references/evaluate_tilelang.sh"
    LABEL="TileLang"
    ;;
  *)
    echo "未知类型: ${SCRIPT_TYPE}，可选: ascendc, tilelang"
    exit 1
    ;;
esac

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "验证脚本不存在: ${EVAL_SCRIPT}"
  exit 1
fi

if [[ ! -d "${OUTPUT_DIR}" ]]; then
  echo "输出目录不存在: ${OUTPUT_DIR}"
  exit 1
fi

LOG_FILE="${OUTPUT_DIR}/.last_${SCRIPT_TYPE}_eval.log"
SUMMARY_FILE="${OUTPUT_DIR}/.last_${SCRIPT_TYPE}_eval.summary"

echo "===== ${LABEL} 验证开始 $(date) =====" > "${LOG_FILE}"

# 执行验证脚本，捕获所有输出
set +e
bash "${EVAL_SCRIPT}" "${OUTPUT_DIR}" >> "${LOG_FILE}" 2>&1
EXIT_CODE=$?
set -e

echo "===== 验证结束 $(date) exit=${EXIT_CODE} =====" >> "${LOG_FILE}"

# 提取摘要: 取最后 30 行中关键信息
if [[ ${EXIT_CODE} -eq 0 ]]; then
  echo "PASS: ${LABEL} 验证通过" > "${SUMMARY_FILE}"
else
  echo "FAIL (exit=${EXIT_CODE}): ${LABEL} 验证失败" > "${SUMMARY_FILE}"
fi

# 从日志尾部提取关键行（测试结果统计）
tail -30 "${LOG_FILE}" | grep -E "(PASS|FAIL|Error|error|通过|失败|assert|All|passed|failed|success|Total)" >> "${SUMMARY_FILE}" 2>/dev/null || true

# 向 stdout 输出摘要（进入 agent 上下文）
cat "${SUMMARY_FILE}"
echo ""
echo "[完整日志: ${LOG_FILE}]"
