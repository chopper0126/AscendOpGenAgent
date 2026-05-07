#!/usr/bin/env bash
# PostToolUse hook: 当 model_new_tilelang.py / model_new_ascendc.py 被写入后，
# 自动执行退化检测，结果写入 .validate_<type>_result.json
#
# 环境变量（由 Claude Code hook 系统提供）:
#   CLAUDE_TOOL_INPUT_FILE_PATH  - 被写入的文件路径
#   CLAUDE_TOOL_NAME             - 工具名称 (Write / Edit)

set -euo pipefail

FILE_PATH="${CLAUDE_TOOL_INPUT_FILE_PATH:-}"
if [[ -z "${FILE_PATH}" ]]; then
  exit 0
fi

BASENAME="$(basename "${FILE_PATH}")"

# 确定项目和 skill 路径
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SKILLS_DIR="${PROJECT_ROOT}/skills/ascendc"

case "${BASENAME}" in
  model_new_tilelang.py)
    VALIDATOR="${SKILLS_DIR}/tilelang-designer/scripts/validate_tilelang_impl.py"
    TYPE="tilelang"
    ;;
  model_new_ascendc.py)
    VALIDATOR="${SKILLS_DIR}/ascendc-translator/scripts/validate_ascendc_impl.py"
    TYPE="ascendc"
    ;;
  *)
    exit 0
    ;;
esac

OUTPUT_DIR="$(dirname "${FILE_PATH}")"
RESULT_FILE="${OUTPUT_DIR}/.validate_${TYPE}_result.json"

if [[ -f "${VALIDATOR}" ]]; then
  python "${VALIDATOR}" "${FILE_PATH}" --json > "${RESULT_FILE}" 2>&1 || true
fi
