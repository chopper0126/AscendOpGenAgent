#!/bin/bash

# ==========================================
# 1. 参数处理与默认值设置
# ==========================================

# 默认配置
DEFAULT_BASE_DIR="/home/y00889327/AscendOpGenAgent/output_ascend_multi_0413_L2_from_json_new_good"
DEFAULT_LOG_FILE="log_preformance_0414.txt"

# 获取传入参数 (如果存在则使用传入值，否则使用默认值)
# ${1:-$DEFAULT} 表示如果 $1 为空，则使用 $DEFAULT
BASE_DIR="${1:-$DEFAULT_BASE_DIR}"
LOG_FILE="${2:-$DEFAULT_LOG_FILE}"

# 将相对路径的日志文件转换为绝对路径 (防止因目录切换导致日志找不到)
# $(pwd) 获取当前执行脚本的目录
if [[ "$LOG_FILE" != /* ]]; then
    LOG_FILE="$(pwd)/$LOG_FILE"
fi

# 定义锁目录 (保持在 /tmp 以防不同运行实例冲突)
LOCK_DIR="/tmp/ascend_locks_$(whoami)"
mkdir -p "$LOCK_DIR"

# 清空或创建日志文件
> "$LOG_FILE"

echo ">>> 启动配置:"
echo "    数据目录: $BASE_DIR"
echo "    日志文件: $LOG_FILE"
echo "    锁目录  : $LOCK_DIR"
echo "----------------------------------------"

# 检查基础目录是否存在
if [ ! -d "$BASE_DIR" ]; then
    echo "错误: 目录不存在 -> $BASE_DIR"
    exit 1
fi

# ==========================================
# 2. 主循环逻辑
# ==========================================

# 初始化设备 ID
device_id=1

# 遍历目录下所有子文件夹
for dir in "$BASE_DIR"/*; do
    if [ -d "$dir" ]; then
        folder_name=$(basename "$dir")
        
        # --- 步骤 1：控制总并发数 (防止系统过载) ---
        # 等待直到运行中的后台任务数量小于 7
        while [ $(jobs -r | wc -l) -ge 7 ]; do
            sleep 0.5
        done

        # --- 步骤 2：寻找空闲设备 (防止 NPU 冲突) ---
        while true; do
            lock_file="$LOCK_DIR/device_${device_id}.lock"
            
            # 尝试非阻塞加锁
            # 如果加锁失败（返回非0），说明设备忙
            if ! ( set -o noclobber; flock -n 200 ) 200>"$lock_file" 2>/dev/null; then
                # 设备忙，切换到下一个设备
                device_id=$(( (device_id % 7) + 1 ))
                sleep 0.2 
            else
                # 设备空闲，跳出循环准备启动任务
                break
            fi
        done
        
        echo ">>> 锁定设备 $device_id 并启动算子: $folder_name" | tee -a "$LOG_FILE"
        
        # --- 步骤 3：启动后台任务 ---
        (
            # 在子 Shell 中重新获取并持有着锁
            lock_file="$LOCK_DIR/device_${device_id}.lock"
            exec 200>"$lock_file"
            flock 200
            
            # 设置环境变量
            export ASCEND_RT_VISIBLE_DEVICES=$device_id
            
            # 执行 Python 任务
            # 注意：这里使用了双引号包裹变量，防止路径中有空格出错
            python3 /home/y00889327/AscendOpGenAgent/skills/ascendc/performance-analyzer/references/performance.py "$dir" 2>&1 | \
            grep -v "tiling struct \[MC2MatmulV3TilingData\] is conflict" | \
            grep -v "tiling struct \[TileInfo\] is conflict" \
            >> "$LOG_FILE"
            
            echo ">>> 算子 $folder_name 在设备 $device_id 上执行完毕" >> "$LOG_FILE"
            
            # 任务结束，文件描述符 200 关闭，锁自动释放
        ) &
        
        # 切换设备 ID，为下一个任务做准备 (轮询算法)
        device_id=$(( (device_id % 7) + 1 ))
    fi
done

# ==========================================
# 3. 等待与清理
# ==========================================

# 等待所有后台任务完成
wait

# 清理锁目录
rm -rf "$LOCK_DIR"

echo "----------------------------------------"
echo "所有并行任务已执行完毕，日志已保存至: $LOG_FILE"