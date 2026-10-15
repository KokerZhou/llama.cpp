#!/bin/bash
# Qwen3-4B 冒烟（note-29 P1 基线口径，910B3）。
# 用法: bash scripts/03_smoke_q3_4b.sh [日志名后缀] [额外 llama-cli 参数]
# 教训内置：stdout 截断防 "> " EOF 洪泛（note-27）；--single-turn -n 显式（note-29）。
set -uo pipefail

SUFFIX="${1:-run1}"
EXTRA="${2:-}"

source /home/developer/Ascend/ascend-toolkit/set_env.sh
export GGML_CANNGE_GRAPH=1

MODEL="$HOME/models/Qwen3-4B-Instruct-2507-F16.gguf"
BIN=/home/developer/cannge-kit/cannge-ggml/build/bin/llama-cli
LOG="$HOME/smoke-q3-4b-$SUFFIX.log"

echo "== smoke start $(date '+%F %T') suffix=$SUFFIX extra='$EXTRA'"
# 输出截断：tail -c 上限 1MB，防止乱码/EOF 洪泛写爆磁盘
timeout 1800 "$BIN" \
  -m "$MODEL" \
  -ngl all \
  -c 512 -ub 512 --temp 0.6 --no-warmup \
  --single-turn --no-display-prompt \
  -p "你好，请用一句话介绍自己。" -n 40 \
  $EXTRA 2>&1 | tail -c 1M > "$LOG"
RC=${PIPESTATUS[0]}
echo "== smoke end $(date '+%F %T') rc=$RC log=$LOG"
tail -40 "$LOG"
exit "$RC"
