#!/bin/bash
# CANNGE / CANN+graph / CANN-eager / CPU 基准对比（910B3, Qwen3-4B F16）
# 一次运行同测 prefill/decode：长 prompt 出 Prompt t/s，-n 64 出 Generation t/s
set -uo pipefail
source /home/developer/Ascend/ascend-toolkit/set_env.sh

MODEL="$HOME/models/Qwen3-4B-Instruct-2507-F16.gguf"
CANNGE_BIN=/home/developer/cannge-kit/cannge-ggml/build/bin/llama-cli
CANN_BIN=/home/developer/cannge-kit/cannge-ggml/build-cann/bin/llama-cli
OUT=/home/developer/bench-results.txt
: > "$OUT"

# ~300 token 中文 prompt（repeat 构造，ubatch 512 内一次 prefill）
LONG="请阅读下面的段落并准备总结。"
for i in $(seq 1 10); do LONG+="机器学习是人工智能的重要分支，它从数据中学习规律并做出预测。深度学习使用多层神经网络逼近复杂函数。"; done

run() { # name binary extra_env...
    local name="$1" bin="$2"; shift 2
    echo "== $name ==" | tee -a "$OUT"
    env "$@" "$bin" -m "$MODEL" -ngl all -c 512 -ub 512 --temp 0 --no-warmup \
        --single-turn --no-display-prompt -p "$LONG" -n 64 2>&1 | \
        grep -aE 'Prompt:|Generation:' | tail -2 | tee -a "$OUT"
}

run_cpu() {
    echo "== cpu-ngl0 ==" | tee -a "$OUT"
    "$CANNGE_BIN" -m "$MODEL" -ngl 0 -c 512 -ub 64 --temp 0 --no-warmup \
        --single-turn --no-display-prompt -p "$LONG" -n 64 2>&1 | \
        grep -aE 'Prompt:|Generation:' | tail -2 | tee -a "$OUT"
}

run_cpu
run "cann-eager"        "$CANN_BIN" GGML_CANN_ACL_GRAPH=off
run "cann-graph"        "$CANN_BIN"
run "cann-graph-prefill" "$CANN_BIN" GGML_CANN_PREFILL_USE_GRAPH=1
run "cann-graph-minb1"  "$CANN_BIN" GGML_OP_OFFLOAD_MIN_BATCH=1
run "cannge"            "$CANNGE_BIN" GGML_CANNGE_GRAPH=1
run "cannge-nkvo"       "$CANNGE_BIN" GGML_CANNGE_GRAPH=1
echo BENCH_ALL_DONE | tee -a "$OUT"