#!/bin/bash
# CANNGE 构建 + TBO 冒烟。用法: bash scripts/02_build_and_test.sh <SOC_TYPE> [仓库目录]
#   SOC_TYPE: 310 -> Ascend310P3 ; 910 -> Ascend910B3（或按 npu-smi 实际型号）
set -euo pipefail

SOC="${1:?缺少 SOC_TYPE，如 Ascend310P3 / Ascend910B3}"
DIR="${2:-cannge-ggml}"

# toolkit 环境：容器路径优先，用户级安装回退
if [ -f /usr/local/Ascend/cann-9.1.0/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann-9.1.0/set_env.sh
elif [ -f "$HOME/Ascend/ascend-toolkit/set_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/Ascend/ascend-toolkit/set_env.sh"
else
    echo "找不到 set_env.sh" >&2; exit 1
fi

cd "$DIR"
cmake -B build -DGGML_CANNGE=ON -DGGML_CANN=OFF -DSOC_TYPE="$SOC" -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ggml-cannge test-backend-ops -j"$(nproc)"

echo "== TBO 窄集（基线口径见 note-16 §3）=="
GGML_CANNGE_GRAPH=1 ./build/bin/test-backend-ops test -b CANNGE0 \
    -o ADD,MUL,MUL_MAT,SCALE,RMS_NORM,SOFT_MAX,SILU,GET_ROWS,CONCAT || true
echo "（退出码 1 = 存在数值 FAIL，属正常；基线 160 OK / 45 FAIL / 2366 NS）"
