#!/bin/bash
# CANNGE 仓库对齐：克隆（或更新）fork 的 cannge 分支。
# 用法: GH_TOKEN=<PAT> bash scripts/01_clone_repo.sh [目标目录]
# token 只写入 ~/.git-credentials（600），remote URL 不含 token。
set -euo pipefail

TOKEN="${GH_TOKEN:?请先把 PAT 放进 GH_TOKEN 环境变量}"
DEST="${1:-cannge-ggml}"
REPO="https://github.com/KokerZhou/llama.cpp.git"

umask 077
git config --global credential.helper store
printf 'https://x-access-token:%s@github.com\n' "$TOKEN" > "$HOME/.git-credentials"
chmod 600 "$HOME/.git-credentials"

if [ -d "$DEST/.git" ]; then
    git -C "$DEST" remote set-url origin "$REPO"
    git -C "$DEST" fetch origin
    git -C "$DEST" checkout cannge
    git -C "$DEST" reset --hard origin/cannge
else
    git clone --branch cannge "$REPO" "$DEST"
fi

echo "== 当前 HEAD =="
git -C "$DEST" log --oneline -3
echo "== 预期应为: 28558340a (fix) + d85b6bd08 (backend) =="
