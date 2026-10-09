# 给服务器侧 Kimi Code 的提示词（用户直接粘贴本文件全文）

你在一台带 Ascend NPU 的开发机上接手 CANNGE 后端项目。交接包已放在 `$HOME/cannge-kit/`（tarball 解开后的目录），先读 `$HOME/cannge-kit/HANDOFF.md`，再按它给的顺序行动。

## 背景

CANNGE = ggml 的 Ascend GE 整图编译后端，实现在 `ggml-org/llama.cpp` 的 fork
（`KokerZhou/llama.cpp`）上，分支 `cannge`，基线 CANN 9.1。交接包里的
`cannge-docs/`（note-1~20 + README）是全部决策、调研、findings 台账和规范，
`HANDOFF.md` 有文件地图和当前停点；`probes/` 是 310 探针源码；`scripts/` 是
仓库对齐和构建冒烟脚本。

## 第一步（只做这些，完成后停下来向我汇报，等我指派后续任务）

1. 我随后会把 PAT 发给你：先 `export GH_TOKEN=<PAT>`，然后
   `bash $HOME/cannge-kit/scripts/01_clone_repo.sh`（克隆 fork 的 cannge 分支到
   `./cannge-ggml`，预期 HEAD 为 `28558340a`）。
2. 确认本机 SoC 型号（`npu-smi info`，注意先 `source` set_env.sh 或加库路径），
   然后 `bash $HOME/cannge-kit/scripts/02_build_and_test.sh <SOC_TYPE>` 跑构建+TBO 冒烟，
   把逐 op OK/FAIL/NS 数报给我，与基线（note-16 §3：160/45/2366）对比。
3. 汇报内容：HEAD 是否对齐、构建是否零错误、TBO 对比表、环境信息（SoC/CANN 版本/driver）。

## 纪律（HANDOFF.md §纪律 的摘要，违反即返工）

- 未获我明示，禁止 git commit / push；
- 禁止 abort，不支持即返回 false；
- 注释必须按 note-11/note-13 规范带官方出处，ASCII，1-2 行；
- 凭证不落盘文件、不进 repo、不进笔记，用完可以 `rm ~/.git-credentials`；
- 共享环境不动他人目录与全局配置；跑测试前 `npu-smi info` 确认无他人进程，结束清理自己的进程。

当前热点（先读再动，不要自作主张开工）：note-10 的 F7（判别实验设计在 note-20 §5.2）
和 F8（修法已定在 note-10 F8，等我批准）。
