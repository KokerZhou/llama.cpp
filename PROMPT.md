# 给 910 服务器侧 Kimi Code 的提示词（用户直接粘贴本文件全文）

你在 910 开发机（1×910B3）上接手 CANNGE 项目，交接包在 `$HOME/cannge-kit/`（v2）。
先读 `$HOME/cannge-kit/HANDOFF.md`，再按它行动。你的前任在 310 上已完成一轮
（代码到 `d3a3ef81f`，笔记 note-21~29），你承接的是它的工作队列。

## 第一步（完成后停下向我汇报）

1. 通读 `cannge-docs/note-27-failure-chain.md` + `note-25-model-integration.md`
   + `note-24-fix-log.md`（当前状态与中断点都在里面）。
2. 我把 PAT 发你后：`export GH_TOKEN=<PAT>`，然后
   `bash $HOME/cannge-kit/scripts/01_clone_repo.sh`（预期 HEAD=`d3a3ef81f`）。
3. `bash $HOME/cannge-kit/scripts/02_build_and_test.sh Ascend910B3`，把逐 op
   OK/FAIL/NS 报给我，注明与 310 基线 324/333 的差异（注意 910 的 f32 matmul 门是
   8 元素口径，与 310P 不同属预期）。
4. 汇报：HEAD 对齐、构建结果、TBO 对比、环境确认（910B3/CANN 版本/driver）。

## 然后（等我点名开工，优先级见 HANDOFF 工作队列）

候选：① 冒烟续跑过 execute（note-27 #13）；② 病态编译二分（note-25 #8，910B 对照）；
③ RMS-in-graph 根因（note-27 #15）；④ TBO 基线对拍。一次只开一项。

## 纪律（违反即返工）

- 未获我明示禁止 git commit/push；
- 禁止 abort；注释按 note-11/note-13 带官方出处、ASCII、1-2 行；
- 凭证不落盘不进 repo；共享环境不动他人目录，测试前后 `npu-smi info` 确认并清理进程；
- 每批工作：改码→构建→验证→落盘 note（note-30 起编号），控制变量。
