# note-40: 减少图分割批 A——ROPE 上设备尝试：死代码修复 + 集成腐化现形（2026-10-14）

目标：减少图分割（note-39 方向 1/2）。本批攻 ROPE（每层 2 个切点之一）。

## 诊断链（全部实证）

1. **门拒绝根因**：DBG-GATE 三返回点插装 → Qwen3-Instruct-2507 的 rope
   带 `beta_fast=32.0/beta_slow=1.0`（YaRN 参数，ext_factor=0 时按
   ggml-cpu 语义是死参数）→ 门拒绝（"no YaRN scaling" 检查过于严格）。
   同时发现 decode rope 输入 [128,32,1] 的 `ggml_n_dims=2` 会被
   `nd>=3` 拒绝（另一个独立缺口）
2. **构建死代码 bug**（TBO ROPE 原本全 NS 从未执行过此路径）：
   `dup_pairs` 里 `Reshape(t, shape3)` 的结果被丢弃，Concat 用的是原始
   2D 张量 + axis=2 → GE 编译 E89999 "concat_dim should be in [-2,1]"
   → 修复为 concat reshape 后的 [n_tok, half, 1] → **TBO ROPE 13/13** ✓
3. **门放开 → 模型乱码**：prompt 相 [128,32,15] rope 上 CANNGE（decode 相
   因 n_dims=2 仍在 CPU）→ 输出 ".# Define the model" 类垃圾；
   NO_ROPE=1 对照完美 → 实锤 rope-in-graph 破坏
4. **精确探针** `ggml_probe_rope`（模型同款形状/参数/freq_base=1e6/
   betas 32/1）：**[128,32,15] max_abs 1.2e-7 PASS，[128,32,1] 门拒**——
   又是"孤立完美、进图即坏"（#20 同款模式）

## 处置

- **门回退**（严格 beta 检查恢复，注释记录本次尝试与原因）——连贯性优先
- 保留：dup_pairs 修复、DBG-GATE 三返回点诊断、ggml_probe_rope、
  TBO ROPE 13/13 的基线

## 重大模式判断（升级 #20）

第三次复现"单元探针全过 + 模型级乱码"：#20（RMS×ADDMUL）、rope（本批）。
每次让新 op 家族进入 CANNGE 合并子图，模型必坏而单元必过 → **嫌疑收敛到
共享集成层**（多 op 子图在模型规模的 staging/alloc/绑定交互），
而非任何单 op kernel。下批根因调查应直接打共享层：
- 模型级 DEBUG_IO 值转储对比"合并子图输入输出 vs 分解执行"
- 重点对象：input_staging 缓冲的跨 plan 复用、ggml-alloc 跨 split 别名
  与 GE 异步执行的重叠窗口、SET_ROWS 与 rope 输出在同一 cache 写链的时序

## 数字小结（本批后，门回退态 = 交付态）

| 配置 | pp t/s | tg t/s |
|---|---|---|
| CANNGE（GEMV，rope/FA 在 CPU） | 5.8 | 1.0 |
| -nkvo | 32.3 | 2.2 |

## 下批（B）

FA 上设备（FusedInferAttentionScore，JI 配方 note-1 §5）——图分割最大
单一来源（每层 rope×2 + FA×1 + 边界拷贝）；**但先做共享集成层根因**，
否则 FA 会第三次撞上同一堵墙（单元过/模型坏）。建议 B0：共享层根因
调查 → B1：FA 实现。

未 commit。探针新增 ggml_probe_rope。
