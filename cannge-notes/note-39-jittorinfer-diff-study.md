# note-39: JittorInfer 对照研究——差异在图切法，不在 GE（独立 agent 产出，2026-10-14）

回答用户之问："图比 CPU 慢，但 JittorInfer 明明加速了，差异到底在哪"。
由独立 explore agent 完成（源码行级核实 + 本笔记体系证据 + 上游官宣），
父 agent 未参与其检索。

## 一句话结论

GE 编译从来不是瓶颈（缓存命中小图 e2e 0.044ms，note-38 实测）。差距全在
**图怎么切**：JittorInfer 把**整模型编成一张常驻 GE 图**，每 decode 步
= 1 次下发 + 1 次同步，host 近乎零参与；CANNGE 被 supports_op 缺口逼成
**每 token 100-150 个 1-6 节点碎片**，每边界 H2D/D2H+全流排空，
KV-on 时 CPU 注意力每 token 拉回 36MB cache。GE 的融合/tiling/内存规划
优势在碎片粒度下完全无法兑现，胶水反超计算。

## JittorInfer 执行模型（证据分级）

| 机制 | 内容 | 证据 |
|---|---|---|
| 图粒度 | 整模型 1 张持久图；decode 用 worst-case 定形 cgraph | 源码实证 note-2 §1 |
| 编译 | 加载期 warm-up 一次 AddGraph→Compile→Load | note-2 §2 |
| 绑定 | 构图时钉 ggml->data 进 gert::Tensor，warm-up 后永不重绑 | note-1 §2 |
| 每步更新 | op_params 就地写 + 几个 KB 级输入张量 set_tensor_async | note-2 §3 |
| 执行 | 每步 1 次 ExecuteGraphWithStreamAsync + 1 次 sync | note-1 §3 |
| KV/注意力 | KV 常驻 device 兼图输入，图内 ScatterUpdate；FA=FusedInferAttentionScore（全段 KV 输入 + length_kv 张量解耦形状） | note-2 §4 |
| host 参与 | logits 不上 CPU：device 侧 top-k 采样只回传 (value,index) | note-2 §6 + 官宣 |
| 算子覆盖 | 24 图内 handler：BatchMatMulV2 raw-op、RopeExtCustomV2 或 5D 纯原语、FusedInferAttentionScore、KV scatter、MoE、TP all-reduce；**GE 路径不开 CPU fallback，不支持即 GGML_ABORT** | note-1/2/4 |
| 效果 | DeepSeek-V2-Lite 910 单卡 73.3 t/s vs vLLM-Ascend 12.2（~500%） | 上游 README/清华官宣 |

## CANNGE 差距清单（源码实证）

图粒度（碎片 vs 整图）/ 算子覆盖（FA 未实现、ROPE 多门、matmul 对齐+256K
门、RMS 生产者门、get_rows 单行门）/ 绑定（每段 staging+strided+bounce）
/ host 参与（每边界全同步）/ 同步（未实现 .event_new → sched 兜底全排空）
/ KV（全 cache D2H 650ms/token）。每 token GE 执行本身仅 ~16ms，tg 实测
1.1s → **>95% 时间在胶水**，设备大部分时间在等（HdcPcieRecvPeek/mmSleep）。

## CANNGE 必须改（按优先级，来自 agent）

1. **注意力链上设备（根治）**：FA→FusedInferAttentionScore（JI 配方：
   input_layout=BSND、sparse_mode=1、length_kv 走输入张量，note-1 §5）+
   ROPE 放开（默认参数子集已在门内，缺省走 5D reshape+transpose 纯原语，
   note-1 §5 双实现对照）。FA/rope 上设备 → 每层 CPU 注意力段消失，
   段数 100+→个位数，KV 全量 D2H 同蒸发
2. **整图接管**：SET 图内 ScatterUpdate、MUL_MAT 换 BatchMatMulV2 raw-op
   破 256K 上限（note-27 #16 已立案），supports_op 吃下整图 → 调度器不再切
3. **消灭逐次全同步**：实现 .event_new/record/wait；跨 split 拷贝合并一次同步
4. **KV 设备化**：常驻 device + 图内 scatter + FA 按 length 读活跃区
5. 去调试残留再测（DEBUG_IO fletcher/DEBUG_TIMING 桩自身开销）
6. 短线：V 视图 4B 成对拷贝、bounce 偏移按签名缓存

## 核心判断：通用后端约束下能搬什么

- **直接搬**：算子映射配方（FA/V2 raw-op/KV scatter）、签名 plan 缓存、
  每 call 按当前地址重绑、加载期 warm-up 编译
- **必须重设计**：JI 靠 fork 级耦合（私有 cgraph flags、持久图、abort、
  device 采样）换"整图一次下发"；CANNGE 通用后端不能假设指针复用/形状
  固定 → 只能"签名 plan 缓存 + 动态形状每类一次编译"。**没有中间态**：
  每有一个 op 不支持，图就碎一次。-nkvo 后仍只有 2.2 t/s（残余胶水
  ~230ms）正是覆盖不全的代价
- **归因**：JI 加速来自全栈耦合换来的"一次下发"，GE 只是让这次下发值得；
  CANNGE 用对了 GE，但没收回三个前提——**完整算子覆盖、稳定缓冲、
  O(1) 次同步**

## 证据出处

JI 侧：note-1/2/4（310 轮对本地 checkout cc18685/55d1c1e 的行级核实）+
上游 README/清华官宣页；CANNGE 侧：ggml-cannge.cpp（graph_compute:518、
supports_op:1110、strided_copy:410、get/cpy:286-352、SET_ROWS:844）、
plan.cpp（analyze:184、signature:401）、graph-build.cpp（GEMV:395-441）、
ggml-backend.cpp:1643-1789、note-36/37/38 实测数字。
