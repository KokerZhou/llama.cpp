# note-43: 批5 - 默认路径性能剖析与优化（rope 2D / 同步削减）

## 基线（note-36 之后，批4 修复之上）
- 默认 KV-on：冒烟 3.4-3.6 pp / 1.0 tg t/s（修前 0.6 tg）。
- DBG-TIME 分解（-n 60，17557 次 graph_compute ≈ 292 split/token）：in-backend steady ≈ 500ms/token，墙钟 ≈ 1000ms/token → **~440ms/token 在 sched（每 split 全同步 + 拷贝）**。
- 每子图 GE 执行仅 ~0.065ms/split（exec=1150ms/17557）。瓶颈从来不在 GE 执行本身。

## 已落地的修复（全部验证连贯）
1. **rope 2D 上门**（decode rope 输入 nd=2 被 `nd>=3` 误拒 → 每层 2 个 CPU split → 72 次 H2D/D2H+全同步/token）：
   - 门 `nd>=2`；build 加 nd==2 路径：[D,H] (d,h) 内存即 [1,H,1,D] BNSD 行主序，纯 reshape 进出。
   - tg 0.6→1.0 t/s。prefill 数值逐位一致。
2. **staged_any 同步精化**：只对"dense host 源"的 staging 做流同步（bounce 路径内部全是阻塞拷贝）；device 源的 kqv-permuted 拷贝不再触发每 split 全同步。bind.sync 4071ms→6.7ms。
3. **GEMV src1 门放开**：WO matmul 的 src1 是 permuted 非连续视图被拒 → CPU split；改为只要求 src0（权重）连续，src1 由 analyze staging 密化。该 CPU split 消除。
4. 诊断设施：DBG-TIME 增加 staged 字节/按名 top 统计；ggml-backend 的 [SPLIT] 打印（GGML_SCHED_DEBUG=1，含首节点名）；GGML_CANNGE_NO_WONCE 旋钮。

## 失败/回滚的实验（勿重走）
- **split 拷贝改 dense（ggml_dup_tensor）**：sched 的回退拷贝路径 ggml_backend_tensor_copy 有 same-layout 断言（ggml-backend.cpp:489）；给 cpy_tensor_async 加跨布局分支也救不了反向（CPU dst）路径。回滚。
- **supports_buft 声明支持 host buffer type**（意图：权重不再触发 need_new_split）：decode 乱码（prefill 正常）。根因未定位——与 RMSADD 实验的症状同类，见下。
- **GGML_CANNGE_RMSADD=1**（#20 门重测，2026-10-15）：仍乱码。首错点在 ffn_norm-0（正确，输出端口）之后的 FFN 块/attn_norm-1——**腐化不在 RMS 本体，而在合并后暴露的物化边界**（GE 工作区 vs ggml 地址，#41 同族）。门保持默认关闭。

## 剩余结构（decode，每层 ~8 split）
- 34 个 RMS_NORM 在 CPU（#20 门：src 是 ADD 残差）——最大单一 split 来源。解 #20 → 全合。
- ~7 个 CANNGE split/层 = **每个权重触发 need_new_split**（ggml-backend.cpp:1334：WEIGHTS-usage 且 src_backend_id(CPU) != cur 且 buffer 不支持）。supports_buft 实验本应解决但引入乱码——需先解合并暴露的物化 bug（与 #20 可能同根）。
- 因此下一步排序：先根因"合并后物化边界"类 bug（#20/RMSADD/weights-merge 三个实验同一症状），再合 split，最后事件化消 sched 同步。

## 数字速览
| 阶段 | tg t/s | 备注 |
|---|---|---|
| 批4 末 | 0.6 | rope 在 CPU |
| rope2d | 1.0 | |
| +同步精化 | 1.0（in-backend -9%） | 墙钟仍 1.0，瓶颈外移 |
| CANN graph 对照 | 71.7 | 目标 |
