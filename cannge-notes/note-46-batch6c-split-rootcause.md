# note-46: 批6c - split 结构的终极根因（pass-1 权重钉死）与 Mul_16 阻塞

## 终极根因（解释此前一切"配毒"现象）
sched split pass 1 的规则"operations with weights preferably run on the same backend as the weights"：**权重在 CPU host buffer → 每个含权重的 op 在 pass 1 就被钉到 CPU**。pass 2 的升级路径要求 `bufts[b] == bufts[cur]`（buffer type 相等）——CPU buft ≠ CANNGE buft → **永远升不到 CANNGE**。pass 3 只处理未分配节点。→ 所有 matmul 恒 CPU，RMS/ADD/SOFT_MAX 靠 pass-3 计数上 CANNGE。这就是 ~6 split/层、219 split/token 的来源，也是 -fa off / 默认路径共同的天花板。

## 正解路线（已验证到头）
`supports_buft` 声明支持 host buffer type → pass-1 的 backend_from_buffer 直接把权重 op 分给 CANNGE（最高优先级）→ split 塌缩。配套：权重 stage-once（按 leaf+非 INPUT+>64KB 判定，不依赖 usage 标志）。

**当前阻塞**：权重直供后 GE 编译报 E40021 `Mul_16 unify_broadcast_shapes` 失败（decode 图）。GEMV 的 Mul 输入对探针全绿，失败只在权重直供的图里出现——疑似某内部张量 holder 的 GE 形状（如 RMS 输出 {1,N} 2D）与输入侧 1D holder 混排广播被拒。下批入口：DEBUG_BUILD 打印每个 Mul 的两个输入 GE shape（需要在 ES 层拿 shape），或先把 gemv 的 es1 显式 Reshape 到与 es0 同秩。

## -fa off  decode 数值（并行的另一条线）
V 侧 batched GEMV 探针过、上下文错（i0 对 i1+ 错）；prefill 全对。若走 -fa off 路线还需解这个；若走权重直供路线，attention 的 kq/kqv matmul 也直接上 CANNGE，-fa off 可能不再需要。

## 批7a 进展（2026-10-15，未落地，已回滚）
supports_buft 重试连环雷（逐个解掉后仍有）：
1. E40021 Mul 广播预编译失败 = **lm head gemv**（151936×2560 的 f32 广播，GE mul kernel 拒超大广播）→ 门加 32M 元素上限。
2. E13025 输出端口尺寸不符 = GE 把 keep-dims 的 ReduceSum 折叠了（输出 26MB 而非 10KB）→ 改 keep=false（探针仍 10 绿）。
3. 剩余：prefill 图里一个 [2560,32]×[2560,4096] 的真·不兼容 Mul（非 gemv 路径的某个 build 在权重直供图里生成了错配形状，疑 RMS/ROPE 链）——未定位。
结论：路线正确但每步都是 GE 未文档化行为。保持 dormant，连同已修的 cap/keep=false 一起留在代码里（默认不触发）。

3. 剩余：prefill 图里一个 [2560,32]×[2560,4096] 的真·不兼容 Mul——**非确定性出现**（同命令+DEBUG_BUILD 通过、冒烟脚本失败；GE 并行预编译 taskID 每次不同）。嫌疑：**graph-build 的视图基址解析按指针序**（`registered` 是 `std::set<ggml_tensor*>`，resolve_view_base 在多个候选基址中按地址选"最高"）→ split/alloc 布局变化时视图可能解析到错误的基址 → 错形状 Mul。这是跨批次 latent bug（默认路径同受威胁，只是布局一直稳定）。下批：把基址解析改为确定性 + 校验候选唯一性。

## 批8a（2026-10-15）
1. **视图基址解析确定性化落地**（plan.cpp fallback 改为收集+排序：span→rank→name→address，多候选时 debug 打印）——这是真 latent bug 消除（默认路径同受威胁），已验证默认连贯。指针序理论对 Mul_2 **不成立**：确定性化后 Mul_2 仍以完全相同形状失败。
2. **GE 形状打印序查明**：GE 报错里 shape 按 GE 维序（外维在前）。Mul_2 input1 '2560,32' = ggml ne {32,2560} = **[32, 2560] = 32 token × 2560 的激活**，input2 = wq {2560,4096}。即某个 mul_mat 在 M=32 的图里被建成了逐元素 Mul。M=32 图来源未查明（prefill 15/decode 1 之外的多余签名）。下批：aclgrphDumpGraph 落地（本次 0 字节未成功）或给 MUL_MAT 非 gemv 分支加 DEBUG 形状打印抓现行。
3. 新增 GGML_CANNGE_DUMP_GE（aclgrphDumpGraph，当前输出 0 字节待调）。

3. 新增 GGML_CANNGE_DUMP_GE（aclgrphDumpGraph，当前输出 0 字节待调）。
4. **Mul_2 非确定性确认**：DEBUG_BUILD 打印开关本身能改变成败（两次复现/两次通过）→ GE 并行预编译（ParallelPreCompileOp）竞态或残余地址依赖。GEMV 分支的 Mul 形状按构造恒可广播 → rogue Mul 来自某处 ctx.get 拿到错误 holder（视图 fallback 的地址包含判定在分配器复用下可能把死指针匹配进错误候选——确定性化后=确定性选错）。
5. **批9a 定位**：计划签名确定性已验证（跨次运行 sig 一致）。失败图锁定：**sig=4083296402530835068，7 节点，kqv_out-0(CONT) → ffn_up-0(MUL_MAT)**（WO→残差→norm→FFN 链，M=32 来自 llama ubatch 填充）。binary() 自检未触发 → rogue Mul 只可能来自 RMS den-Mul 路径（链中唯一非 binary 的 Mul）。
6. **批9b GE IR dump（决定性）**：`GGML_CANNGE_DUMP_GE` 换 `ge::Graph::Dump(kReadable/kTxt→ostream)` 成功。失败图 IR：
   - Mul_2 = `Mul(Cast(wo{2560,4096}GE), Cast(kqv-0(permuted){32,128}GE))` — **WO gemv 的 Mul**。
   - **Data 口 "kqv-0 (permuted)" 注册形状 GE {32,128} = ggml {128,32} = kqv-0 本体（mul_mat(v,kq) 输出），不是 cont_2d 合并后的 {4096,M}！** WO 的 src1 本应是 cont 输出（CPU 钉住，offload_kqv=false），但实际进了 kqv-0 → K=128 vs 4096 → 不广播。
   - **两个 K-mismatch 自检都没触发**（gate 阶段 src 与 build 阶段 src 不一致：sched 在 supports_op 之后 rewrite 了 node->src）。错误图编号跨次变化（whack-a-mole），但结构同类。
   - 下批：在 analyze/plan 层比对"supports_op 时看到的 src 集合"与"cgraph 实际 src 集合"，或直接在 plan_analyze 里做 gemv K 校验（拿到 cgraph 时已 post-rewrite，校验结果与 build 一致 → refuse 该 op 走 CPU）。
7. **批10a 分析**：kqv-0=mul_mat(v,kq)={128,32,M}；llama `cont_2d(permute(kqv), 128*M, 32)` 得 {128M, 32}；WO=mul_mat(wo{4096,2560}, cont) 要求 128M==4096 → **M=32（llama ubatch 填充）**。dump 的 {32,128} 端口=M=1 解码图。
8. **批11a ROOT CAUSE FOUND & FIXED**：`GGML_OP_CONT` build 把输出**别名到输入 holder**（`ctx.tensors[node] = es0`）。M=1 时 `permute(0,2,1,3)` 把 {128,1,32} 变成**连续** {128,32} → CONT 门的 `ggml_is_contiguous(src0)` 放行 → 但 `cont_2d` 的输出形状是 {4096,1}！别名把 **src 的形状 {128,32}** 带给消费者 WO gemv → Mul({2560,4096}×{32,128}) → Mul_2 E40021。这解释了全部：gate 时 WO src1={4096,1}（节点 ne）✓、plan 时 cgraph src1=kqv_out{4096,1} ✓、只有 build 后的 holder 是 {128,32}；以及为何时有时无（只有 M=1/形状 squeeze 使 permute 连续的图触发；DEBUG 打印改变时序→不同 graph 先编译）。
   **修复**：CONT build 在 ne 不同是 `Reshape(es0, ge_shape(node))` 重标形状（字节不变）。验证：HOST_BUFT 下两次 40-token 运行 errs=0（编译全过）；默认路径连贯 3.7/1.0、探针 10 PASS。
9. **剩余（批11b 结论：supports_buft 方向证伪）**：HOST_BUFT 下 split 结构**零变化**（decode 仍 362 split、144 CPU MUL_MAT）——`supports_buft` 只影响 split-拷贝决策，**不改 pass-1 的权重-op 指派**；副作用是权重变成直供输入 → 每 split-plan 各传一份（315GB/40tok）→ 0.2 t/s。
10. **批12a：weight_buft_supported 链路与 256K 顶复查**：llama 权重 buft 选择走 `select_weight_buft→weight_buft_supported`（f32 测试激活探针）。已修：非 GEMV 门放开 f16×f32（build 双 cast f32）。
11. **批12b BREAKTHROUGH：256K 帽在 CANN 9.1 已不存在**——探针 `probes/ggml_probe_bigmm.cpp`（K=4096 N=2560 M=32 混合 dtype，10.5M 元素权重）**PASS 1.4e-6**。帽改为 env opt-out（`GGML_CANNGE_BIGMM=1` 恢复旧行为）。**效果立竿见影**：decode split 362→221、CPU MUL_MAT 181→37、**tg 1.1→1.7 t/s**、权重实际上设备（stage 仅 37MB）。
12. **批13a 收编与回滚**：CONT 非连续收编**证伪**（staging 字节序 ≠ cont 物化序 → v_cont 乱码，已回滚）；GQA BroadcastTo（kqv 的 8→32 heads）触发 GE 内部 Reshape_122 推断失败（BroadcastTo 可能内部降低为 Reshape），已回滚。批11a 的 cont_2d reshape 重标保留。稳态：decode 221 split、tg **1.8**、bigmm/gemv 探针全 PASS、默认连贯。剩余 CPU：36 kqv（GQA matmul）+ 36 v_cont（真转置物化，需 GE Transpose 或接受 CPU）+ 36 swiglu（跟随）+ lm head。
13. **批13b sched 事件化落地（收益被 CPU 计算挡住）**：CANNGE 实现 event_new/free/synchronize + backend event_record/event_wait（aclrtEvent）；sched 侧 `GGML_SCHED_EVENTS=1` 为 n_copies==1 也建事件（默认关零风险）。验证连贯 3.1/1.7。墙钟未降的原因：**当前墙钟大头是 37+36+36 个 CPU op 的实际计算**（每 token ~200ms+ CPU 算力），同步已非主犯——事件化收益要等 CPU op 收编后才兑现。次要残留同步：cpy_tensor_async 的前置全同步（sched 已用事件互锁时可免除）。

## 状态
- 默认路径：视图确定性化后连贯、探针 10 PASS、输出质量优。
- 默认基准：3.4/1.1（KV-on）、8.5/2.7（-nkvo）；距 CANN 71.7 tg 仍 ~27x，收敛到权重直供链。
