# note-53: 根因与全面平反——is_weight staging 误判（子 Agent 独立诊断）

## 根因（一句话）

`ggml-cannge.cpp` 输入 staging 的 **is_weight 启发式**（op==NONE、无 INPUT 标志、>64KB → plan 内只暂存一次）漏了"必须 host 驻留"这一条件：调度器用 `ggml_dup_tensor_layout` 创建的 **device 驻留、非稠密 split 输入拷贝**（无 INPUT 标志）被误判为不可变权重 → plan 首次执行（层 0）暂存后，**后续 35 层复用同一 plan 时全部读到层 0 的 staging 字节**。

## 因果链

1. CONT_NC 放宽 CONT 门槛 → prefill 的 `kqv_out = CONT(kqv permuted)` 上 GE 成片；
2. 该片唯一输入是调度器的 split 拷贝（保留转置步长 → 非稠密，n_copies==1 时不带 INPUT 标志）；
3. 非稠密 → staging；is_weight 误判 → 只暂存一次；
4. 同签名 plan 服务 36 层 → 层 1+ 的 kqv_out 全部 = 层 0 的值（36 层输出校验和全同，输入各异——实锤）；
5. kqv 错 → wo 错 → l_out 错 → prefill 把错误 K/V 写进 cache（层 0/1 对、层 2 起错）→ decode 从污染 cache 读 → 乱码。

M=2 pass 不受影响（32KB < 64KB 阈值每次重暂存）；**1432 节点合并 decode 图本身计算正确——被污染 cache 冤枉**。

## 修复

is_weight 增加 `buffer != nullptr && ggml_backend_buffer_is_host(buffer)`——只有 host 驻留叶子（CPU 权重）才只暂存一次；device 驻留 split 拷贝每次执行重暂存（+12/-2 行）。另加 DEBUG_IO 绑定地址打印（旧 in 校验和读 t->data 而非 GE 实际绑定的 staging 地址，制造了"输入正确"的假象）。

## 验证

- 子 Agent：两种模式 prefill 全部 286 个同名输出**逐位相同**；decode 起点 72 个 cache 全同；pass2 + 9 步 decode logits 全同；文本一致连贯。
- 本批复核：合并配置（CONT_NC）冒烟**连贯**（修复前乱码）；gemv 16/16；setrows 0.000；restage 防回归探针（plan 复用 + staged 输入，此前探针全盲的类别）。
- 修复者：独立子 Agent（最小提示词、零理论污染）；本会话此前六批（K~W）在污染观测的误导下追错了方向。

## 全面平反（对本会话此前"GE 平台 bug"清单的最终清账）

| 曾归因 GE 的"bug" | 真相 |
|---|---|
| GE 散射族静默 no-op（note-49） | 探针读法假象；可信重验 TSU 精确（批T） |
| 合并图"残差流冻结"（批K~Q 追六批） | plan cache 签名不含强制输出 → 陈旧 staging 读取（批S，我的 bug）+ 本 is_weight staging 污染的同构症状 |
| "GE 运行时同形端口 collapse"（批O/V） | 观测同被 is_weight 污染（36 层 kqv 同值实为 staging 冻结，非端口 collapse） |
| PFA 布局错乱（note-42，更早） | 未重验，列入待办 |

**用户判断正确：GE 无罪，全部是我侧封装层的 bug。** 方法论教训：污染观测 + 锚定理论 = 六批弯路；独立视角 + 最小提示词一轮命中。

## 下一步（合并路径已通，转性能）

## 批X 性能首测（arena + DEBUG_SIG）

合并 -n 100 = 2.5 t/s，但 DBG-TIME 显示稳态 ~141ms/token（≈7 t/s）+ 编译 ~57s 摊销主导短基准。bind 占稳态 75%：arena 打包（每执行 1 次 H2D 替代 ~150 次小拷贝）后 bind 5.7s→4.35s。DEBUG_SIG 暴露关键矛盾：**运行时每 decode 步收到 ~28 个 3 节点小图（norm-N ×2），而调度器报告 1432 节点合并片**——split->graph 在运行时远小于 SCHED_DEBUG 所示，是下一个合并杠杆（查 fork 的 split 打印与运行时状态为何不一致）。


- 合并 decode 正确但 ~0.2 t/s：瓶颈 = f32 全 cache refresh 回写流量（批P 设计）。优化方向：回写改 in-place 通道 / f16 / 只写变化行（GE outputReuse 声明）。
- 优化后合并路径（4 片/token）预期远超分裂路径 9.4——性能战场正式转到单图执行效率。

## 批Y/Z：子 Agent 二轮调查 + 直连 TSU

子 Agent（二）证伪"每 token 250 小图"前提：那些是**一次性启动成本**（server 探针 common_context_can_seq_rm 的 M=2 decode + prompt M=17 预填，M 不对齐 16 → matmul 落 CPU → 碎块；签名跨层共享所以编译 ~8s/块一次性）。真实 per-token：1432 大图设备侧执行 ~350ms（**~2400 GE op × 逐 op 地板**，有效带宽仅 23GB/s = 910B3 的 1.4%）+ bind ~30ms。

结论：**算子流 GE 的逐 op 下发地板是合并路径的结构墙**——JI 的答案是手工融合核（整层一个 kernel），ES 算子流做不到。合并路径当前 2.7 t/s（正确），分裂路径 9.1 t/s 为交付配置。已落地：直连单 op TSU（省 144 op/token）、arena 打包（bind -25%）、DEBUG_SIG。

遗留杠杆：启动探针 M 对齐（消除 250 个一次性小图编译，~省 30-50s 启动）、大图 op 数削减（每层 ~30 op → 理论上限 ~12 t/s）、lm head 尾片并入大图。
