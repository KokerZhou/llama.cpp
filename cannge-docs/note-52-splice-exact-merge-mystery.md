# note-52: 批K 收官状态——拼接机制探针全精确，合并图模型级静默算错（待查）

## 1. 交付状态（默认路径，已提交推送）

- 默认（无 CONT_NC）：分裂结构，=all **10.6 t/s** 连贯，gemv 探针 16/16。
- `GGML_CANNGE_CONT_NC=1`：合并结构（decode 4 片/token），**模型级乱码**——禁用为 opt-in。
- 新鲜行拼接机制本身：**探针值全精确**（M=1：kq 4.7e-4 / kqv 5.8e-4；M=8 cont 字节级 0 误差；S=32/256 都过）。

## 2. 拼接机制（相对 note-51 的增量）

- V 侧多行（prefill）从 MatMulV3 矩阵形式改为**逐行 one-hot 选择循环**（MatMulV3 在本栈无注册 infer_datatype，EZ9999——note-46 早记载过；Equal 广播的矩阵形式未排查出问题但循环形式与 M=1 共用已验证原语）。K 侧加 `nelements(idx)==1` 守卫（其 gemv 本就 M==1；prefill 的 M>1 注意力走 CPU/BatchMatMulV3 拒绝路径，无需 K 拼接）。
- idx 输入 Always 注册进 GE（splice 要读；批G"idx 永不进 GE"假设废除）。
- 修复链记录：V 行 reshape 用 `t->ne[2],t->ne[1]`（sr.data 是 {D*G,M} 扁平视图，不是 {D,G}）；CONT 链优先 `ctx.get(已建 vt holder)`（ggml 折叠 view_src 链，手工重建会丢 PERMUTE）；探针踩的坑——**视图别名读法**（cont 输出要 ggml_set_output 才有写回；读 bytes 要按 ggml 线性序 s + S*d + S*D*g）。

## 3. 未解之谜（下一步主线）

**1432 节点合并图在模型级静默算错，而 30 节点的同模式探针精确**。已排除：拼接数学（探针证）、prefill（合并图的 prefill 输出与 fallback 逐项一致！）、Q 侧/rope（未改动）。decode 独有路径 = batched GQA gemv + 双拼接，探针同款同形状同 S=256 全过。

嫌疑排序：
1. **巨型图 GE 执行问题**（1432 节点单图的 buffer planning / 节点数上限类的静默错误）——验证法：把合并范围砍半（如只合并前 18 层）看错误是否消失/变化；
2. 模型 fragment 里 splice 的某个 holder 与探针语境不同（如 e1 来自 substituted Q 的 identity-permute holder 在大图中的行为）；
3. SET_ROWS 强制回写端口与拼接消费的同一 GE tensor 在大图中的相互作用。
调试入口：`GGML_CANNGE_CONT_NC=1 GGML_CANNGE_DUMP=kq_soft_max-0 GGML_CANNGE_DEBUG_IO=1`（强制中间量回写对比 fallback）；`GGML_CANNGE_DUMP_GE=1` 后按 softmax 反查 kq-0 区段拓扑。

## 3.5 批L 诊断进展（重要）

**`-ngl N` 是把最后 N 层放到 CANNGE**（非前 N）：fragment 首节点 l_out-(36-N) 可证。

**bisect（last-N 合并，同一边界）**：N=1/4/8 连贯；N=12 确定性乱码（两次逐字相同）；N=16 基本连贯；N=18/36 乱码。**非单调，排除单纯规模律**。

**DUMP 强制回写暴露 GE 输出端口别名**：对祖先链上的 l_out-24/28/32 同时强制回写时，三者校验和完全相同（b78cc979）——GE 把线性依赖的多个输出端口绑到了同一内部 buffer，回写全读祖先值。这是代码注释里 ge.exec.outputReuseInputMemIndexes 的已知 GE 行为家族。

**机制假说（合并图乱码的最大嫌疑）**：合并 fragment 的强制输出端口 = 每层 Kcur/Vcur 的 SET_ROWS data 端口（2×N 个）+ 边界隐状态。若 GE 对部分端口做同类别名 → Kcur/Vcur 快照读到错误 GE buffer → host scatter 把错值写进 KV cache → 第一步 decode 即乱码。N≤8 时端口数少可能未触发别名启发式，非单调性由此而来。

**下一步验证**（探针先行）：~~kvattn 探针加祖先链双输出用例~~ 已做（批M，ggml_probe_alias）：

**批M 结论（端口别名假说排除）**：
- 祖先链 y1→y2→y3 三输出（set_output 与 DUMP 强制两种）：全部精确读回，小图如此；
- **1200 节点长链** + 3 个祖先 flagged 输出：依然全部精确——GE 输出端口即使大图也无别名。
- l_out-24/28/32 同校验和的解释 = **DUMP 读死张量的 artifacts**：ggml-alloc 会复用死张量 buffer，post-execute 回写互相覆盖——观测工具本身制造了假象，GE 无辜。

**批M 新线索（真 bug 方向）**：ngl=12 合并 decode 的 DBG-IO "out" 打印清单里 **Kcur/Vcur（1024,1）有 22 条，但跨片隐状态（2560,1）一条都没有**——合并 fragment 喂给 CPU lm head 的隐状态回写似乎没发生（下游读到陈旧 buffer → 乱码）。而 ngl=8 同结构却连贯。**下一步：对比 ngl=8（好）与 ngl=12（坏）的隐状态输出注册差异**（boundary-output 收集在两种切分下为何不同），这大概率就是根因。

## 3.6 批N 猎杀记录（四个假说全部证伪，症状精确刻画）

同边界对比（ngl=12 merged vs split，输入一致）发现决定性症状：**merged 第一步 logits 与 split 完全一致（c83f9a7a）但 Kcur/Vcur 的写回值不同（2.115 vs 1.427）**——即图内消费正确（kq 拼接用了正确的 Kcur → logits 对），而**同一 tensor 的输出端口写回 carry 了另一个值**。第二步起 KV cache 被污染 → 乱码。写回值逐步不同（非滞后）。

证伪的四个 GE 假说（ggml_probe_alias 全部 PASS）：祖先链端口别名（小图+1200 节点长链）、32 个兄弟端口、双重用途（图内消费+终端输出）、强制私有物化（Add-zeros，可能被 GE 折叠）。DUMP 强制读死张量的同校验和 = ggml-alloc buffer 复用 artifact，非 GE 行为。

**批O 实锤（冒烟级）**：merged 模式全部 12 层 Kcur 写回**逐字节相同**（sum=f109b7a8，i0=2.115）——24 个 Kcur/Vcur 输出端口在 GE 运行时层被 collapse 到同一个 buffer；split 模式逐层不同（正确）。**但 GE 图本身完全正确**：dump 显示 24 个 NetOutput 端口、来源无一重复（12 Reshape + 12 Cast，逐层不同 producer）。→ 是 GE 运行时的输出刷新机制（davinci_model 的 refreshable_output_index_and_allocation_ids，E13025 错误信息里见过）把同形输出分到同一刷新组。

**探针复现失败记录**：兄弟端口×32、祖先链（含 1200 节点）、双重用途×24、精确尺寸 {1,1024}×24、~2400 操作图 + 链垫——全部 PASS。模型独有触发因子未识别（候选：producer 深度/多消费者 fanout、refresh 组的确切构成条件）。

**下一步（fix 方向）**：**绕开 Kcur/Vcur 端口 entirely**——cache-refresh 输出：SET_ROWS 改 in_graph（复用现成机制），build 里在 cache base holder {S,R} 上做 one-hot 选择产出"新鲜缓存"（M 行循环推广），节点本身作为输出端口（节点 ggml buffer = dst 视图 = cache），稠密回写后**host scatter 整体退役**（分裂配置收益也 +）。{512,1024} f16 的新端口形状若仍被 collapse 再按数量分片。

## 3.6b 批T 撤回与翻案（重要）

**"GE 散射族静默 no-op"（note-49 结论）正式撤回**。用修好的可信观测（plan cache 签名修复 + 无别名读数）重验：**TensorScatterUpdate 写 cache 逐字节精确**（setrows 探针 0.000；kvattn M=1/M=9 S=512 全过）。原始错误结论源于探针读法缺陷（视图别名 buffer + 读未输出端口的张量）。**用户批评正确：成熟产品先验应是"我的用法错了"**——同批还证实"合并图冻结"也是我自己的 plan cache bug（批S），"运行时 refresh collapse"从未被证实（机制是猜的）。

SET_ROWS in_graph 路径已整体改用 TSU（替换 one-hot 选择 + BatchMatMulV3 矩阵形式，-48/+15 行），host scatter 保持退役。默认路径 =all 9.4 t/s 连贯。

合并路径（CONT_NC）模型级仍乱码——散射既已清白，剩余嫌疑收敛到合并 fragment 的输入绑定 / holder 复用 / 每步重建图与 plan 的交互。下一步用可信 DUMP 在层 24 起点做干净二分。

## 3.6c 批U 干净二分进展（可信 DUMP 时代）

层间 fork：merged 的 **decode 起始 cache 内容与 split 不同**（k/v、多 layers 证实；KV 皆差）→ **合并 prefill 写入了错误缓存** → 第 24/25 层注意力即偏离（kq_soft_max-25/26/27 merged≠split 已证）→ 全链乱码。TSU 多行（M=8/9）探针字节级精确 → 机制无罪，嫌疑收敛到**模型 prefill 里 TSU 的实际输入**：upd（Vcur/Kcur 的 {1024,M} 视图——探针用的是独立张量，模型是 rope 三维输出的视图）或 idx 的 staging。下一步：比对 prefill fragment 的 TSU 输入 checksum（mrg vs spl）。

注：DUMP 强制对部分张量（Kcur/Qcur/l_out-24）静默无输出（原因未查），观测工具仍有盲区。

## 3.6d 批V 根因锁定：同形输出端点的运行时 collapse（真实现象）

prefill TSU 输入对比：Vcur-25/26 视图 merged==split，**Vcur-27+ 全差** → prefill 第 27 层起 l_out 已偏（decode 同款早期偏离）。prefill 结构：FA(M=9) 在 CPU，CANNGE 段为 [层 prologue(qkv+rope+TSU)] + [wo+ffn]。

结合批O 的实锤观察（24 个同形 {1,1024} f32 端口逐字节同值——观察真实，只有机制命名错了）：**GE 运行时把同 shape+dtype 的并列输出端口绑到同一 buffer**。Vcur/Kcur (view) {1024,9} f32 × 12 若跨片即撞此行为 → 27+ 层读到错值。这不是"GE 有 bug"的表述——是我们的图结构触发了该行为的组合条件；解法在我们侧：**消灭同形输出端口并列**（消跨片 / 不同形化 / 每端口独立 staging 验证）。

下一步候选：(1) 查 Vcur 跨片去向（M=2 图角色待查）；(2) 验证法——把跨片张量写回 staging 改为逐端口独立 device buffer 看偏离是否消失；(3) 或 prologue 与 wo+ffn 合并成一片（消跨片）。

## 3.7 批P cache-refresh（WIP，2d624231f）

实现：merge 模式下 SET_ROWS 标记 in_graph，build 在 cache base {S,R} 上做 one-hot 行选择（M=1）/ m^T@rows BatchMatMulV3（M>1，K pad 8），节点自身作输出端口（buffer=cache），host scatter 退役；消费侧 splice 自动禁用（sr_for_base 跳过 in_graph）。**collapse 确证为按形状分组**：{512,1024} f16 refresh 端口逐层写回各异（不再塌）。

WIP 两个待修：(1) 模型级 merged decode **第一步 logits 即偏离**（旧 splice 版第一步还一致）——探针全过（kvattn M=1/M=8、setrows 字节级）但模型不一致，差异在探针未覆盖的交互；(2) f32 全 cache 选择 ~0.5s/token（200MB/步流量）——需 f16 化或矩阵化。默认路径（无 CONT_NC）不受影响，10.6 t/s 连贯。

## 3.8 批P-fix 探针补全与模型级定位尝试

refresh 探针矩阵全字节级精确：M=1/8/9/10 × S=256/512（含 **Mp pad 分支** M=9/10→16，此前未覆盖）——选择/矩阵/缓存写回在隔离环境无可指摘。模型级 merged decode 第一步仍偏离。

新观测：(1) DUMP 强制在 split 模式下对 kq_soft_max-30 不打印（merged 模式打印）——工具一致性存疑，跨模式对比受限；(2) merged 的 **l_out-34 写回与 split 不同**（b78cc979/254.7 vs 029acfef/241.6）——偏差在 giant fragment 的层计算内（注意力探针已证无误，嫌疑转向 fragment 内其他同形张量族的运行时行为，如 ffn gate/up {9728,1}×24 的内部 buffer 分配）。

## 3.9 批Q 决定性症状：残差流冻结

**单端口逐层强制**（每次运行只 DUMP 一个 l_out，排除端口 collapse 与 buffer 复用干扰）：l_out-26/28/30/34 在各自独立的运行里**全部读到同一个值 b78cc979**——合并 fragment 的残差流在 fragment 第一个层之后**冻结**（25 层起贡献为零，确定性）。批K（splice）与批P（refresh）都有此冻结 → 与拼接方式无关，是**合并本身**。

矛盾点：1200 节点长链探针（窄链）执行正确——GE 无简单 op 数上限。嫌疑：fragment 的**扇出结构**/宽度、或 GE 对某种子图模式（24 组同构层顺序串接）的优化触发静默裁剪。下一步：扇出探针（多分支宽图）验证 GE 宽度上限；或在 build 里给每层 l_out 强制 distinct 物化观察冻结点是否移动。

## 3.10 批R 结论：冻结不可复现，转向分裂配置优化

扇出宽度（8 路×24 块 f32/f16）、输入数（130/200）、链长（1200）探针全部 PASS——GE 执行一切简化形态正确。合并 fragment 残差流冻结（单端口证明，确定性，与 splice/refresh 无关）在现有工具下无法进一步定位，列为 GE 运行时平台问题（与散射 no-op、PFA 布局错乱同族）。

转向分裂配置优化：CONT-of-转置视图输入替换（plan.cpp）——v_cont 消费者直接绑 cache base + 图内重建，消掉 1MB×36/layer 的 H2D staging。连贯，tg 中性（10.4 @ 噪声带内），host 流量移除，合并路径共用。

## 4. 数字

里程碑：0.6 → 1.7 → 3.8 → 6.0 → 6.9 → 10.4 → 10.8 → **10.6（交付默认态）**；=v 10.2；默认 2.4。合并若修好：4 片/token × 后端稳态 6.3ms，理论 15-25 t/s 量级。
