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

**下一步验证**（探针先行）：kvattn 探针加祖先链双输出用例（out2=f(out1)，两者都 set_output）→ 若读回相同值 = GE 输出端口别名实锤。若实锤，workaround 方向：SET_ROWS data 不走高强制端口（改为在 build 里把 scatter 快照源换成 device 侧，或每层独立小图执行散射）。

## 4. 数字

里程碑：0.6 → 1.7 → 3.8 → 6.0 → 6.9 → 10.4 → 10.8 → **10.6（交付默认态）**；=v 10.2；默认 2.4。合并若修好：4 片/token × 后端稳态 6.3ms，理论 15-25 t/s 量级。
