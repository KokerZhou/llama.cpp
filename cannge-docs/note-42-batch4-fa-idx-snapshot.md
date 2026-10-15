# note-42: 批4 - SET_ROWS 快照修复 + IFA decode 数值解剖（FA=on 乱码根因）

## 1. 段错误根因（已修复）：SET_ROWS 的 sched 拷贝生命周期

**症状**：FA=on 时 layer 35 的 set_rows 散射读 idx 设备拷贝（`CANNGE0#leaf_10#0` @ 0x...c03800）得垃圾槽位（`sl0=-4.76e18`），l0-34 全部正确（sl0=15），随后 aclrtMemcpyAsync 野指针段错误。

**根因链**（/tmp/fa7.log 全量过滤日志实证）：
1. n_copies=1（llama-cli parallel=false）→ ggml-backend.cpp:1406 的 split-input 拷贝**不标** INPUT/OUTPUT。
2. 该拷贝对象每 (src,dst-backend) 只建一个，跨 split 共享；每个 split 边界 sched 重拷（compute_splits 1669-1789）。
3. 但 ggml-alloc 眼里该拷贝的 liveness 到**最后一个 SET_ROWS 节点**为止（gallocr 是全局单遍分配，concat graph）。
4. l35 的 kqv_out-35 输出（16KB @ 0x...c00000）的分配区间覆盖了 idx 拷贝（0x...c03800，该处 kqv_out 与 attn_norm 复用同一片）；kqv_out 是 GE 工作区重叠的 staged 输出 → **staged copy-back 在散射前把 idx 8 字节覆写**。
5. 我们后端在**图执行后**消费 idx/data —— 对 ggml-alloc 而言这是"死后读取"。

l0-34 正常的原因：idx liveness 贯穿全图，只有 l35 之后无消费者，恰好 l35 的 tensor 复用其地址。

**修复**（ggml-cannge.cpp，后端局部、不动共享核心）：
- GE execute **前**：idx 快照（host memcpy 或 sync+D2H，几字节到百字节）。
- GE execute **后、staged copy-back 前**：非 staged 的 data 快照（sync+D2H）。
- copy-back **后**：staged 输出的 data 快照（copy-back 是 async，需先 sync）。
- 散射循环全部改读快照；es_d!=es_t 的 cast 路径不再需要逐行 stage_row（快照已是 host）。

代价：每个含 set_rows 的 split 1-2 次 aclrtSynchronizeStream + KB 级 D2H/H2D。批5 做事件化/批量化。

**验证**：FA=on 冒烟不再段错误、正常出 token（之前第三个 decode 图必崩）。

## 2. FA=on 乱码根因：PFA 不认 BSND

修复段错误后输出乱码。DEBUG_IO 对照（faoff/faon 同 prompt）→ 第一层 `kqv_out-0` 输出**全零** → 残差恒等 → 全链路乱码。

探针矩阵（probes/ggml_probe_fa.cpp，模式 0/1/4/6/7/8/9/11）：
- **IFA decode（S_q=1）bit-exact（max_rel=0.000e+00）**，任意 KV 长、q f32/f16。
- PFA（S_q>1）全挂：MHA/GQA、有面具/无面具、S=2..16、D=64/128。
- mode 6/9（N=1）全过 → n 轴退化时一切正常。
- **mode 11 解剖**（k=0 → 均匀注意力，v[d,t,g]=d+10t+100g）：输出 (s,h) 块 = v[t=h%2, g=2+(h//4)%2] 单行 —— 正是 **BNSD 误读**（把 (s,n,d) 内存按 n'*S*D+s'*D+d 解读 q/k/v）。
- 旁证：cfg2（BNSD+in/out perm）与默认逐位相同（permute 被 GE 与上游 ggml permute view 折叠）；in-only 时 m7-S8 曾"假 PASS"（形状巧合+优化器折叠，不可复现于其他 S）。

**结论：GE PromptFlashAttention 的 input_layout="BSND" 被静默忽略（按 BNSD 读）**。IFA 真认 BSND（256 KV decode 精确）。BSND 大概率是 IFA 专属布局。IncreFlashAttention 跑 S_q>1 也半数元素错（mask 语义未明），不可用作 prefill 替代。

## 3. 决策：FA 门收窄到 decode（s_q==1）

supports_op 的 FLASH_ATTN_EXT 门加 `src0->ne[1] == 1`：prefill 回落 kq_soft_max 路径（FA=off 已验证连贯）。生成场景 tg 是大头，IFA decode 保留核心价值（消掉 K/V cache 每 token 650ms D2H）。

PFA 正确接法（后续工作）：物理转置输入到 (n,s,d)（防折叠：用非 reshape-permute-reshape 形式，如 Slice+Concat）+ layout="BNSD" + 输出 [B,N,S,D] 转回；或对 GE 提 BSND 支持问题单。

## 4. IFA decode 继续深挖（本轮结论：暂停 FA，转向主瓶颈）

乱码未解。逐层 fletcher 对照（off2/on2 同 prompt 同种子）：
- prefill + decode 前 2 步全链一致；第 3 步 kqv_out-0 分歧（输入 Q/K/V 全一致）。
- **第 1 步即发现 cache_v_l1..l35 与 FA=off 不一致**（cache_k 一致）——V 行内容在 FA=on 下被写坏/写晚。
- 探针：IFA+ASL 在 KV<=8 全 KEEP 精确；KV>=16（16/32/64/128/256，MHA/GQA、有/无 mask、任意 ASL）全错，且输出不是任何前缀长度的注意力——s_kv>=16 走了另一条错误 kernel 路径。
- 模型 KV=256 却在前 2 步精确：模型 cache 视图经 CPU ping-pong 多 split（rope/set_rows/attention 不同 split，fa7 日志 492-497 split/层），时序与探针单 split 不同；第 3 步分歧的触发条件未定位（怀疑当前行可见性：IFA 与同 split 的 set_rows 散射的先后）。
- 修复尝试 GGML_CANNGE_FA_ASL（静态常量验证 ASL 语义正确，EZ3002 要求 DT_INT64 [B]）→ 已升级为运行时端口：plan_io.fa_mask 记录 mask 输入，plan 持有 8B 设备缓冲，每 execute 从 mask 可见前缀计数 H2D（f16 位模式比较符号坑：负数 fp16 位序相反，必须 ggml_fp16_to_fp32 后比）。探针 m12（KV=8 任意 KEEP）PASS。

**决定：FA=on 保持 opt-in（默认关）暂停**。核心障碍是 GE IFA/PFA 的布局与 kernel 路径黑盒，继续烧时间性价比低。当前默认路径（FA=off）连贯：冒烟 3.4 pp / 1.0 tg t/s（CANN graph 7300/71.7），主瓶颈仍是 note-37 的 650ms/token KV D2H 序列化 + 230ms 胶水——批5 转向那里。

## 遗留
- ggml-backend.cpp 的 DBG-SCHED 打印（env GGML_CANNGE_DEBUG_SR）保留。
- graph-build 的 FA knobs：FA_LAYOUT/FA_INPERM/FA_OUTPERM/FA_PRE/FA_NEXT/IFA_ALL/FA_ASL + DBG-FA 形状打印（调试用）。
- IFA 输入绑定正确性待查项：s_kv>=16 kernel 路径、IFA 与同 split set_rows 散射的当前行可见性。
- 同步批量化（批5）：每 split 1-2 sync 的开销待事件化。
