# note-44: 批5b - 跨 split 物化缺失根因（#20  closure）+ RMS 全上设备

## 根因（三个实验同症状的同一个 bug）

**GE 图后端只在输出端口（io.outputs）把值写回 ggml 地址；图内中间张量的 ggml 缓冲从不被写入。**

analyze 的边界输出判定是 `n_consumers[node] == 0 || FLAG_OUTPUT`，而 **n_consumers 只在当前 split 的 cgraph 内统计**。一个既有图内消费者、又被后续 split 消费的张量（如合并后 ADD→RMS 里的 ffn_inp）：
- 图内被 RMS 消费 → 不算边界 → 无 GE 输出端口 → ggml 缓冲保持陈旧字节；
- 后续 split 的 sched 从该 ggml 缓冲拷出陈旧值 → 下游全错。

split 世界此前"恰好正确"：ADD 单独成 split 时图内无消费者 → 成为边界被物化。任何合并（RMSADD 门、weights-merge）都会制造双消费者张量 → 乱码。批4 的 IFA cache_v 分歧同族。

实锤链（bis0/bis1 对照）：rmsadd 下 ffn_inp-0 图内值正确，但其 ggml 缓冲读回 = 952eb641 = 上游客体 node_32 的字节（allocator 复用死槽），下游 "in ffn_inp-0" 读到它 → attn_norm-1 错。

## 修复（ggml-backend.cpp split pass）

1. 拷贝路径（跨后端 split 输入）：创建 sched 拷贝处 `ggml_set_output(src)`（且每处 rewrite 都执行，不受拷贝缓存 if 块限制）。
2. 同后端隐式跨界（need_new_split 权重边界，无拷贝无 rewrite）：split 定界后扫一遍——记录每个节点的 producer split，凡 src 的 producer split != 当前 split 且非 leaf → `ggml_set_output(src)`。

analyze 侧无需改动（FLAG_OUTPUT 已参与边界判定）。

## 结果
- **#20（RMS×ADDMUL 腐化）closure**：门删除，`GGML_CANNGE_NO_RMS` 旋钮保留。RMSADD=1 与默认输出均连贯。
- decode split 292 → 219/token；34 个 CPU RMS split 全消；tg 1.0 → 1.1 t/s，pp 3.7。
- weights-merge（supports_buft host）重试：**连贯了**（物化修复生效）但无 split 收益且 stage-once 未识别权重（usage 标志没设）→ 每 token 重传 800MB → 0.3 t/s，回滚。split 并非权重驱动。

## 剩余（批5c）
- stage bounce 24s/60tok（≈400ms/token）仍是 in-backend 首项：`CANNGE0#kqv-N (permuted)#0` 设备转置每 split 走 D2H+host 逐元素 scatter+H2D。方向：分块转置提速 / 消 split 后自然消失。
- 墙钟 ~55% 在 sched（每 split 全同步 + 拷贝），事件化（CANNGE 未实现 sched event 接口）或继续减 split。
- ~6 split/层的来源待查（非权重、非 RMS——疑为 CPU fallback 的 FA/attention 链与 M 对齐拒绝的 matmul 边界）。
