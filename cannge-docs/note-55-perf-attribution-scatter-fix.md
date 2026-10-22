# note-55: 性能归因实测定案——CPU 岛与逐 kernel 下发；scatter 修复、磁盘编译缓存（2026-10-22）

基于 codex 重构树（cannge 分支 9224022f5 起）。本轮提交：c5b849a30（no-op cast 跳过）→
396357f73（空张量硬化批，CRLF 污染清洗）→ 78883ef08（scatter 写回修复）→ b12874d61（GE 磁盘缓存）→
a28222341（Const 去重 + BMM gemv 选项 + emit 版本号）。

## 1. 分裂路径 SET_ROWS 图内散射修复（头等正确性修复）

**现象**：`GGML_CANNGE_SCATTER=1` 开图内 TSU 散射后，固定种子文本变乱码。

**根因**（DEBUG_IO 校验和对拍定位）：set_rows 节点是 cache 的致密视图
（offset=0、同形），boundary 输出注册规则"致密视图别名输入 → 无需输出端口（字节已在缓冲区）"
把它跳过了——但图内 TSU 的新字节只存在于 GE 工作区，从未写回 cache。于是 decode 每 token
的 K/V 写入全部丢失，attention 逐 token 读陈旧 cache。合并路径（CONT_NC）无恙是因为
collect_outputs 有独立分支显式把 set_rows 推为输出。

**修复**（plan.cpp，+8/-1）：输出注册门从 `allow_noncontiguous_cont()` 扩到
`|| options.scatter`，且对 index_graph 预标记但 cache base 未绑定的情形回退 host scatter。
验证：固定种子 A/B 与基线逐字一致；cannge 单元测试 6/6 过。

**收益**：post（host 散射）18.5ms/token → ~1ms/token；索引快照 D2H ~3ms/token → 0。
分裂路径稳态后端 33ms → ~8.7ms/token。

## 2. GE 图编译磁盘缓存（消灭每次进程 23s 编译）

ge.cpp：session 选项 `ge.graph_compiler_cache_dir`（默认 `~/.cache/cannge/ge`，
`GGML_CANNGE_GE_CACHE_DIR` 可改、置空可关）；AddGraphWithCopy/LoadGraph 传
`ge.graph_key` = `cannge_v<EMIT>_<sig-hash>`。签名本就地址无关（形状/步长/op/端口序/选项全覆盖）。

实测：冷 22.7s → 热 7.5s（21 图全命中，余为每图 restore ~0.36s + 首图 GE 运行时初始化 ~6.5s）。
**注意**：GE 编译缓存按 key 命中，改 ES 降法不改 key 会静默跑旧图——所以 key 里加了
`GGML_CANNGE_EMIT_VERSION`（graph-build.h），改发射逻辑必须 bump。

## 3. 性能归因（msprof + 自加计时桩，910B3 Qwen3-4B F16）

**口径教训**：DBG-TIME 的 exec 只算入队；设备等待落在调度器的 synchronize（本次给
`ggml_backend_cannge_synchronize` 加了计数+计时，sync_calls/sync_ms 字段）。

### 分裂路径（GQA=all+SCATTER，稳态墙钟 71.5ms/token，13.7 t/s @t=4）

| 成分 | ms/token | 依据 |
|---|---|---|
| GE 后端累加（sig+analyze+bind+exec+post） | ~8.4 | DBG-TIME |
| 边界同步 | ~7.6（53 次 × 0.14ms） | backend_sync 桩 |
| **CPU v_cont 岛 ×36** | **~55** | 排除法 + 线程扫描（t=1 8.6 / t=4 13.7 / t=16 9.5） |

v_cont 岛 = CONT(permute(cache_v view)) 落 CPU：每岛 调度器 D2H 拷贝（跨后端
get_tensor，全 span 扁平读）+ CPU 置换 + H2D + ggml-cpu 线程池派发。**每 token 36 岛**。

### 合并路径（+CONT_NC，稳态 ~79ms/token，12.6-13.8 t/s）

decode = CPU(embed) + GE(1432 节点整图) + CPU(取末行) + GE(lm head)，**每 token 仅 3 边界**。
msprof 设备侧：op_summary 设备时长 ~22ms/token，其中 **Mul+ReduceSum（gemv 分解）14ms**——
Mul 物化 [K,N] 乘积再 ReduceSum，**权重被读 3 遍**（21MB→63MB/次）。但 task_time_slice 显示
设备忙 ~54ms/token：**差值是逐 kernel 下发空泡**（整图 6575 op、每 token ~数万小 kernel，
kernel 间等待占满时间线）。GE 融合确实工作（rmsnorm 五件成一串、rope Mul+Sin+Cos 成串），
但 broadcast-Mul+ReduceSum 不是可融合模式。

### 结论

两条路都卡在 ~13 t/s，机制不同：分裂=CPU 岛胶水；合并=逐 kernel 下发。
两侧共同的宝石：**把 gemv 换成真 MatMul**（消除 3 倍读放大 + 可融合/立方单元）。

## 4. BatchMatMulV3 gemv（GGML_CANNGE_BMM_GEMV=1，已入库默认关）

M==1 的 Mul+ReduceSum 分解换成单发 BatchMatMulV3（x1={1,1,K} 激活，x2={1,N,K} 权重，
adj_x2；f16 输入内部 f32 累加）。lm head 不再切 12 块（无 broadcast Mul 的 E40021）。
探针：gemv 3328/151936/3d 全过，max_rel 2.4e-4；模型文本与参考逐字一致。
**注意**：MatMulV3 在本 toolkit 无 infer_datatype 注册会编译失败（mul-mat.cpp 注释），
只能用 BatchMatMulV3。合并热 12.2→13.8 t/s（墙钟不只设备一个变量，见 §3）。

## 5. Const 去重（已入库）

构图上下文按值缓存 Const/Scalar/Vector：合并 decode 图 2335→1483 Const、7427→6575 总 op。
运行时不敏感（墙钟基本没变）→ 残余 Const 是 Reshape 的 shape 入参（编译期 ConstToAttr）。
值：编译图变小、冷编译略省。

## 6. 当前最优配置与数字（-n 100，含一次性编译/恢复摊销）

| 配置 | 冷 | 热 | 稳态墙钟 |
|---|---|---|---|
| 原始 CANN（目标） | — | 71.7 t/s | 13.9ms |
| 分裂 GQA+SCATTER（t=4） | ~6.8 | ~14.8 | ~68ms |
| 合并 +CONT_NC（+BMM） | ~0.9-1.2 | ~13.8 | ~73ms |

## 7. 下一步候选（按预期收益）

1. **分裂路径消灭 v_cont 岛**：让 kqv 的 mul_mat 直接吃 V cache 的重排视图
   （rearranged_input_base 已备），或设备侧 gather 替代 host 往返。预期 −55ms/token → 冲击 ~30 t/s。
2. **启动探针 M 对齐**：启动期探针另编一堆图（冷启动总编译的大头之一）。
3. **合并路径减 kernel 数**：BMM 之外，Slice（lm head 分片残余）/Transpose/Concat 清理；
   或评估 GE 的 AutoFuse/超核（super_kernel 目录存在于本 toolkit include）。
4. **设备侧采样**（JI 做法）：logits 不回 host，top-k 在图内——消灭每 token 的
   0.6MB D2H + CPU softmax（~2-5ms/token，未实测）。
5. FA=1（IFA/PFA）仍是乱码（note-42 末行语义老 bug），暂不碰。

## 8. 观测工具现状（本轮新增/确认）

- DEBUG_TIMING：后端七阶段 + cpy + backend_sync（新）。
- DEBUG_COMPILE：编译起止带 steady_clock 秒级时间戳（新）。
- GGML_SCHED_DEBUG=1：分裂结构（SPLIT 行）。
- msprof：op_summary（设备时长/等待/算子）+ task_time_slice（忙闲时间线）+ api_statistic
  （host API）。用法见 scripts 或本 note 的历史命令。
- 探针纪律继续有效：单算子单图先行（ggml_probe_gemv 本轮拦下并验证了 BMM 路径）。

## 9. 续批：--ignore-eos 真稳态 + GE 运行时下发坐实（2026-10-22 晚）

**口径修正**：Qwen3 对短 prompt 的回答 ~30 token 即 EOS，此前 -n 100/200/400 全被 EOS
截断，测的是"首 token 恢复摊销"而非稳态。note-36 早有此教训（长跑必须 --ignore-eos）。
-n 400 --ignore-eos 真稳态（ctx~400，t=4）：分裂 9.0 t/s（111ms/token）、合并+BMM 7.8 t/s
（128ms/token）。

**GE 运行时下发坐实**（gdb 全线程抽样，note-36 方法）：合并路径 decode 主线程热点 =
`ge::DavinciModel::Run`——墙钟花在 GE 运行时逐 kernel 下发。op_summary 的 kernel 实际
时长合计仅 ~15-22ms/token：**下发开销 ≈ 3-4× kernel 工作**。合并 decode 图 ~7300 op
（BMM 后 249 BatchMatMulV3 替代了 gemv 分解，但 Reshape 反升至 2079）。

**融合 RmsNorm 试探失败**：es::RmsNorm 单算子探针在 910B3 全过（72/72，含 310P 掉行的
{128,32}/{128,8} 形状——310P bug 不在 910B），但模型级 GE 的融合 pass 把它和前序 Add
自动改写成 InplaceAddRmsNorm，后者 tiling 要求 x1/x2 全同 dims，遇广播 Add 即编译失败。
已回滚（无 submit）。教训：写融合算子要防 GE 二次改写；或自建 AddRmsNorm 控形状。

**结构结论**：GE 图路径在本栈（910B + CANN 9.1.0）的收益上限受限于逐 kernel 下发；
要到 CANN 的 71.7（13.9ms/token）需要**整图 kernel 数降到几百级**——那意味着融合注意力
（修 note-42 的 PFA 末行语义 bug + IFA decode squeeze，或从 CANN 12.x 移
FusedInferAttentionScore 的 ES 封装），不是继续削零碎 op 能到的。

**复核基线**：原生 CANN（aclnn + aclrt 捕获）71.7 t/s 是本机同一 llama.cpp 框架实测
（note-36），框架本身不是瓶颈——瓶颈在我们的 GE 核粒度。

## 10. 续批 2：IFA 平台坑定界 + M-pad + note-36 口径基准（2026-10-22 深夜）

### IFA（decode 融合注意力）平台坑——探针定界

ggml_probe_fa mode 12（IFA decode，s_q=1，mask 前 N 可见）参数化扫描：

| KV 窗口 | keep | 结果 |
|---|---|---|
| 8 | 3 | PASS 2.4e-04 |
| 8 | 10（全可见） | PASS |
| 9/12/15/16/17/32/256 | 3 | **FAIL max_rel~1.0，~96% bad** |
| 16 | 16（全可见） | **FAIL 0.18** |
| 256 | 256（全可见） | **FAIL 0.04** |

- asl 正确到达（DEBUG_IO 实测 asl=3 n_kv=16），**与 mask/asl 无关**——全可见也错。
- 结论：本 toolkit（CANN 9.1.0 910B）的 GE IncreFlashAttention 在 kv 窗口 >8 时算错，
  疑似平台算子缺陷/约束（opp 里只有数个预编译变体）。**不是封装 bug**。
- 模型的 q 是 f32（探针是 f16）曾误判为嫌疑——加 q→f16 cast 后模型级仍乱码，已回滚
  （FA 路径默认关，不留未证实的改动）。
- FA 大面积收益（每层注意力 ~50 op → 1 op）被此平台坑卡住。JI 用的
  FusedInferAttentionScore 的 ES 封装在本 toolkit 不存在（仅 aclnn 有）。

### M-pad（99d75ba03）

rank-2 M>1 且 M%16≠0 的 matmul 之前直接落 CPU 碎图。现在后端内补零到 16 倍数再走
BatchMatMulV3、结果切回。探针 ggml_probe_mpad：m17-ffn / m2-srv 上 GE 且数值过
（5.6e-05 / 2.5e-06）；rank-3 仍按设计拒绝。**注意**：note-36 基准的 ~480 token prompt
M=480 本就 16 对齐，此改动对长 prompt 无感——它救的是短/奇数 prompt。

### note-36 口径基准（--ignore-eos -n 64，~480 token prompt，t=4，热缓存）

| 配置 | Prompt t/s | Generation t/s |
|---|---|---|
| CANN graph（原生基线） | 7373.7 | **72.3** |
| CANNGE 分裂（GQA） | 4.7 | 6.3 |
| CANNGE 合并（+CONT_NC+BMM） | 1.8 | **18.0** |

- 合并路径是**当前 decode 最优**（长上下文下分裂路径的 CPU 岛随窗口变大而更贵）。
- 会话起点 9.4 → 现 18.0（约 2x）；对 CANN 72.3 还差 4x，卡点在 GE 逐 kernel 下发
  （§9）+ prefill 碎图（M>1 的 rank-3 attention/rope/softmax 仍未覆盖）。
- prefill 1.8 t/s：长 prompt 的 rank-3 attention 与 rope/softmax 落 CPU + 同步风暴，
  与 decode 的 v_cont 岛同构。要救需 PFA（也被 IFA 同族平台问题威胁，note-42 末行 bug）。

### 诚实的路线评估

GE 图路径在 910B+CANN9.1.0 的天花板 ≈ 20 t/s 量级（逐 kernel 下发 ~15µs/op ×
每 token 数千 op）。要继续逼近 72，必须让每层注意力收敛到单算子（IFA/PFA），
而当前 toolkit 的 GE 版 IFA 在 kv>8 即错——**这是平台侧硬约束**。
可选：a) 深挖 GE IFA 的宽 cache 参数（block_table/paged 语义）；b) 修 PFA 末行 bug
（note-42，prefill 用）；c) 接受 aclnn 原生后端为生产路径，CANNGE 作研究验证。
