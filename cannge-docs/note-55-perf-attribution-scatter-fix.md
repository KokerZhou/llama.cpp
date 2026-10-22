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
