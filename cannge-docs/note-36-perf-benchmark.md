# note-36: 性能基准——CANNGE vs CANN(graph/eager) vs CPU（2026-10-14）

Qwen3-4B-Instruct-2507 F16，`-ngl all -c 512 -ub 512 --temp 0 --no-warmup`，
~140 token 中文 prompt（一次 prefill）+ -n 64 decode。910B3 / CANN 9.1.0。

## 基准矩阵

| 配置 | Prompt t/s | Generation t/s |
|---|---|---|
| CPU (-ngl 0) | 33.8 | 4.6 |
| **CANN eager**（GGML_CANN_ACL_GRAPH=off） | **7401** | **44.8** |
| **CANN graph**（默认，decode 图化） | **7300** | **71.7** |
| CANN graph + GGML_CANN_PREFILL_USE_GRAPH=1 | 4159 | 72.9 |
| CANN graph + GGML_OP_OFFLOAD_MIN_BATCH=1 | 7314 | 71.7 |
| **CANNGE**（GGML_CANNGE_GRAPH=1 全默认） | **3.1** | **0.9** |
| CANNGE + -nkvo | 5.8 | 0.9 |

结论：**CANNGE 比原生 CANN 慢 ~2300×（pp）/ ~80×（tg），比纯 CPU 还慢
~11×（pp）/ ~5×（tg）**。用户的预判成立，差距是结构性的。

## 栈采样剖析（decode，6 样本×全线程，聚合）

热点帧（去空闲后）：
1. `ggml_compute_forward_mul_mat` / `ggml_graph_compute_thread`（~46 帧）——
   **所有 decode matmul 在 CPU**（M=1 过不了 f16 16 元素对齐门）
2. `HdcPcieRecvPeek`/`ioctl`/`DmsIoctl`/`mmSleep`（~36 帧）——设备驱动
   轮询等待 = **device 大部分时间在空转等同步**
3. `httplib::ThreadPool::worker`/`server_queue::worker_loop`（~90 帧）——
   cli-server IPC 线程池（结构性常开，非热点）
4. `ggml_backend_cannge_graph_compute → ggml_cannge_strided_copy`（3/6 样本命中）

## 根因解读（三层）

1. **切图碎片化**：rope/FA 不支持 + matmul M%16 门 → 每层被切成多段
   CPU/CANNGE 交替子图；每个 split 边界 = H2D/D2H + sched 全同步 →
   流水线每边界排干一次（设备线程在 mmSleep/HdcPcieRecvPeek 等待印证）。
   原生 CANN 全算子在设备、无边界，所以 72 t/s。
2. **首 token 编译**：pp 一次性承担所有签名子图的 GE 编译（病态编译未发
   生，但常编译 ~分钟级）；graph 缓存命中后 decode 不再付。
3. **#21 修复的代价**：V 转置视图 bounce 回退按元素 scatter（9.4M 次 2B
   memcpy/token，估 ~280ms/token ≈ 25%）——正确性优先的已知开销，
   正路是注意力上设备消除 V 跨界（见下）。

## 优化方向（按性价比）

1. **注意力链上设备**（根治切图）：FA / ROPE / M=1 GEMV 支持 → 每层一个
   大子图，边界消失；CANNGE 的 GE 整图优势才能发挥。工作量大。
2. **降同步开销**：split 边界改事件驱动（去 sched 每 split 全同步）、
   边界拷贝合并批量。
3. **混合架构**（note-15 阶梯的落地）：CANN 原生跑整模型（72 t/s 基线），
   CANNGE 只接管 GE 有优势的大算子/整段 —— 需要 sched 级双后端协作。
4. 短线：bounce 路径模式缓存（按签名预计算偏移）、nb0%4==0 时 4B 成对
   拷贝特例。

## 复现

- `scripts/04_bench_compare.sh`（六配置矩阵）
- 采样：setsid 起 `--ignore-eos` 长 decode（注意 EOS 会提前终止 -
  n 长跑必须 --ignore-eos），gdb `thread apply all bt` × N 聚合
- 原生 CANN 构建：`cmake -B build-cann -DGGML_CANN=ON -DUSE_ACL_GRAPH=ON
  -DGGML_CANNGE=OFF -DSOC_TYPE=Ascend910B3`（USE_ACL_GRAPH 910B 可开，
  310P 禁；运行时 GGML_CANN_ACL_GRAPH=off 切 eager）
