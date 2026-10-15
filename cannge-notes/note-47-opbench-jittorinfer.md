# note-47: 逐个算子过关（opbench）+ JittorInfer 差异的量化回答

## 1. 每算子验收表（probes/ggml_opbench，Qwen3-4B-2507 真实形状，值与 CPU 逐元素比对）

| 算子（形状） | 值 | CPU ms | CANNGE ms | 判定 |
|---|---|---|---|---|
| mul_mat prefill M=32 {2560,4096} | OK | 6.39 | 0.085 | GE **75x** 快 |
| mul_mat gate gemv {2560,9728} | OK | 1.45 | 0.13 | GE **11x** 快 |
| mul_mat wo gemv {4096,2560} | OK | 0.71 | 0.095 | GE **7x** 快 |
| soft_max {256,1,32} | OK | 0.058 | 0.041 | GE 快 |
| glu swiglu {9728,1} | OK | 0.022 | 0.042 | 相当（地板） |
| cast f32->f16 {2560,1} | OK | 0.016 | 0.045 | 相当（地板） |
| rms_norm {2560,1} | OK | 0.009 | 0.050 | 地板 |
| add {2560,1} | OK | 0.007 | 0.046 | 地板 |
| cont_2d {128,32}->{4096,1} | OK | 0.007 | 0.047 | 地板 |
| get_rows embd {2560,1} | OK | 0.004 | 0.050 | 地板 |
| rope 2D/3D（单独验证） | OK | — | — | 通过 |
| gemv 2D/3D、bigmm 10.5M 权重 | OK | — | — | 通过（1.4e-6） |

**结论：没有慢算子，也没有错算子。** 全部 12 类值正确。大算子 GE 全面 7-75 倍快于 CPU。4 个"GE SLOWER"全是 **~0.05ms 的图执行地板**（ExecuteGraphWithStreamAsync 下发+同步的固定成本）——8KB 的 add 在 CPU 上 7us，但任何 GE 图最少 50us。**小算子的唯一正道是合图减少执行次数**，不是修算子。

## 2. CPY 门 bug（本批修复）

ggml 约定 **CPY 的 dst 编码为 src[1] = 节点自身**；门的"自引用=原地危险"检查把**所有 CPY 全部误杀**（cast 恒落 CPU）。修复：自引用检查豁免 CPY（其 case 内查真 src0/src1）。

## 3. JittorInfer 为什么快/没错——量化回答

JittorInfer 与 CANNGE 用的是**同一个 GE**。差异全部在图工程（note-39 源码级结论 + 本测量）：

| | JittorInfer | CANNGE（当前） |
|---|---|---|
| 图粒度 | 整模型 1 张常驻图，decode=1 次下发+1 次同步 | 每 token **221 个碎片**，每碎片 1 次下发（50us 地板）+边界 H2D/D2H+同步 |
| 每 token GE 执行 | ~16ms（note-38） | ~11ms（221×0.05ms）——**相当** |
| 胶水 | ~0 | ~450ms（下发地板 221×0.05=11ms + 边界拷贝/同步 + CPU op 计算 ~200ms） |
| 算子覆盖 | 24 个图内 handler，**不支持即 GGML_ABORT**（保证永不碎片） | 有缺口 → 调度器切碎片（kqv GQA、v_cont、lm head、prefill 对齐） |
| KV | 常驻 device，图内 ScatterUpdate | host 侧 scatter+跨片拷贝 |
| 采样 | device 侧 top-k 只回传 (value,index) | logits D2H |

**JI 加速 = 全栈耦合换来的"一次下发"；GE 让这一次下发值得。** CANNGE 的 GE 用对了（算子全面快于 CPU），但没收回三个前提：**完整算子覆盖、稳定缓冲、O(1) 次同步**。差距不在任何单个算子。

## 4. 路线（不改 GE 语义，只减碎片）

1. 权重已上设备（256K 帽移除后，note-46）；decode 221 片中 CPU 的：36 kqv（GQA 8→32）+36 v_cont（转置物化）+36 swiglu（跟随）+lm head。
2. kqv/v_cont 收编 = decode 碎片 221→~110；swiglu 跟随 →~74（2/层）。
3. 开启 GGML_SCHED_EVENTS=1（事件化已落地，批13b）消除边界全同步。
4. 理想终态 ~4-6 执行/token：tg = 墙钟/(6×0.05ms + 计算 ~16ms) → ~50 t/s 量级。

## 5. 批C 进展（GQA/CONT 收编，note-46 续）

- **CONT 非连续**：探针值正确（cont permuted-view PASS, 7x 快于 CPU），但模型级乱码（vc1）——疑与 GQA 同批引入的其他改动纠缠；现 env 门控 `GGML_CANNGE_CONT_NC`（默认关）。
- **kqv GQA（8→32 heads）**：BroadcastTo/Tile 在 GE 内被降低为 Reshape 且误读 ES 常量形状（E19999）；改用**纯 Reshape 广播技巧**（e0 加头维 1、e1 拆 {R,G}、输出 {R,G,N}→{H,1,N} 全为字节恒等重标）——**隔离探针精确**（ggml GQA 映射 = h = g*R + r，g=h/R，探针 gemv-gqa PASS 1.8e-6）；**模型级半乱码**（真 k/q/v cache 视图的 strided staging 与探针 dense 张量有别）。已 env 门控 `GGML_CANNGE_GQA`（默认关）。
- **GQA 开启时 tg 6.5 t/s**（碎片塌缩生效，较 1.7 提升 3.8x）——证明收编即收益；剩余为模型级数值细节。
- 稳态：连贯、3.0/1.7、探针 11+1 PASS。下批：对照真 cache 视图（非 dense staging）的 GQA 值。
- **批D 追加（g-major permute 也失败）**：纯 reshape 方案的根本障碍——ggml GQA 映射 h = g·R + r 与"内存恒等 reshape 拆分 h = r·G + g"不兼容；加 Permute(0,1) 换序后模型级仍乱码（gqa9）。隔离探针过、模型挂的差异锁定在 **k/q/v cache 视图的 strided staging 字节布局**（探针用 dense 张量）。GQA 保持 env 门控默认关；下批第一步 = 探针改用与模型一致的 permuted-cache-view 输入复现模型级失败，再修 strided_copy 或 build 的输入重解释。
