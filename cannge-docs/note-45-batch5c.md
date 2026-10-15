# note-45: 批5c - repack2d / GEMV 混合 dtype / 分裂结构真凶（attention 配毒）

## 本批落地（全部连贯验证）
1. **repack2d 分块重排**：strided_copy 的 bounce 路径从逐元素 memcpy(2B) 改为 16x16 分块 2D repack（`ggml_cannge_repack_2d`，显式四向 stride）。命中模式：dim1..3 连续 + dim0 携带 C*es（cache_v 转置视图族 nb={2048,2,256}）。stage 26.0s→23.7s/60tok。DBG-COPY bounce 打印（DEBUG_COMPILE）。
2. **GEMV 门混合 dtype**：M==1 路径 build 本就双 cast f32，门却要求 src0/src1 同 dtype → decode QKV/FFN 投影（f16 权 × f32 norm 输出）全被拒。把等 dtype 检查移到 GEMV 分支之后。decode matmul 门全过（DBG 实证 gemv=1）。

## 关键发现：~6 split/层的真凶是 attention 在 CPU 反向"配毒"
- llama.cpp 默认 flash_attn=AUTO→ENABLED → 图里是 FLASH_ATTN_EXT 节点 → 我们的门（FA env off）拒 → **CPU**。
- sched pass-2 从输出侧反向升级：attention 在 CPU → 把喂它的 QKV 投影、rope 链全拉去 CPU → decode 每层 ~5 个 CPU MUL_MAT split（实测 181 CPU MUL_MAT/graph）。这正是 note-37"650ms/token"的同族结构。
- GEMV dtype 修复对 split 数无效（matmul 本来就将被接受也被毒害）。

## -fa off 实验（经典 attention 全 CANNGE 路线）——撞墙记录
`-fa off` 让 llama 建经典 attention 图（kq matmul+soft_max+V matmul，全 CANNGE 支持），但**模型加载期 abort**：
`ggml-backend.cpp:941: pre-allocated tensor (cache_v_l0 (view)) in a buffer (CANNGE0) that cannot run the operation (VIEW)`
- 机理：-fa off 改变 llama KV cache 的 buft 选择，cache 视图进了 CANNGE buffer，但 CANNGE supports_op 不认 VIEW/RESHAPE/PERMUTE/TRANSPOSE 元数据视图 → backend_from_buffer 找不到可执行后端 → ABORT。

### 批6 进展（2026-10-15）
1. supports_op 放行元数据视图（VIEW/RESHAPE/PERMUTE/TRANSPOSE）→ **abort 解除**，-fa off 可跑。
2. GEMV 门再放开（`return true`：非连续 src0、混合 dtype 全接，staging+cast 兜底）；非 GEMV 去掉连续性要求（对齐/256K 帽保留）。默认路径回归连贯（冒烟实测）。
3. **批量 GEMV build**（classic attention V 侧 [kv,N,B]×[kv,1,B] → Mul 广播 + ReduceSum 末轴）+ 探针 3D 用例全绿（1.8e-6）。
4. **2D GEMV n_dims 匹配 bug**（重大）：门条件 `ggml_n_dims(src0)==ggml_n_dims(src1)` 对 [K,N]×[K,1] 恒假——src1 尾维为 1 时 ggml_n_dims 更小！批6 早期 decode 的 WO/QKV/FFN 2D GEMV 全因此被拒回 CPU 而未被发现。改为按 ne 匹配（`src1->ne[1]==1 && src0->ne[2]==src1->ne[2] && src0->ne[3]==src1->ne[3] && ggml_n_dims(src0)>=2`）。探针 2D+3D 全绿；默认基准 3.4/1.1、-nkvo 8.5/2.7。
5. **-fa off decode 仍乱码**：kq/kq_soft_max decode 值与默认逐位一致，分歧在 V 侧 matmul 输出（kqv-0 (permuted)：i0 精确一致、i1+ 错；探针同形状通过）→ 上下文内某操作数到达方式与探针不同（疑 CONT/视图链）。prefill（2/9 token）全对。下批入口：在上下文里 dump V-matmul 的两个操作数逐项对值。

## 最终基准（-n 100 --ignore-eos, 910B3）
| 配置 | pp t/s | tg t/s |
|---|---|---|
| 批4 末（基线） | ~4.4 | 0.6 |
| 批6 默认 KV-on | 3.2 | 1.1 |
| 批6 -nkvo | 6.5 | 2.6 |
| CANN graph 对照 | 7300 | 71.7 |

tg 距 CANN 仍 ~27x。剩余路径收敛到一条：attention 上 CANNGE（-fa off 需批量 GEMV + classic 链数值调试；或修 IFA 的 KV>=16 kernel 路径后开 GGML_CANNGE_FA）。
