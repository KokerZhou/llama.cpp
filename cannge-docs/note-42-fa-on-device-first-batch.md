# note-42: FA 上设备首批——IFA/PFA 构建落地，语义差最后一行（2026-10-14）

目标（用户新目标）：修潜在 bug，性能逼近原生 CANN（71.7 tg / 7300 pp）。

## 本批交付：GGML_OP_FLASH_ATTN_EXT 的 GE 构建（opt-in）

- **选型**：CANN 9.1.0 的 es_transformer **无 FusedInferAttentionScore ES 封装**
  （JI 的 cann1230 社区包才有），但有 `IncreFlashAttention`(decode) 与
  `PromptFlashAttention`(prefill)——选用这对原生 ES 算子
- **门**（ggml-cannge.cpp）：K/V f16、q f16/f32、输出随 prec（F32 独立判）、
  拒 sinks/ALiBi/softcap、D%16==0、GQA 比例；**默认关，GGML_CANNGE_FA=1 开**
- **构建**（graph-build.cpp）：ggml [D,S,N,1] → reshape [1,N,S,D] → permute
  {0,2,1,3} → BSND；mask [S_kv,S_q] → transpose → [1,1,S_q,S_kv]（Cast f32）；
  S_q==1→IFA（key/value 列表），else→PFA；输出 reshape 回 GE shape + 按 prec
  Cast；S_q=1 输出先显式 reshape 4D 再 permute（GE 可能 squeeze）

## 探针结论（ggml_probe_fa，模型形状对拍 CPU）

- **单位阵 mask：bit-exact PASS** —— q/k/v/mask 逐行寻址全对
- **causal：仅最后一行 (s=S-1) 错**（S=15/16 同、GQA/非 GQA 同、mask f16/f32 同、
  next_tokens/pre_tokens 窗口消融无效）——聚集/归一化语义在"全可见行"上与 ggml 分歧
- decode S=1：GE 编译错（Permute 轴越界，IFA 输出 squeeze 相关，待修）
- 模型级 FA=on：decode Compute error（默认路径不受影响）

## 默认路径回归（FA 关）

- 冒烟连贯 ✓；基准：全默认 4.4/0.9，-nkvo 31.6/2.3（与 note-41 持平）

## 下一批（按优先级）

1. **PFA 末行语义**：用 2-3 个 kv 的手工对拍拿到 got 的解析式（是多了项还是
   归一化错），或试 inner_precise=2 / sparse_mode=1 / mask 转 BOOL(1/0) 三变体
2. **IFA decode 路径**输出 squeeze 修复
3. #20（b 类）根因：同款值转储对 m2 的 {ADD,RMS,MUL} 子图
4. 同步批量化/事件化（KV-on 的 650ms/token 胶水）
5. 探针：ggml_probe_fa（含 mode 0-4 变体）

## 当前全景（vs 原生 CANN：tg 71.7 / pp 7300）

| 配置 | pp | tg | 主要瓶颈 |
|---|---|---|---|
| CANNGE 全默认 | 4.4 | 0.9 | KV-on 序列化 650ms + 残余胶水 |
| CANNGE -nkvo | 31.6 | 2.3 | 边界拷贝/同步 ~230ms |
| +FA（待修） | — | — | 预期消除 KV 跨界，tg 目标 5-10 |

未 commit。工作树含：SET_ROWS 输出口根治、散射同步、ROPE 全链、GEMV、
FA 构建（opt-in）、调试工具族、note-30~42 共 13 篇。

## 续批补充（同日晚）

- **探针参考系修正**：CPU 后端对探针里的 f16 mask 实际按均匀权重处理（参考
  本身失真）；用确定性公式反解证实 CANNGE 的 attention 权重与解析解一致
  （a=0.8933 vs 0.8926）——PFA 权重语义是对的
- **模型输入布局结论**：llama 的 q（permute 视图）与 k/v（cache slice 视图）
  内存序都是 (d,n,s) → 天然 BSND；to_bsnd 已改为纯 reshape（探针同步 mimic）
- **mask 1D 修复**：[KV,1] mask 的 ggml n_dims=1 → 跳过 transpose 直接 reshape
- **误改回退教训**：通用 PERMUTE/TRANSPOSE 的 nd=ggml_n_dims 是对的（holder
  秩跟随 ggml 秩）；强制 4 轴打爆了低秩 holder（默认路径 Compute error 已回退
  恢复，冒烟连贯 3.1/1.0）。FA 路径需要的 4D permute 必须在显式 4D reshape
  之后局部使用
- **当前态**：FA opt-in（GGML_CANNGE_FA=1），模型级仍 Compute error（FA 局部
  permute 秩问题待修）；默认路径全绿

## 下一批修正路线

1. FA 内 permute 秩：to_bsnd 显式 reshape 4D 后，permute 用 4 轴（仅 FA 局部）
2. #20（b 类）根因
3. 同步批量化/事件化（KV-on 胶水 650ms）

## 批 3：permute 秩根因修复 + IFA 编译通过（2026-10-14 深夜）

- **permute 秩根因（通用潜在 bug 修复）**：PERMUTE/TRANSPOSE 构建的秩取自
  `ggml_n_dims(node)`，但 shuffle 尾部 size-1 维时 node 秩 > 输入 holder 秩
  （decode q 视图 [128,32,1,1]→permute→[128,1,32,1]，nd 3 vs holder 2）→
  GE InferShape 拒。修为 `ggml_n_dims(src0)`（跟随输入 holder 秩）。
  默认路径回归连贯 ✓。**此 bug 此前潜伏：decode permute 从未进过 CANNGE 子图**
- **IFA/PFA 现已编译通过**（mask 1D + BSND reshape + permute 秩三修复叠加）
- **mask 必须是 DT_BOOL**（EZ3002 明示：仅 BOOL/INT8/UINT8，拒 DT_FLOAT）：
  `Greater(Cast(mask,f32), -1e30)` 把 ggml 加性 0/-inf 转 keep-bool
- **当前阻塞**：IFA execute 期 driver memcpy segfault（aclrtMemcpyAsync 内，
  35 层 decode 图最后一层；疑某设备指针非法——下一帧查 input_base 别名/
  输出端口地址）。FA 仍 opt-in，默认路径无恙
