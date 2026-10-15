# note-21: CANNGE 算子覆盖差距审计（本机 310P3，2026-10-08）

基线：/root/cannge-ggml @ 28558340a。审计方式：test-backend-ops.cpp 注册段
(:8822-10855) + `test-backend-ops --show-coverage` 实测（109 covered / 13 无测试）。
完整原始结论由 explore agent 产出，本文件为落盘摘要。

## 1. CANNGE 当前支持集（ggml-cannge.cpp:590-735 supports_op + graph-build 分发）

- NONE（叶子，仅 env=1）: :598-600
- ADD/SUB/MUL/DIV（f16/f32 广播）: :614-630 → graph-build.cpp:172-179
- SCALE: :632-633 → :181-189
- SOFT_MAX（**仅无参**：src1==null 且 scale==1 且 max_bias==0）: :635-645 → :191-198 SoftmaxV2
- UNARY（**仅 SILU**）: :647-649 → :200-208 Swish
- RMS_NORM: :651-652 → :210-230 元语分解
- MUL_MAT（**仅 2D**、同 dtype f16/f32、32B 对齐）: :654-679 → :232-243 BatchMatMulV3
- RESHAPE/PERMUTE/TRANSPOSE: :681-684 → :245-287
- VIEW（自身 dense 连续）: :686-690 → :87-153
- CONT（限 src0 连续）: :692-695 → :292-301 纯 alias
- CPY（f16/f32、同 shape、dst 非 view）: :697-707 → :303-319
- GET_ROWS（src0 f16/f32、I32 索引）: :709-711 → :321-328 GatherV2
- CONCAT: :713-730 → :330-339
- 所有 *_inplace 通用拒绝: :604-608（GE 静默丢弃别名写回，note-10 F1）
- 其他 default false: :732-733；build default fail: graph-build.cpp:341-342
- 图模式开关：无 eager 路径，GGML_CANNGE_GRAPH≠1 时 supports_op 全 false（:593-595）

## 2. 模型刚需缺口（Qwen3 dense 前向实证）

Qwen3 图 src/models/qwen3.cpp + src/llama-graph.cpp + src/llama-kv-cache.cpp：
**整图当前无法在 CANNGE 运行**，断点按出现顺序：

| 缺口 | 证据 | 说明 |
|---|---|---|
| ROPE | qwen3.cpp:91,100；llama-kv-cache.cpp:1956,1966 | rope_ext；es_transformer 有 RotaryPositionEmbedding（note-6 §2），310P 支持性待按 note-7 §1 取证 |
| SET_ROWS | llama-kv-cache.cpp:1350,1385,1406 | KV 写入（cpy_k/cpy_v）；note-6 无 Scatter 类 ES op，需设计；plan.cpp:152-163 有写回预留 |
| FLASH_ATTN_EXT | llama-graph.cpp:2615-2632 | 默认 flash 路径；备选手工路径同样断：4D batched MUL_MAT 被 :678"仅 2D"拒；soft_max_ext 参数化被 :635-645 拒；cont(transpose(v)) 非连续被 :692-695 拒 |
| batched MUL_MAT(>2D) | ggml-cannge.cpp:677-678 | GE BatchMatMul 支持 3 维 batch（note-7 §4），映射已是 BatchMatMulV3，**改 L0 即可** |
| 参数化 SOFT_MAX | :635-645 | 元语可组：Mul(scale)+Add(mask)+SoftmaxV2，**改 L0+build 即可** |

## 3. 补齐优先级（模型刚需 × ES 可取证 × 改动量）

1. **参数化 SOFT_MAX + batched MUL_MAT**——纯 L0 放宽 + build 元语，打通手工注意力路径
2. **ROPE**——es_transformer 原生，先按 note-7 §1 operatorlist 取证 310P/910B
3. **SET_ROWS**——无 ES 对应，KV 写入硬依赖，需设计（staging 写回机制已有雏形）
4. **FLASH_ATTN_EXT**——PromptFlashAttention，需取证；且 310 无 FA 硬件（CrispASR 经验）
5. **DUP / NORM / UNARY 扩展（GELU 系等）/ GLU**——es_nn/es_math 原生多，纯增量
6. 明确不补（理由见下）

## 4. 明确不补清单 + 理由

| 组 | 理由 |
|---|---|
| 量化 MUL_MAT/GET_ROWS（Q4_0…Q8_0、IQ 系） | ggml_cannge_ge_dtype 仅 F16/F32/I32（graph-build.h:37-51）；量化需引入反量化路径，超出当前阶段；NS 回退 CPU 是正确行为 |
| 全部 *_inplace | GE 静默丢弃别名写回（note-10 F1，ge memory_copy.cc:574-579），等 GE outputReuseInputMemIndexes 声明机制，从上游规避而非补丁 |
| 训练侧（CROSS_ENTROPY_LOSS、OPT_STEP_*、各 _BACK） | 推理后端定位（note-3）；es_nn 有优化器 op 但非刚需 |
| SSM/RWKV/GATED_DELTA_NET/LIGHTNING_INDEXER/SOLVE_TRI/DSV4_HC_* | note-6 全量 ES 清单无对应，需自定义核或超长分解，性价比低 |
| CONV/POOL/IM2COL/COL2IM/UPSCALE | es_cv 面向检测/图像，当前文本/VL-LLM 路径不触发（ViT  coverage 后续单独立项） |
| 13 个无测试 op（ADD1、CUSTOM、WIN_PART 等） | test-backend-ops 不覆盖，无验收标准，不补 |

## 5. TBO 基线复现记录（本机）

- 命令：GGML_CANNGE_GRAPH=1 ./build/bin/test-backend-ops test -b CANNGE0 -o ADD,MUL,MUL_MAT,SCALE,RMS_NORM,SOFT_MAX,SILU,GET_ROWS,CONCAT
- 结果：160/45/2366，逐 op 与 note-16 §3 基线完全一致
- 日志：tbo-310p3-28558340a.raw.log / .stripped.log；解析器：parse_tbo.sh
- 解析坑：FAIL 用例行首为 `[OP] ERR = x > y` 前缀；verdict 可能跨行（stderr 噪声穿插）
