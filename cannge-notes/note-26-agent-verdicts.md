# note-26: 卡死裁决 + 算子补齐批（子 Agent 产出，2026-10-08）

## A. 卡死三连现象裁决（explore agent，证据链见原文）

| 现象 | 裁决 | 关键证据 |
|---|---|---|
| A：18min 98.8% CPU | **非 GE 病态编译**。状态 A 下 build fast-fail（旧 build_view 链根未注册 → "no ES tensor" → FAILED 先于 add_and_compile），GE 编译根本没启动。CPU 来源最可能是 **CLI stdin EOF 空转**（console.cpp:811 EOF 返回空串 → cli-context.cpp:511 continue → getwchar 在 EOF 不阻塞 → 死循环）；但 EOF 空转会打印 "> " 洪泛，与"无输出"记载矛盾——**若当时 stdin 是 TTY 则现象 A 的 CPU 来源仍未解释**（server 端已排除重试循环：server-context.cpp:3724 throw "stop, do not retry"） | ge-src compiler/graph/manager/graph_manager.cc:5212 编译固定 pass 流水线、pattern_fusion.cpp:31-50 fixpoint 上限 3 次——**无界编译循环无源码依据** |
| B：100% CPU 死循环 | 我方 bug：build_view 名字回退自匹配乒乓（张量与 view_src 同名，如 "Vcur-0"），visited 集合已修 | graph-build.cpp 链行走 |
| C：view wraps fast-fail | 我方 bug 当前形态：地址回退平局打破错——[1024,n_tok] 与 [128,n_tok,8] 同 span（同 buffer），std::map 指针序选中 3D → Slice 无法表达。"卡住"是框架行为（server 回 Compute error 后阻塞等待输入） | llama-kv-cache.cpp:1335 cpy_k 的 view_2d |

**F7"GE 病态编译 >10min"结论维持"无源码依据"**：EH0008 是 tiling 失败后 cleanup 的次生错误消息（非崩溃）；闭源 op 插件（CheckDimsAligned310P/matmul_v3_base_tiling）的病理无法从 ge-src 排除，但属另一类问题。

**下次判别流程**（A agent 制定）：
1. DBG-COMPILE 三标记（start/ES build done/ done）+ 建议加时间戳；只有第二条无第三条 >60s = 卡在 GE CompileGraph
2. GE 日志：ASCEND_GLOBAL_LOG_LEVEL=0 + ASCEND_SLOG_PRINT_TO_STDOUT=1（默认 level=3 只打 error，编译进度全被吞）
3. /proc 采样 bt 判别矩阵：libge_compiler→GE 编译；console/readline→CLI EOF 空转（查 fd/0 是否 /dev/null）；aclrtSynchronizeStream→驱动；strided_copy→拷贝风暴

## B. 算子补齐批（coder agent，静态落码，未构建——动态验证在 note-27）

view 修复：Tier1 名字==view_src->name+地址包含+最小 span；Tier2 地址包含+最小 span。

| op | 实现 | 出处 |
|---|---|---|
| DUP | CPY 同 dtype 别名（plan.cpp 强制端口循环扩展到 CPY+DUP） | 现有 CPY 分支 |
| NORM | 原生 es LayerNorm（CreateConst gamma/beta=1/0） | include/es/es_nn/es_LayerNorm.h |
| UNARY×11 | ABS/SGN/NEG/TANH/EXP/FLOOR/CEIL/ROUND/TRUNC→es_math 原生；RELU/SIGMOID→es_nn；GELU→GeluV2(tanh)；GELU_ERF→Gelu；GELU_QUICK→FastGelu | es_math/es_nn 各头 + 310P ops-info 确认注册 |
| GLU | SWIGLU→原生 SwiGlu(nd-1)；REGLU/GEGLU/GEGLU_ERF/GEGLU_QUICK→Slice 两半+原生 act+Mul。**修正：ggml 语义是 act(前半)*后半**（ops.cpp reglu 实现） | es_SwiGlu.h + mindspore swiglu 文档 |
| ROPE | 元语算 cos/sin（Cast/Reshape/Mul 常量 freqs/Cos/Sin）+ 原生 RotaryPositionEmbedding；NORMAL=相邻对复制、NEOX=两半复制；x Permute 成 BNSD 再逆回。门只放最小子集（mode NORM/NEOX、无 scaling、nd 3/4、contiguous） | es_RotaryPositionEmbedding.h + ops-info-transformer.json |

**未实现（明确理由）**：STEP（es_math 无）、SQRT/SQR/LOG/SIN/COS 单目（本 fork ggml_unary_op 枚举里不存在）、ELU/SOFTPLUS/HARDSIGMOID/XIELU 等（清单外）、SWIGLU_OAI/CLAMP（清单外）。
**ROUND 语义差**：Ascend 半值进偶 vs ggml roundf 半值远离零（仅精确 .5 输入有差异）。
**ROPE 风险点**（待上机验证）：cos/sin 满宽表布局按 MindSpore 文档推断；mode=1 相邻对复制未经 kernel 确认；f16 x 时 cos/sin 为 f16。
