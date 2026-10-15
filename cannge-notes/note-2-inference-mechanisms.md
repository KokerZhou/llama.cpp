# CANNGE 研读笔记二：JittorInfer 的可借鉴推理机制

> 参考代码：`../JittorInfer-tip/`（master，commit `55d1c1e`，GE/geir 构图版；这是 PR #11 的基线，机制与 ES 版相同，仅构图 API 不同）。
> 目的：提炼其中值得 CANNGE 后端借鉴的推理机制，并在第 9 节给出 upstream llama.cpp + CANNGE 下的落地分工建议。路径均相对 `JittorInfer-tip/`。

## 1. persistent decode graph

**机制**：启用 GE（`cparams.enable_ge`，src/llama-cparams.h:32，从 `params.enable_ge` 拷贝，src/llama-context-load-utils.cpp:266）后，llama_context 在建模加载期构建**一张 worst-case decode 图并永久持有**，之后每次 decode 不再调用 `llama_build_graph`，直接复用。

- 持有位置：`ggml_cgraph* graph_decode`（src/llama-context.h:141），配套独立的 `sched_decode` 与 `buf_compute_meta_decode`（src/llama-context.h:138-140）——即**为持久图单独建一套 scheduler + meta buffer**，与普通路径的 `sched`/`buf_compute_meta` 完全隔离。
- decode 主循环：`ggml_cgraph* gf = lctx.graph_decode; sched = lctx.sched_decode.get();`（src/llama-execute-decode.cpp:224-226），只做 `llama_graph_builder::llama_update_graph(lctx, ubatch, false)`（:227）而非重建；非 GE 路径则每次 `llama_build_graph` + `ggml_backend_sched_alloc_graph`（:210-222）。
- **为什么不重建**：GE 图的 AddGraph/CompileGraph/LoadGraph 是秒级开销，且后端图缓存键是 cgraph 创建时的时间戳（见第 5 节），重建即新键 → 重新编译。图 shape 不变是前提：token 数固定为 worst-case（`n_tokens = min(n_ctx, n_ubatch)`，src/llama-context-load-utils.cpp:156），可变语义（KV 长度、位置、索引）全部走"图内张量输入 + op_params 就地改"（见第 3 节）。
- 约束：GE 路径要求 `enable_cann_flash_attention`（src/llama-context-load-utils.cpp:143-147），即注意力必须图内化，不能依赖 CPU fallback。

## 2. warm-up 编译加载

**机制**：持久图构建后、服务请求前，先跑一次"只编译不执行"的 warm-up，把 GE 编译开销和 buffer 分配挪到加载期。

调用序列（src/llama-context-load-utils.cpp:143-188）：

1. 为 decode 单独 `initialize_sched_and_reserve(..., buf_compute_meta_decode, sched_decode)`（:148-151）。
2. 构造 worst-case ubatch：`n_tokens = min(n_ctx, n_ubatch)`、`n_seqs = 1`（注释 TODO worst-case，:156-159）。
3. `graph_decode = llama_graph_builder::llama_build_graph(*ctx, buf_compute_meta_decode, ubatch_pp, true)`（:173-174，worst_case=true）。
4. **`ggml_graph_set_flags(graph_decode, 3)`**（:175）——bit0=GE 整图、bit1=跳过执行。
5. `ggml_backend_sched_alloc_graph(sched_decode, graph_decode)`（:176）——**一次性把全图 tensor 的 device buffer 定死**，这是笔记一 §2"地址稳定假设"的上游保障。
6. 把位置/输出索引输入清零防脏值（:177-179）。
7. 跑一次 `llama_graph_compute`：后端走 AddGraph→CompileGraph→LoadGraph 全链，但 `cgraph->flags & 2` 时在执行前返回（ggml/src/ggml-cann/ggml-cann.cpp:1961-1963）。
8. 恢复 `ggml_graph_set_flags(graph_decode, 1)` + `ggml_backend_sched_set_flags(sched_decode, 1)`（:185-186，后者把 flags 广播到 scheduler 的 split 子图，ggml/src/ggml-backend.cpp:2175-2178），打印 Warmup 耗时（:187）。

**bit1 的替代方案**：这是 fork 私有 flags 字段（upstream `ggml_cgraph` 无此字段）。CANNGE 中应换成显式后端接口（如 `ggml_backend_cannge_graph_compile(backend, cgraph)` 只编译）或 graph 级别的 hint，warm-up 语义由 llama.cpp 调一次 compile + 一次"空执行或真执行"实现。

## 3. 重复执行与输入更新机制

**机制**：图不变，每步只更新三类东西。

1. **op_params 就地更新**：`llama_update_graph`（src/llama-graph-builder.cpp:154-166）按 arch 分发到 `llm_update_deepseek2_ge`（src/llama-graph-deepseek2ge.cpp:343-365）——遍历所有节点，把 `GGML_OP_FLASH_ATTN_PROMPT` 节点的 `op_params` reinterpret_cast 为 `flash_attn_params*` 并直接写 `sequence_lenth_kv = lctx.kv_self.n`（:358-363）。**结构体二进制布局必须与后端构图时读 `op_params` 的 layout 严格一致**（后端侧同款 struct 见 ggml/src/ggml-cann/ascend_graph_ops.cpp:1746-1757）。
2. **图输入张量内容更新**：pos / attn_indices / length_q / length_kv 都是图内 `ggml_set_input` 张量（src/llama-graph-deepseek2ge.cpp:9-28），每步 `llama_set_inputs(lctx, ubatch)` 写入（src/llama-execute-decode.cpp:254）；数据经后端 `set_tensor_async` 落到首次 alloc 定死的 device 地址。
3. **执行**：`llama_graph_builder::llama_graph_compute(lctx, gf, sched, ...)`（:256）→ backend `ExecuteGraphWithStreamAsync` + `aclrtSynchronizeStream`（ggml/src/ggml-cann/ggml-cann.cpp:1974, 1977）。输出固定为图最后一节点 `result_output`（src/llama-execute-decode.cpp:251），直接从其 device buffer `ggml_backend_tensor_get_async` 取回（:285-286）。

**要点**：这套"persistent graph + 每步小更新"模式是 GE 类后端降延迟的核心，CANNGE 应原样借鉴；但"地址不变"依赖 warm-up 时一次性 alloc（第 2 节第 5 步），upstream ggml-alloc 会复用 buffer，落地时要由后端接管该图的 buffer 分配权（见笔记一 §2）。

## 4. KV cache 稳定存储与 scatter 更新

**机制**：KV cache 是**常驻 device buffer**，同时充当 GE 图的图输入；新 token 用 ScatterUpdate 原地写入，注意力每步从整段（padded 到 n_ctx）KV 读取。

- 写入路径（`llm_build_kv_ge`，src/llama-graph-attn.cpp:361-425）：
  - 新 k/v 先 reshape 并 **pad 到 64 对齐**（`pad_n_embd = GGML_PAD(max(n_embd_head_k, n_embd_head_v), 64)`，:403-409，作者注释 TODO 想去掉）；
  - 常驻 KV `kv.k_l[il]/v_l[il]` reshape 成 `[n_embd_gqa, n_ctx]` 后 `ggml_scatter_update(ctx, k, indices, k_cur)`（:416-424）——`indices` 即本步 token 的 slot 下标（图输入张量）；
  - GE 图内对应 `GGML_OP_SCATTER_UPDATE` handler（SetSlice，GE 版 ggml/src/ggml-cann/ascend_graph_ops.cpp 的 `handle_set_slice_op`）。
- 读取路径（:426-440）：k/v reshape 回 3D `[pad_n_embd, n_head_kv, n_ctx]`，**整段 n_ctx 作为 `FusedInferAttentionScore/PromptFlashAttention` 的 KV 输入**，有效长度由 `length_kv` 输入张量（每步更新为 `kv_self.n`）控制，而非切图。
- slot 分配：scatter 模式用环形扫描 `llama_kv_cache_find_scatter_slot`（src/llama-kv-cache.cpp:406-454，`require_slots + used > size` 即失败 :415-419），选中 slot 的 pos/seq_id 写入 cell（:445-453）；decode 循环里 scatter 模式 `kv_self.head = 0` 恒成立（src/llama-execute-decode.cpp:262-264），且 GE 模式 `kv_self.n = min(size, cell_max)` 不做 pad 对齐（:120-124）。`enable_scatter_kv` 在 GE 开启时强制为 true（src/llama-context-load-utils.cpp:291，cparams 定义 src/llama-cparams.h:33）。

**取舍**：写一次 scatter + 读整段 KV，换来图 shape 与 KV 长度完全解耦（图只需编译一次）；代价是注意力每步扫 n_ctx 全量（靠 FA 算子的 length_kv 提前退出）和 64 pad 的额外访存。CANNGE 值得照抄"KV 即图输入 + 图内 scatter"的设计，但上游化时 `ggml_scatter_update` 需要映射到 upstream op（用 custom op 或 attention 算子的 cache 入参通道）。

## 5. 图缓存 / plan 缓存设计

**机制**：后端 context 持有 `std::unordered_map<size_t, uint32_t> processed_graphs`（ggml/src/ggml-cann/common.h:225-240），键 = `std::hash<std::string>{}(cgraph->graph_name_by_time)`（ggml/src/ggml-cann/ggml-cann.cpp:1885）。

- `graph_name_by_time` 是 **cgraph 创建时打的微秒时间戳**（ggml/src/ggml.c:5459-5485，`snprintf("%lld", ggml_time_us())`，:5467），`ggml_graph_view` 继承父图名（ggml.c:5485 附近 struct 拷贝）。因此缓存语义 = "同一 cgraph 对象只编译一次"。
- 被**注释掉**的替代方案（ggml-cann.cpp:1882-1884）：`graph_key = 地址 ^ (时间戳hash << 1)` 复合键——说明作者考虑过"地址+时间"但放弃了（地址会被复用，不可靠）。
- 命中分支只取回 `graph_idx` 并调空转的 `reuse_ascend_graph`（ggml-cann.cpp:1948-1952；函数体全注释，ggml/src/ggml-cann/ascend_graph.cpp:734-770），不重建任何 GE 对象。
- graph_idx 从 1 递增（`processed_graphs.size()+1`，ggml-cann.cpp:1904）。

**借鉴点**：1)"后端按 cgraph 身份缓存已编译图"的 plan cache 概念必须保留；2)键应改为**结构签名**（op 序列 + 每节点 shape/dtype + flags），upstream 每次 decode 都新建 cgraph，时间戳键会退化成全量重编译；3)缓存值可扩展为 `{graph_id, input 绑定表}`，配合第 3 节的更新机制做地址校验。

## 6. device-side presampling

**机制**：输出侧 top-k 在 device 上完成，只把 k 个 (value, index) 拷回 host，logits 全量 `[n_vocab, n_batch]` 永不上 CPU。

- llama 侧：`logits.type == LLAMA_LOGITS_TYPE_TOPK` 时调 `llama_decode_presample_cann(backend_res, res, logits.values, logits.indices, logits.len)`（src/llama-execute-decode.cpp:283；presample 数量由 `cparams.presample_count` 控制，src/llama-cparams.h:34）。
- 实现（src/llama-execute-decode.cpp:132-154）：临时 `ggml_init` 建 values/indices 两个张量 → `ggml_backend_alloc_ctx_tensors(ctx, backend)` 在 device 上分配 → `ggml_backend_cann_presample(...)` → `ggml_backend_tensor_get_async` 各取 `k*n_batch` 回 host → 立即 free buffer/ctx。
- 后端侧：`ggml_backend_cann_presample`（ggml/src/ggml-cann/ggml-cann.cpp:2617-2623）→ `ggml_cann_topk`（ggml/src/ggml-cann/aclnn_ops.h:677-710），即 `aclnnTopkGetWorkspaceSize` + `aclnnTopk`（sorted, 取 dim=1 的最大 k 个）。

**注意**：这个实现是**图外**的 aclnn 单算子调用，不是 GE 子图——即"整图执行 + 图后单算子"混合链路。CANNGE 同样可保留 aclnn 通道做图外小算子（top-k、采样辅助），无需为了统一而全部图化。

## 7. 并行布局

- **TP 权重切分（加载期）**：`llama_model_loader::build_viewer(ne, split_dim, split_num, tp_id)`（src/llama-model-loader.cpp:1505-1525，`LLAMA_AVG_SPLIT` 模式）生成 per-rank 视图；`get_spliter/del_tp_id/add_tp_id` 在 :719-721、:867。切分规则按张量名配置，调用点在 `src/llama-model.cpp:396`（静态 build_viewer 包装）、`:795`（按 `split_method` + `hparams.tp_id/num_parallel` 生成 viewer）。即 **TP 布局转换发生在权重加载期，图构建期张量已是本地形状**。
- **TP 头切分（构图期）**：`llm_build_kv_ge` 里 `head_split = num_parallel`（仅 TP 非 DP 时），`n_head_local = n_head / head_split`（src/llama-graph-attn.cpp:375-380）；GQA KV 头不足切分数时有专门的 kv_start/kv_end 换算（:382-395）。
- **all-reduce 位置**：每层 attention 后与 MLP 后各一次 `ggml_all_reduce_sum`（src/llama-graph-deepseek2ge.cpp:218-220、:276-278）——master GE 版**图内**有 `handle_allreduce_sum_op`（ggml/src/ggml-cann/ascend_graph_ops.cpp:3220）支撑；aclnn 路径则是 HCCL `HcclAllReduce`（ggml/src/ggml-cann/aclnn_ops.cpp:528-542）。PR #11 ES 版**删掉了图内 all-reduce handler**，GE 图模式下 TP 不可用——CANNGE 若用 ES 构图必须补回。
- **EP/MoE 专家切片（构图期）**：`expert_group_id = enable_expert_parallel ? tp_id : 0`，`n_expert_groups = num_parallel`（src/llama-graph-deepseek2ge.cpp:52-53）；`llm_build_moe_ffn` 内 `start_expert = group_id * (n_expert/n_groups)`、`end_expert = ...`（src/llama-graph-ffn.cpp:213-214、366-367），区间作为属性传给 fused MoE 算子（:229、:232）。专家权重本身由加载器按 rank 切好，图里只算本 rank 的 [start, end) 段。
- **DP 限制**：`enable_dp_gather` 路径直接 `GGML_ABORT("dp is not implemented.")`（src/llama-graph-deepseek2ge.cpp:244-246），DP 未实现。

## 8. model-aware fusion 与 fusion_switch.cfg

- GE 编译期融合开关文件 `ggml/src/ggml-cann/fusion_switch.cfg:1-9`：仅一项 `GraphFusion.InplaceAddRmsNormFusionPass: off`（UBFusion 段留空）。
- 注入方式：环境变量 `FUSION_SWITCH_FILE_PATH` → `AddGraph(idx, graph, {ge::ir_option::FUSION_SWITCH_FILE: path})`（ggml/src/ggml-cann/ggml-cann.cpp:1909-1924）。
- 动机（推断）：InplaceAddRmsNorm 融合会原地改残差张量，而持久图 + 地址绑定的设计里残差 buffer 生命周期由 ggml 侧管理，inplace 融合与之冲突，故按模型特性关断——这就是"model-aware fusion"的落地形态：**不是自动搜索，而是对 GE 默认 pass 的显式黑名单**。
- CANNGE 借鉴：保留"编译选项外部文件 + 环境变量注入"的通道即可；upstream 不需要更多，除非后续做 pass 调优。

## 9. CANNGE / llama.cpp 侧应如何落地

| 机制 | 归属 | 落地建议 |
|---|---|---|
| persistent decode graph（§1） | **llama.cpp 层为主** | 类似 upstream `llama-context` 增加"GE 模式"分支：持有 worst-case cgraph + 独立 sched。需要 llama.cpp 接受"图 shape 固定、语义走输入"的约定；arch 支持矩阵（先 deepseek2/qwen 系）在 graph builder 里按 `worst_case` 参数构建。后端只暴露"该 cgraph 我能否整图吃下"（supports_op 全图判定）。 |
| warm-up 编译加载（§2） | **各半** | llama.cpp 侧：加载后主动触发一次"编译+空跑"。后端侧：提供显式 `compile(warmup=true)` 接口替代 flags bit1 魔数；不要占用 `ggml_cgraph.flags`。 |
| 每步输入更新（§3） | **后端为主** | 后端缓存 `{cgraph -> (graph_id, input 表)}`，每次 `graph_compute` 前用当前 `ggml_tensor->data`/op_params 重建 gert::Tensor 输入（修复笔记一 §2/§7 的空转风险）。`op_params` 结构体 layout 由后端定义并在自定义 op 中声明。 |
| KV 常驻 + scatter（§4） | **各半** | llama.cpp 层：KV cache buffer 类型设为后端独占（可借鉴 fork 给 KV 张量用独立 buffer 的做法），indices/length 作为 `ggml_set_input`。后端层：scatter 更新映射为 GE `ScatterUpdate`/`SetSlice`，flash-attention 用 `FusedInferAttentionScore` + length 输入。 |
| plan cache（§5） | **后端** | 键 = 结构化图签名（op+shape+dtype hash），值 = `{graph_id, 编译产物}`；这是纯后端内部实现，llama.cpp 无感。 |
| device-side presampling（§6） | **后端 + llama.cpp 小改** | 后端保留 aclnn topk 通道。llama.cpp 侧可把 fork 的 `presample_count`/`LLAMA_LOGITS_TYPE_TOPK` 思路化为 sampling 阶段的 device top-k 优化，初版可不做。 |
| TP/EP 布局（§7） | **llama.cpp 层为主，后端补算子** | TP 权重切分参考 fork 的 loader viewer（upstream 已有 TP 加载基础，主要补 head/GQA 切分换算 :382-395）；all-reduce 若走图内则后端必须实现 all-reduce handler（补回 ES 版删掉的能力，master ggml/src/ggml-cann/ascend_graph_ops.cpp:3220 可对照），或规划 HCCL 图外调用。EP 只需把 [start,end) 专家区间作为 MoE 算子属性。 |
| fusion 开关（§8） | **后端** | 编译选项文件 + env 注入照搬；默认空配置，遇到 inplace 类 pass 与 buffer 管理冲突再加黑名单。 |

**总结**：1-5 是"GE 整图后端"的通用骨架（persistent graph、warm-up、plan cache、输入重绑定、KV scatter），CANNGE 应以"后端 plan cache + 显式 compile 接口 + 输入重绑"三件套为核心，把 fork 里靠私有 flags/时间戳/空转函数隐式完成的事全部显式化；6-8 是性能增强项，可后续迭代。架构分工上，**"图怎么持久、输入怎么更新"属于 llama.cpp 与后端的接口契约，"图怎么编译缓存"完全属于后端内部**。

---

**取证备注**：本笔记全部行号已对照 `../JittorInfer-tip/`（master，commit 55d1c1e）实际文件核实；其中 master 与 PR #11 ES 分支的关键差异（图内 all-reduce 有无、rope 单双实现）已随文标注。
