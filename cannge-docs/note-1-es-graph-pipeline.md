# CANNGE 研读笔记一：JittorInfer PR#11 的 ES/GE 构图与执行链路

> 参考代码：`../JittorInfer-pr11/`（PR Jittor/JittorInfer#11 头分支，commit `cc18685`，"适配cann 1230 的社区包"；该 PR 截至 2026-09-09 仍为 open，未合入 upstream）。
> 目的：为在 llama.cpp 内置 ggml（upstream）中实现 CANNGE（Ascend Graph Engine）后端提供对照。以下所有路径均相对 `JittorInfer-pr11/`。

## 1. 构图总入口与数据流

### 调用链总览

```
ggml_backend_sched_graph_compute_async(sched, cgraph)
  └─ ggml_backend_cann_graph_compute(backend, cgraph)          ggml/src/ggml-cann/ggml-cann.cpp:1810
       ├─ (flags bit0==0 或 USE_ACLNN=1) → ggml_backend_cann_graph_compute_aclnn   ggml-cann.cpp:1761
       │     └─ 逐节点 ggml_cann_compute_forward(...)                            ggml-cann.cpp:1788
       └─ (GE/ES 整图路径)
             ├─ 首次: GEInitialize + new Session (+ processed_graphs map)         ggml-cann.cpp:1833-1867
             ├─ graph_key = hash(cgraph->graph_name_by_time)                      ggml-cann.cpp:1879
             ├─ 缓存未命中:
             │    build_ascend_graph(cgraph, ctx, input_init, output_init)        ggml-cann.cpp:1901-1902
             │      └─ build_ascend_graph_es(...)                                ggml/src/ggml-cann/ascend_graph.cpp:289-300
             │           ├─ process_input_tensors_es(leafs)  → CreateInput        ascend_graph.cpp:312-314, 202-251
             │           ├─ process_input_tensors_es(nodes)  → CreateInput        ascend_graph.cpp:318-320
             │           ├─ for node in cgraph->nodes: switch(node->op)
             │           │     → handle_<op>_es(graph_builder, node, map, i)      ascend_graph.cpp:328-595
             │           ├─ create_output_tensors_es(...)                         ascend_graph.cpp:601
             │           └─ return *graph_builder.BuildAndReset(graph_outputs)    ascend_graph.cpp:603
             │    session->AddGraph(idx, graph[, FUSION_SWITCH])                  ggml-cann.cpp:1918-1924
             │    session->CompileGraph(idx)                                      ggml-cann.cpp:1930
             │    session->LoadGraph(idx, {}, stream)                             ggml-cann.cpp:1935
             ├─ 缓存命中: reuse_ascend_graph(...)（空壳，见 §2）                  ggml-cann.cpp:1946-1948
             └─ session->ExecuteGraphWithStreamAsync(idx, stream, in, out)        ggml-cann.cpp:1967-1969
                  + aclrtSynchronizeStream(stream)                                 ggml-cann.cpp:1976
```

### 构建器内部数据流（`build_ascend_graph_es`，ascend_graph.cpp:297-609）

1. **创建 ES builder**：`ge::es::EsGraphBuilder graph_builder("Graph")`（ascend_graph.cpp:302），同时 `cann_ctx.n_ctx = cgraph->n_ctx`（:303，供 RoPE cache 定长用）。
2. **输入两遍扫描**：`process_input_tensors_es`（ascend_graph.cpp:202-251）先扫 `cgraph->leafs`，再扫 `cgraph->nodes` 中 `op == GGML_OP_NONE` 的节点及其 src 中的 OP_NONE 张量；每个输入调 `graph_builder->CreateInput(input_index, name, dtype, format, dims)`（:223-229），建立 `std::map<ggml_tensor*, ge::es::EsTensorHolder> ggml_tensor_to_es_tensor_map`（:306），并同步生成绑定好地址的 `gert::Tensor` 推入 `input_init`（:232-233）。索引 = `graph_inputs->size() + index_offset`（leaf 偏移 0，node 偏移 `n_leafs`，:222）。
3. **算子节点逐个转换**：大 switch（ascend_graph.cpp:336-594），每个分支调一个 `handle_*_op_es`，返回值（`EsTensorHolder`）登记进 map；**只有最后一个非空 op 节点**（`find_last_op_node`，:189-198）被推入 `graph_outputs`（如 :341-343）。
4. **收尾**：`create_output_tensors_es` 为输出创建绑定 `last_op_node->data` 的 `gert::Tensor`（:260-276），然后 `BuildAndReset(graph_outputs)` 产出 `ge::Graph`（:603）。输入/输出为空时打印错误并返回空图 `ge::Graph("EmptyGraph")`（:605-607）。

### GE（pre-#11）版对照

master（`../JittorInfer-tip`）同一函数是全程裸 GE API：`op::Data` + `graph->AddOp`（JittorInfer-tip/ggml/src/ggml-cann/ascend_graph.cpp:262-267）、`graph.SetInputs(...).SetOutputs(indexed)`（:724）。ES 版把这些换成 `CreateInput`/builder 链式 op/ `BuildAndReset`，但**自定义算子仍回退到裸 GE Graph**（见 §5）。executor/session 流程两版完全一致（JittorInfer-tip/ggml/src/ggml-cann/ggml-cann.cpp:1815-1980）。

## 2. 输入/输出张量的绑定机制

### gert::Tensor 与地址绑定

- 每个图输入张量在构图当下就创建一个 `gert::Tensor`，**直接内嵌 ggml 张量的裸 `->data` 设备地址**：
  - `create_bound_tensor_with_ptr(desc, data_ptr, data_size)`（ascend_graph.cpp:86-113）：设置 OriginShape/StorageShape/Format（ND）/`TensorPlacement::kOnDeviceHbm`/DataType，然后 `tensor.SetData(gert::TensorData(ptr, nullptr, size, kOnDeviceHbm))`（:108-111）。
  - 输入侧封装：`create_graph_input_tensor_es`（ascend_graph.cpp:124-144）——注意 ES 的 `EsTensorHolder` 没有 `SetPlacement`，代码从 producer op 的 output desc 绕出来再设 `kPlacementDevice`（:133-138，注释里标了 TODO）。
  - 输出侧：`create_output_tensors_es` 绑定 `last_op_node->data` + `ggml_nbytes(last_op_node)`（ascend_graph.cpp:271-273）。
- 这些 `gert::Tensor` 存进 backend context 的 `std::vector<gert::Tensor> input_init, output_init`（ggml/src/ggml-cann/common.h:309），由 `ExecuteGraphWithStreamAsync` 每次整体传入。

### 重复执行时地址如何处理

- **什么都不做**。缓存命中分支调 `reuse_ascend_graph`（ascend_graph.cpp:611-647），但函数体内重建输入/输出张量的代码**全部被注释掉**（:615-644），实际只是 `return SUCCESS`。
- 也就是说：**第二次及以后的执行，GE 拿到的还是第一次构图时绑定的地址**。正确性依赖三个上游条件同时成立：
  1. cgraph 是 persistent 的（同一 `ggml_cgraph*` 对象 ⇒ 同一 `graph_name_by_time` 键，见 §7）；
  2. 图中所有 ggml 张量的 `->data` 指针跨执行不变（warm-up 时一次性 `ggml_backend_sched_alloc_graph` 定死）；
  3. GE 运行时使用 zero-copy/地址复用语义（`ge.exec.reuseZeroCopyMemory=1`，ggml-cann.cpp:1845）。
- llama 侧的配合：warm-up 构图后 flags 改回 1 并每次仅更新输入内容（pos/indices/kv 长度）与 `op_params`，不重建图（`src/llama-context-load-utils.cpp:164-174`、`src/llama-execute-decode.cpp:224-228`、`src/llama-graph-deepseek2ge.cpp:289-310`）。

### 风险（实现 CANNGE 时必须正面解决）

upstream llama.cpp 的 compute buffer 由 ggml-alloc 按张量生命周期复用，同一地址会被不同张量复用、跨图 realloc；且 backend split 会把一个 cgraph 切成多段。**照搬"构图时绑死裸指针 + 复用时空转"的模式会静默读到旧地址**。CANNGE 需要：或维护 `ggml_tensor* → gert::Tensor` 并在每次 `graph_compute` 前按当前 `->data` 重建 input vector；或强制整图后端独占一块稳定 buffer 并把该约束写进 `supports_op`/buffer type 的契约里。

## 3. Session 生命周期

### 创建（惰性，首次走 GE 路径时）

`ggml_backend_cann_graph_compute` 中 `cann_ctx->ascend_graph == nullptr` 分支（ggml-cann.cpp:1833-1867）：

GEInitialize 配置逐项（ggml-cann.cpp:1835-1847）：

| 配置项 | 值 | 含义 |
|---|---|---|
| `ge.exec.deviceId` | 当前 device 字符串 | 图编译/执行目标设备 |
| `ge.graphRunMode` | `0` | 整图下发模式 |
| `ge.tiling_schedule_optimize` | `1` | tiling 调度优化 |
| `ge.exec.precision_mode` | `allow_fp32_to_fp16` | 允许 FP32→FP16 自动降精度（提性能） |
| `ge.exec.reuseZeroCopyMemory` | `1` | 复用/zero-copy 内存语义——与 §2 的地址绑死假设直接相关 |

随后（:1856-1866）：`cann_ctx->ascend_graph = std::make_unique<ggml_ascend_graph>()`；`processed_graphs` 清空（:1861-1862）；`session = new Session(options)`（空 options）。

### 图的注册-编译-加载-执行（每次 graph_compute 都走）

- 未命中缓存：`AddGraph(graph_idx, graph, {FUSION_SWITCH_FILE: env FUSION_SWITCH_FILE_PATH})`（ggml-cann.cpp:1918-1924；cfg 内容见 `ggml/src/ggml-cann/fusion_switch.cfg:1-9`，仅关 `InplaceAddRmsNormFusionPass`）→ `CompileGraph(idx)`（:1930）→ `LoadGraph(idx, {}, stream)`（:1935）。graph_idx 从 1 递增（`processed_graphs.size()+1`，:1897-1898）。
- 执行：`ExecuteGraphWithStreamAsync(idx, stream, input_init, output_init)`（ggml-cann.cpp:1967-1969），之后**同步流** `aclrtSynchronizeStream`（:1976）。异步版 `RunGraphWithStreamAsync` 的调用被注释（:1960-1965）。
- flags bit1=1 时（warm-up）在 Add/Compile/Load 之后、Execute 之前直接返回（ggml-cann.cpp:1955-1957）。

### 销毁

`~ggml_backend_cann_context`（common.h:330-348）：set_device → 销毁 copy_event → 销毁 8 条 stream → `ge::GEFinalize()` → `delete ascend_graph->session`。**顺序是先 GEFinalize 再 delete session**（common.h:342-347），且 `ggml_ascend_graph` 是裸指针 `ge::Session*`（common.h:238）——整个 GE 环境按"一个 backend context 一个 session"设计，多 device 各自独立 GE 初始化。CANNGE 移植时要考虑多 backend 实例（多卡）下 GEInitialize/GEFinalize 的配对计数，upstream 环境比 fork 更容易触发重复 init/finalize。

## 4. 维度顺序约定

- **统一反转**：ggml `ne[0..3]`（低维在前）→ GE shape 需要倒序 push。两处实现：`create_tensor_desc_for_node`（ascend_graph.cpp:61-77，`for d = GGML_MAX_DIMS-1 downto 0`，跳过 `ne[d]<=0`）和 `build_output_shape(reverse=true)`（ggml/src/ggml-cann/ascend_graph_ops.cpp:28-47，注释明写"GGML 维度从 0 到 3，昇腾通常反向"）。**例外**：部分 op 用 `reverse=false`（保持 ggml 顺序），另有 `squeeze_ggml_tensor_shape` 去掉前缀 1（:55-63）。
- **Permute/Transpose 轴号直接复用 ggml 语义**：RoPE 中 5D reshape `[2, ne0/2, ne1, ne2, ne3]` 后 `Transpose(perm_order={0,1,2,4,3})`（ascend_graph_ops.cpp:1211-1236），输出 shape 手工按 perm 重排（:1229-1233）。
- **RoPE 的 shape hack**：输入 reshape 成 5D 把旋转维拆成 `[dim/2, 2]` → transpose → `RopeExtCustomV2` → transpose 回去 → reshape 回 4D（ascend_graph_ops.cpp:1214-1321）。这是自定义算子把"交错布局"转成"分半布局"的关键，移植时若改算子实现需同步改这段。
- **padding 约定**：KV 侧 q/k/v 一律 pad 到 `GGML_PAD(max(head_k, head_v), 64)`（`src/llama-graph-attn.cpp:363-369`，作者注释 TODO 想移除）；matmul 行 padding 常量 `MATRIX_ROW_PADDING 512`（common.h:51）；`fusion_switch.cfg` 关闭 InplaceAddRmsNorm 融合（见 §3）。CANNGE 把这些隐性约定收进后端时要保留或显式化，否则 GE 编译期 shape 推断会和 ggml 侧不一致。
- **dtype 映射**：`get_data_type`（ascend_graph_ops.cpp:137-158），注意 `GGML_TYPE_Q4_0→DT_INT4`、`Q8_0→DT_QINT8`，默认落到 `DT_FLOAT`。

## 5. raw GE 兜底写法（自定义算子接法）

ES builder 没覆盖的算子用"拿到裸 Graph → 加裸节点 → 包回 EsTensorHolder"的模式，模板见 RoPE（ascend_graph_ops.cpp:1238-1305）：

```cpp
ge::Graph *graph = graph_builder.GetCGraphBuilder()->GetGraph();   // :1240
ge::op::RopeExtCustomV2 rope_op(name_rope.c_str());                // :1271 自定义算子（OPP proto）
rope_op.update_output_desc_dst(desc_out_rope);                     // :1276
rope_op.set_attr_ne0(ne0); ...                                     // :1279-1281
ge::GNode g_node = graph->AddNodeByOp(rope_op);                    // :1284
graph->AddDataEdge(*es_in.GetProducer(), es_in.GetProducerOutIndex(),
                   g_node, input_idx);                             // :1289-1300
ge::es::EsTensorHolder result(
    graph_builder.GetCGraphBuilder()->GetTensorHolderFromNode(g_node, 0));  // :1303-1305
```

要点：ES 张量 → 裸节点用 `GetProducer()/GetProducerOutIndex()` 拆接；裸节点 → ES 张量用 `GetTensorHolderFromNode()`。flash attention 则完全用 ES API `FusedInferAttentionScore(...)`（ascend_graph_ops.cpp:1677-1823，attrs：`input_layout="BSND"`、`sparse_mode=1`、`inner_precise = seq_q>1 ? 2 : 0`）；MoE 的 GroupedMatmul 也有 ES 封装 `create_moe_grouped_matmul_es`（ggml/src/ggml-cann/ascend_graph_ops_create.cpp:6-44，`split_item=2, group_list_type=1, group_type=0, act_type=0, y_num=1`）。**结论：ES 版并非纯 ES，自定义算子通道必不可少**；CANNGE 应直接规划同样的双轨（ES 原语 + GetCGraphBuilder 兜底）。

## 6. ES 依赖的构建接入方式

全部在 `ggml/src/ggml-cann/CMakeLists.txt`：

1. **前置**：`CANN_INSTALL_DIR` 取自 `ASCEND_TOOLKIT_HOME`（:1-4）。
2. **引入 ES 生成工具**：`list(APPEND CMAKE_MODULE_PATH "${CANN_INSTALL_DIR}/include/ge/cmake")` + `find_package(GenerateEsPackage REQUIRED)`（:135-136）。
3. **原型库**：`add_library(opgraph_all INTERFACE)`，输出目录指向 run 包内置 proto so（:141-144，注释说明正常场景直接用 run 包已有 ES 产物，无需自己生成）。
4. **生成 ES API 包**：`add_es_library(ES_LINKABLE_AND_ALL_TARGET es_all OPP_PROTO_TARGET opgraph_all OUTPUT_PATH ${CMAKE_BINARY_DIR}/output)`（:147-151）。
5. **链接**：`target_link_libraries(ggml-cann PRIVATE es_all)`（:154-156）。
6. **头文件**：`es_all_ops.h`（聚合头，ascend_graph.cpp:33）、`es_c_graph_builder.h`（ascend_graph_ops.cpp:14），均经 es_all 传递；另有 `all_ops.h/ge_api.h/graph.h`（ascend_graph.h:23-29）与自定义 proto `op_proto.h`（include 路径含 `${CANN}/opp/vendors/customize/op_proto/inc`，:69）。
7. **库清单**：`ascendcl, nnopbase, opapi, acl_op_compiler, ascendc_kernels, hccl, graph, ge_runner`（:93-102）+ `es_all`；链接目录 `${CANN}/lib64`、`${CANN}/compiler/lib64/stub`（:112-115）。
8. **CANN 版本**：README 要求 **>= 8.2.RC1.alpha001**（README.md:25）；本分支 commit 信息即"适配 **cann 1230 的社区包**"（2025-12-30 开源社区 run 包，REFACTORING_STEPS.md 阶段零说明 ES 能力依赖该 run 包内置产物）。
9. **已知坑（已在工程里打补丁，移植时照抄）**：
   - 引入 es_all 后头文件搜索路径变化，会搜到**不含 `GE_FUNC_VISIBILITY` 的 `ge_error_codes.h`** 导致编译失败 → `-DGE_FUNC_VISIBILITY=` 空定义绕过（CMakeLists.txt:119-121）。
   - 强制旧 ABI：`-D_GLIBCXX_USE_CXX11_ABI=0`（:118）。
   - SOC 版本用 `npu-smi` 自动探测并写 cache（:18-45），编译宏 `-DASCEND_910B` 之类（:117）。
10. **自定义算子是构建期强依赖**：`add_custom_target(custom_rope_ext ... install.sh)` + `add_dependencies(ggml-cann custom_rope_ext)`（CMakeLists.txt:127-133），把 `kernels/custom_rope_ext/op_project` 打包安装进 `${ASCEND_TOOLKIT_HOME}/opp/vendors`；可选的是 `LLAMA_USE_JITTOR_OPS`（jittor_infer 的 `cust_opapi`，:7-14, 79-90，控制 `GGML_OP_FLASH_ATTN_JITTOR_V1` 等编译，ggml-cann.cpp:1542-1546）。

## 7. 移植到 CANNGE 时不能照搬的清单

| # | 不能照搬的东西 | 证据 | CANNGE 应改为 |
|---|---|---|---|
| 1 | **私有 op 枚举**：`GGML_OP_ALL_REDUCE_SUM/DPSKV2_FUSED_MOE/TO_ZERO/MOE_FUSED[_CPU]/FLASH_ATTN_PROMPT[_CPU]/FLASH_ATTN_JITTOR_V1/GET_SLICE/SCATTER_UPDATE/RMS_NORM_FUSED` | ggml/include/ggml.h:533-545 | upstream 枚举值不同，不能按数值 switch；用 upstream 已有 op（FLASH_ATTN_EXT 等）+ `ggml_op_custom`/graph extra 表达私有语义 |
| 2 | **`ggml_cgraph.flags` 魔数**：bit0=GE 整图、bit1=跳过执行（warm-up），`USE_ACLNN` 环境变量可覆盖 | ggml/src/ggml-cann/ggml-cann.cpp:1815-1819, 1955-1957；`ggml_graph_set_flags` ggml/src/ggml.c:5499；广播到 split 子图 ggml/src/ggml-backend.cpp:2175-2178 | 显式 backend 接口（如 `ggml_backend_cannge_set_graph_mode`）或 cgraph opaque extra，禁止占用上游 struct 字段 |
| 3 | **时间戳图 id**：`graph_name_by_time` 在 `ggml_graph_new` 时打微秒时间戳（ggml/src/ggml.c:5461-5469），图缓存键 = `std::hash` thereof（ggml-cann.cpp:1879）；`ggml_graph_view` 继承父图名（ggml.c:5520） | ggml/src/ggml-impl.h:302 | 结构化图签名（op 序列 + shape + flag），否则 upstream 每次新建 cgraph 都全量重新编译 |
| 4 | **地址稳定假设**：构图时绑死裸 `->data`，命中缓存后 `reuse_ascend_graph` 空转（函数体全注释） | ggml/src/ggml-cann/ascend_graph.cpp:611-647；绑定 ascend_graph.cpp:86-113 | 每次执行前按当前地址重建 `gert::Tensor` 输入表（mutable input），不要依赖上游 buffer 布局 |
| 5 | **强制 OPP 自定义算子依赖**：`add_dependencies(ggml-cann custom_rope_ext)` 构建期安装 vendor 包 | CMakeLists.txt:127-133；算子使用 ascend_graph_ops.cpp:1271 | standalone ggml 后端应保持可独立编译：自定义算子改为可选（编译开关/运行期 dlopen），RoPE 优先用纯 ES 原语实现 |
| 6 | 附带差异（注意而非照搬）：ES 版**删了** DIV / SUM_ROWS / ALL_REDUCE_SUM 的图内 handler（GE 版在 JittorInfer-tip/ggml/src/ggml-cann/ascend_graph_ops.cpp:343, 3157, 3220）；TP 的 all-reduce 只在 aclnn 路径有（aclnn_ops.cpp:528-542），GE 图内 `GGML_ASSERT(!enable_tensor_parallel)`（src/llama-graph-deepseek2ge.cpp:61） | 同上 | CANNGE 若要多卡 TP，要么补图内 AllReduce 算子，要么规划跨 backend split（HCCL 在图外） |

---

**取证备注**：笔记中所有行号已对照 `../JittorInfer-pr11/`（commit cc18685）核实。GE 版对照行号引自 `../JittorInfer-tip/`。
