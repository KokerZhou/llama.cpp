# CANNGE 研读笔记四：版本对比与四方执行模型对照

> 2026-09-09/10 讨论定稿。三个代码版本：`JittorInfer-pr11/`（PR #11 head，cc18685，ES 构图）、`JittorInfer-tip/`（master，55d1c1e，GE 构图）、`llama.cpp/`（434ddbbc，vendored ggml）。

## 1. JittorInfer 版本谱系与差异（git 实证）

PR #11 基点 = `dd5ddfd`（pre#11）。三组关系：**pre#11 (GE) → post#11 (ES)** 是同 lineage 的前端重写；**master = pre#11 + 2026 年新演进（仍 GE）**。

**post#11 vs pre#11：纯前端重写，语义未变**
- handler 集合完全相同（各 24 个，一一对应）；`ascend_graph_ops.cpp` 整体重写，`ascend_graph.cpp` 重写 472 行（`op::Data/AddOp` → `EsGraphBuilder/CreateInput/BuildAndReset`）。
- 执行链路零变化：session 生命周期、AddGraph/Compile/Load/Execute、图缓存键、地址绑定、`reuse_ascend_graph` 空转，`ggml-cann.cpp` 几乎未动。
- 附带：ES 构建接入（GenerateEsPackage + add_es_library + GE_FUNC_VISIBILITY/CXX11_ABI workaround）、`REFACTORING_STEPS.md`、rope_cache 小改、输出 index 获取修复（9369264）。

**master vs pre#11：实质演进，全部在 GE 路径**
- 新图内 handler：`DIV`、`SUM_ROWS`、`ALL_REDUCE_SUM`（TP allreduce，配 `RANKTABLEFILE` -> `ge.exec.rankTableFile/rankId`）。注意这三个是 master 后加的，非 ES 版删除。
- 正确性修复：输入 Data op 命名加 index 后缀（重名冲突）等。
- 头文件布局适配新 CANN（`ge_api.h` -> `ge/ge_api.h`）。
- 新模型图：qwen2/2.5、qwen3、qwen3moe、llama2、llamage（各含 GE 变体）；qkv/ffn 合并优化；TP GE。
- 新资产：`tests/ops/` 算子测试框架（~1500 行，对 CANNGE op 测试有参考价值）；混入 atrace 追踪垃圾文件。

**选型结论（已定）**：普通 op 用 ES 构图（借 PR #11 的 24 个 handler 形态）；raw GE 通道（GetCGraphBuilder）保留为兜底；**op 语义基准用 master 的 GE handler**（div/sum_rows/allreduce/rope 逐一对齐）；executor/session/plan 缓存全部自研。ES 分支"落后"影响小：只借构图模式，落后部分在 master 有 GE 写法对抄。

## 2. 四方执行模型对照表

| 维度 | llama.cpp CANN (aclGraph OFF) | llama.cpp CANN (aclGraph ON) | JittorInfer | CANNGE（本方案） |
|---|---|---|---|---|
| 图粒度 | 逐算子 eager | 整 cgraph 录制回放（一张 RI graph） | 整模型一张 GE 图 | 整图一张 GE 图；编译失败处分段 |
| 编译器优化 | 无 | 无（仅消下发开销；手写 ADD+RMS_NORM 特判） | GE 全量（融合/tiling/内存规划） | GE 全量 |
| 编译开销时机 | 无 | capture 在首次执行（毫秒级） | 加载期 warm-up 一次（秒级） | 加载期显式 warm-up（秒级），plan 缓存复用 |
| 输入重绑定 | 不适用 | 录制即绑死地址/shape；cgraph 变则重录（LRU） | 绑死裸指针+空转 reuse（脆弱） | 每次执行按当前地址重建 gert::Tensor（设计目标） |
| 310P | eager 可用 | **CMake FATAL_ERROR 禁用** | 主战场（整图已验证） | 第一目标 |
| 910/910B | eager 可用 | 可用 | 可用，功能最全 | 第二阶段验证 |
| 不支持 op | supports_op 拒绝->调度器切 CPU | 同左 | **GGML_ABORT** | supports_op 拒绝->调度器；内部按 D1 阶梯降级；绝不 abort |
| 持久 decode 图 | 无（每步重建 cgraph，allocator 复用 buffer） | 无（靠重录） | 有（worst-case 图常驻） | llama.cpp 层 persistent 图 + 后端 plan 缓存 |
| prefill | eager | 默认不 capture（seq>1，可 env 开） | prompt/decode 图分开 | 同为 GE 图，按 shape 签名分别缓存 |
| 自定义算子 | 无 | 无 | **构建期强制依赖** OPP 包 | 可选+门控（当前 hold，见 note-3 D2） |
| 多卡 TP | 无 | 无 | GE 图内 allreduce（master） | 初版不做；预留图内 allreduce 通道 |
| 回退阶梯 | eager+CPU 两级 | eager+CPU 两级 | GE 整图 或 abort（两级） | 完整阶梯（note-3 D1） |

一句话：**llama.cpp aclGraph 是"eager 的加速器"（capture/replay，无编译）；JittorInfer 是"整图编译但耦合 fork 且只许成功"；CANNGE = "整图编译 + 上游可维护 + 失败按阶梯回退"**。

## 3. ACL Graph vs GE vs ES（勿混，文档实锤见 note-3 D6）

- **ACL Graph**：`aclmdlRICaptureBegin/End` + `aclmdlRIExecuteAsync`，运行时任务捕获/回放（模型运行实例），无编译器；捕获期 device 内存冻结；试验特性，部分版本不含 310P。llama.cpp USE_ACL_GRAPH 属此类。
- **GE**：编译器+运行时（AddGraph/CompileGraph/LoadGraph/Execute）。
- **ES**：GE 的 C++ 构图前端，产物与 raw GE 同为 `ge::Graph`，优化等价。
