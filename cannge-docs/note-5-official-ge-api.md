# note-5: CANN 9.1 GE C++ API 官方参考（头文件精读 + 官网文档）

来源：本地容器 `quay.io/ascend/cann:9.1.0-310p-ubuntu22.04-py3.10-devel` 头文件
（`/usr/local/Ascend/cann-9.1.0/include/`，签名均原文摘录）+ 昇腾官网 CANN 9.1.0 文档树
（`hiascend.com/document/detail/zh/CANNCommunityEdition/910/...`）。
整理日期：2026-09-10。本笔记是 plan/compile/load/execute 各阶段编码的权威依据。

## 1. 两套 API 并存：ge_api.h (v1) 与 ge_api_v2.h (v2)

`ge/ge_api_v2.h:10-51` 迁移注释原文要点：

- v2 取代 v1：`libge_runner_v2.so` 取代 `libge_runner.so`。
- v2 移除：所有 Feed/Fetch 接口、所有 `std::string` 参数接口、`SetGraphFixedFeatureMemoryBase`、`GetVariables`、`PaRemapped`。
- `AddGraphWithCopy` 更名为 `AddGraphClone`；`Session` 被 `GeSession` 取代。
- v2 合并了编译接口（BuildGraph+CompileGraph）与 WithStreamAsync 执行接口。

**当前选型：v1（ge::Session + libge_runner）**，与 JittorInfer 一致。理由：v1 在 9.1 仍完整可用；
`ExecuteGraphWithStreamAsync`（gert::Tensor 零拷贝执行）为 v1/v2 共有。若日后升级 v2，需改
AddGraphWithCopy->AddGraphClone、RunGraph 系接口换 gert::Tensor。

## 2. 生命周期 API（ge/ge_api.h）

```cpp
// :35-41，std::string map 版标 deprecated，9.1 仍可用；应使用 AscendString map
GE_FUNC_VISIBILITY Status GEInitialize(const std::map<AscendString, AscendString> &options);
GE_FUNC_VISIBILITY Status GEFinalize();  // "Finalize GE, release all resources"
```

- `ge::Status = uint32_t`（`external/ge_common/ge_common_api_types.h:603`），非枚举。`ge::SUCCESS=0`，
  `ge::FAILED=0xFFFFFFFF`（`ge/ge_api_error_codes.h:99-100`，位域宏编码）。错误描述经
  `GEGetErrorMsg()` 查询。
- ACL 风格错误码在 `ge/ge_error_codes.h`：如 `ACL_ERROR_GE_PARAM_INVALID=145000`、
  `ACL_ERROR_GE_LOAD_MODEL=545001`、`ACL_ERROR_GE_MODEL_EXECUTE_TIMEOUT=545601`、
  `ACL_ERROR_GE_UNLOAD_MODEL=545009`。失败日志应打印 `GEGetErrorMsg()` 文本而非裸码。

## 3. ge::Session 全表面（ge/ge_api.h:51-418）

构造/析构：

```cpp
explicit Session(const std::map<AscendString, AscendString> &options);  // :57
~Session();                                                              // :58
```

- **析构语义头文件无注释**（是否卸载图/释放显存：未找到官方依据）。我们的约定（整改#6）：
  先 `aclrtSynchronizeStream` 再 `delete session` 再 `GEFinalize`，官网异步流程示例同样是
  synchronize -> free -> GEFinalize 顺序（见 note-7）。

关键方法（参数类型必须精确）：

| 方法 | 签名要点 | 备注 |
|---|---|---|
| AddGraph | `Status AddGraph(uint32_t id, const Graph &graph);` + 带 options map 重载 | **const Graph&**；官网：同 Graph 对象重复注册会共享对象；graph_id 须唯一 |
| AddGraphWithCopy | `Status AddGraphWithCopy(uint32_t id, const Graph &graph [, options]);` | v1 的"拷贝入图"版；v2 中叫 AddGraphClone。JittorInfer 观察到 AddGraph 会改传入 Graph 对象，故保留原对象时应用此接口（待 310 实测确认） |
| RemoveGraph | `Status RemoveGraph(uint32_t graph_id);` | 官网约束"无"；IsGraphNeedRebuild 返回 true 时需 RemoveGraph 后重新 AddGraph+BuildGraph |
| CompileGraph | `Status CompileGraph(uint32_t graph_id);` | |
| LoadGraph | `Status LoadGraph(uint32_t id, const std::map<AscendString,AscendString> &options, void *stream) const;` | 头注释："只用于加载已完成 CompileGraph 的图，不支持重复加载图"；链路 CompileGraph + LoadGraph + ExecuteGraphWithStreamAsync |
| RunGraph | `Status RunGraph(uint32_t id, const std::vector<Tensor> &in, std::vector<Tensor> &out);` | ge::Tensor（host 描述类型，graph/tensor.h） |
| RunGraphWithStreamAsync | 同上用 ge::Tensor | |
| ExecuteGraphWithStreamAsync | `Status ...(uint32_t id, void *stream, const std::vector<gert::Tensor> &in, std::vector<gert::Tensor> &out);` | **gert::Tensor**（POD，exe_graph/runtime/tensor.h:31），outputs 非 const 引用。零拷贝执行主接口 |
| RunGraphAsync | `(uint32_t id, const std::vector<ge::Tensor> &in, RunAsyncCallback cb);` | 回调 `RunAsyncCallback = std::function<void(Status, std::vector<ge::Tensor>&)>`；status==SUCCESS 时即可处理数据 |
| GetCompiledGraphSummary | `CompiledGraphSummaryPtr GetCompiledGraphSummary(uint32_t id);` | shared_ptr，ge_graph_compile_summary.h:119；可查"模型执行所需内存资源大小及内存是否可刷新、复用"（官网 07_0103） |
| 内存基址系列 | SetGraphConstMemoryBase / UpdateGraphFeatureMemoryBase / SetGraphFixedFeatureMemoryBaseWithType / UpdateGraphRefreshableFeatureMemoryBase | feature 内存地址可刷新的配套入口；与 ge.featureBaseRefreshable 配置配合 |
| 外置权重/调试 | RegisterExternalAllocator / ShardGraphs / SaveGraphsToPb / GraphDebugJSONPrint | |

`ge::CompileSession`：**全 include 树无此符号**，不存在。

GE 无"Session 并发"支持（官网 07_0088："Session 暂不支持并发执行，同时 Session 会独占资源，
多个 Session 同时创建时，Session 可能因为资源不足而创建失败"）。多线程+每线程一 Session+
各传不同 `ge.session_device_id` 是官方认可的多 device 模式。

## 4. ge::Graph 与 IR 层（graph/graph.h）

```cpp
Graph &SetInputs(const std::vector<Operator> &inputs);    // :51 "触发内部图的构建"
Graph &SetOutputs(const std::vector<Operator> &outputs);  // :53
graphStatus SetOutputs(const std::vector<std::pair<GNode, int32_t>> &);  // :56-59 输出索引=节点的第几个输出，顺序=图的输出顺序
Graph &SetTargets(const std::vector<Operator> &targets);  // :69 语义无注释（未找到官方依据）
```

ES 流程不用 SetInputs/SetOutputs（见 note-6）。离线编译（aclgrphBuild*）入口在
`ge/ge_ir_build.h`，与本后端运行时编译路径（Session::CompileGraph）无关。

## 5. ge.exec.* 配置常量（external/ge_common/ge_api_types.h）

声明形如 `const char_t *const OPTION_XXX = "ge.exec.xxx";`，注意**前缀不统一**：

| 常量 | 字符串 | 级别/备注 |
|---|---|---|
| OPTION_EXEC_DEVICE_ID | ge.exec.deviceId | 全局；在线推理默认 -1 |
| OPTION_SESSION_DEVICE_ID | **ge.session_device_id** | session 级；多线程多 Session 各传不同 device id 用此键。**ge.cpp 已用此键（2026-09-10 修正，原误写 session_device_id）** |
| OPTION_EXEC_SESSION_ID | ge.exec.sessionId | |
| OPTION_GRAPH_RUN_MODE | **ge.graphRunMode**（无 exec 前缀） | 0=在线推理(默认)，1=训练。ge.cpp 已用 |
| TILING_SCHEDULE_OPTIMIZE | **ge.tiling_schedule_optimize**（无 exec 前缀） | 0=关(默认)，1=开；条件：静态 shape + 算子支持 tiling 下沉；310P 在支持列表。ge.cpp 已用 |
| PRECISION_MODE / PRECISION_MODE_V2 | ge.exec.precision_mode / _v2 | 语义详见 note-7 §3 |
| OPTION_EXEC_REUSE_ZERO_COPY_MEMORY | ge.exec.reuseZeroCopyMemory | 0=关(默认)，1=开：输入输出零拷贝内存独立规划、跨执行复用 |
| OPTION_EXEC_DISABLE_REUSED_MEMORY | ge.exec.disableReuseMemory | 0=开启内存复用(默认) |
| 静态字符串版 | ge.exec.atomicCleanPolicy / memoryOptimizationPolicy / **ge.exec.staticMemoryPolicy** | 这三个只有 std::string 常量名，语义见 note-7 §2 |
| 动态 shape 族 | ge.exec.dynamicInput / dynamicGraphExecuteMode / dataInputsShapeRange 等 | |
| 其他 | OPTION_EXEC_ENABLE_DUMP*、PROFILING_*、OPTION_EXEC_ENABLE_INCRE_BUILD、OPTION_EXEC_GRAPH_EXEC_TIMEOUT、OPTION_STATIC_MODEL_OPS_LOWER_LIMIT 等，共 60+ | 完整清单在同文件 :31-238 |

## 6. 与现有代码的对照结论

- ge.cpp 用 AscendString map 调 GEInitialize/Session 构造：正确，避开 deprecated。
- config 键 ge.exec.deviceId / ge.graphRunMode / ge.tiling_schedule_optimize /
  ge.exec.precision_mode / ge.exec.reuseZeroCopyMemory：与头文件常量一致。
- ge.session_device_id：已修正。
- AddGraph 目前未被调用（graph-build 为桩）；启用时若担心 GE 改传入 Graph 对象，用
  AddGraphWithCopy 保原对象，代价待 310 实测。
- 错误处理应补 GEGetErrorMsg() 取文本（当前只打数字码）。
