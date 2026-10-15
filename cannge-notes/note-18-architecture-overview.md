# note-18: CANNGE 架构总览（规范）

建立：2026-09-11。代码基线：`llama.cpp` 分支 cannge（2026-09-13 由 cann-ge 统一改名），commit `d85b6bd08` + 修复 commit 28558340a（F1/F2/F3 + in-place）。本文描述**修复后**行为。
GE/ES API 语义的权威出处按 note-11 的文档来源注册表分级引用。

## 1. 定位与职责边界

CANNGE 是 ggml 的 Ascend Graph Engine（GE）整图编译后端：一次 `graph_compute` 把整张
cgraph 经 ES 前端翻译成 `ge::Graph` IR，GE 编译为设备可执行图并缓存，后续同签名图直接
复用编译产物执行。**禁止 abort**；吃不下即拒绝，交上层调度（note-3 D1、note-4 §2）。

与 GGML_CANN（eager ACLNN 后端）的职责切分：

| 项 | CANNGE | GGML_CANN |
|---|---|---|
| 执行模型 | GE 整图编译（ES 构图 → AddGraph → CompileGraph → LoadGraph → Execute） | 逐算子 eager aclnn |
| 设备发现/buffer/stream | **自有**一套（aclInit、aclrtMalloc buffer、aclrtStream），与 GGML_CANN 完全独立 | 自有 |
| 后端名/设备名 | `CANNGE` / `CANNGE<i>` | `CANN` / `CANN<i>` |
| 图路径开关 | env `GGML_CANNGE_GRAPH=1`（默认 off） | 无整图路径 |
| 共存 | 两后端可同时注册；调度器按 supports_op 各自裁决，互不知晓 | 同左 |

两者必须保持独立、可共存：CANNGE **禁止**复用 GGML_CANN 的 context/buffer/stream，
**禁止**以任何方式干扰其设备初始化（ggml-cannge.cpp:23-25 注释为契约表述）。

## 2. 分层架构

```
cgraph (graph_compute)
  │  ① L0 静态能力预检（逐节点 supports_op）
  │  ② plan analyze（每次调用重建，plan.cpp）：IO 分析 / produced 语义 / 边界输出 / 视图解析
  │  ③ plan signature（结构签名：op+flags+shape+strides+op_params+src 连接）
  ▼
plan cache（backend context，signature → plan，上限 GGML_CANNGE_MAX_PLANS=64）
  │  未命中：④ ES 构图（graph-build.cpp，needs_staging 即失败）
  │           ⑤ GE 惰性初始化（首个 plan 编译时）→ AddGraphWithCopy → CompileGraph
  │           编译失败 → 缓存 FAILED plan（构建失败不缓存）
  │  命中且 LOADED：⑥ 绑定（每次调用按当前 t->data 重建 gert::Tensor）
  │                 输出-输入地址重叠检测 → staging buffer（F1）
  │                 ⑦ ExecuteGraphWithStreamAsync → staging/副作用 D2D 写回（同 stream）
  ▼
GE session（每 backend context 一个；进程级 GEInitialize/GEFinalize 引用计数）
```

四个层次各自的关键抽象：

### 2.1 ES 构图层（graph-build.cpp/h）

- `EsGraphBuilder builder("ggml_cannge")`；外部输入按 `io.inputs` 顺序 `CreateInput(i, name,
  dtype, FORMAT_ND, ge_shape)`，保证执行期绑定下标与构图下标一致（graph-build.cpp:70-82）。
- 逐节点 switch 翻译为 ES 原语；张量表 `std::map<ggml_tensor*, EsTensorHolder>`，
  查不到即 "no ES tensor" 构建失败（**禁止**静默跳过）。
- 轴序约定唯一入口 `ggml_cannge_ge_shape()`：GE shape = 反转 ggml `ne[:ndims]`
  （ggml 低维在前、GE 最外维在前；graph-build.h:53-64）。Permute 序、Slice 偏移一律经此换算。
- 输出端口按 `io.output_tensors` 精确顺序 `SetOutput(holder, i)`，与 analyze 注册序一致。
- 产物 `BuildAndReset()` 交出 `ge::Graph*`，编译成功后调用方立即释放 IR
  （session 经 AddGraphWithCopy 持有副本，见 §3）。

### 2.2 GE 编译层（ge.cpp）

- 进程级 `GEInitialize`/`GEFinalize` 引用计数 + 互斥锁（ge.cpp:50-83）；配置项：
  `ge.exec.deviceId`、`ge.graphRunMode=0`、`ge.tiling_schedule_optimize=1`、
  `ge.exec.precision_mode=allow_fp32_to_fp16`、`ge.exec.reuseZeroCopyMemory=1`
  （注释标注 re-tune on 310 measurements）。配置键与头文件常量一致性已对照
  `ge/ge_api_types.h`（note-5 §5）。
- 每 backend context 一个 `ge::Session`，session 级选项 `ge.session_device_id`
  （note-5：多线程多 Session 各传不同 device id 的官方模式）。
- **禁止**用裸 `AddGraph`：D6 实证 AddGraph 会修改传入的 Graph 对象，必须
  `AddGraphWithCopy`（v1 名；v2 中叫 AddGraphClone，note-5 §3、note-3 D6-4）。
- `CompileGraph` 失败必须 `RemoveGraph` 回滚，禁止半个图留在 session（ge.cpp:139-145）。
- graph_id 从 1 递增、用户自分配 uint32（note-3 D6）。
- 已知限制（TODO M7，ge.cpp:62-64）：`ge.exec.deviceId` 是全局量、钉死首个 device；
  多 device 支持需按 device 重构 GE session，当前未实现。

### 2.3 session 执行层

- 惰性加载：plan 首次执行前 `LoadGraph(graph_id, {}, stream)`；头注释"不支持重复加载"
  （note-5 §3），代码以 plan 状态机（COMPILED→LOADED）保证只加载一次。
- 执行接口 `ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs)`，
  gert::Tensor 零拷贝绑定（note-5 §3；gert::Tensor 为 POD，include/exe_graph/runtime/tensor.h）。
- 绑定契约（ggml-cannge.cpp:294-318）：地址 = ggml `t->data`（view 的偏移已含在指针内）、
  shape = `ge_shape(t)`、FORMAT_ND、placement kOnDeviceHbm。
- **输出-输入重叠（F1 修复后行为）**：绑定前对每个输出端口做地址区间重叠检测，命中则
  该端口绑 plan 自持的 staging buffer（每端口一个槽位、懒分配、plan 析构释放），
  执行后在同一 stream 上 `aclrtMemcpyAsync` D2D 写回真实地址。依据：未声明的输入/输出
  重叠是未校验 UB，910B 会静默丢弃写回（ge 仓 runtime/v2/kernel/common_kernel_impl/
  memory_copy.cc:574-579 TensorToOut src==dst @00ecb5c）；官方原地语义通道
  `ge.exec.outputReuseInputMemIndexes` 要求地址精确相等，offset view 不满足
  （ge 仓 docs/zh/design/features/memory_management.md:139-167 @00ecb5c）。详见 note-10 F1。
- 副作用写回：`io.writebacks`（为将来 GGML_OP_SET 预留），同样在图后同 stream D2D 拷回，
  当前 analyze 只收集不产生 writeback（SET 未注册进 supports_op）。

### 2.4 plan cache 层

- 键 = 结构签名（plan.cpp:214-269）：逐节点 op、flags、type、ne[4]、nb[4]、op_params
  原始字节、src 连接（按首次出现分配张量 id）。**禁止**使用 JittorInfer 的时间戳图名键
  （note-1 §7、note-2 §5：upstream 每次新建 cgraph 会退化为全量重编译）。
- 值 = `ggml_cannge_plan{state, signature, graph_id, mem_estimate, seq, staging}`，
  **禁止**持有任何 ggml 指针——compute context 可能在两次调用间被释放，plan 必须可跨
  context 存活（ggml-cannge.cpp:351-353、plan.h:71-74）。
- 状态机：CREATED → COMPILED → LOADED；FAILED 为终态。**编译/加载失败缓存为 FAILED
  plan**（避免同一签名反复触发秒级编译），构建失败不缓存（下次重试）。
- 逐出：`plans.size() >= 64` 时按 seq 最老逐出；逐出前 `aclrtSynchronizeStream` 排干在途
  异步任务，再 `RemoveGraph` 释放 device 侧图。HBM 预算驱动的逐出按 note-3 D6-1/D6-2
  仍为后续工作（代码注释 "HBM budget based eviction per D6 comes later"）。

## 3. GE 生命周期与异步安全（规范性约定）

- 初始化顺序：registry 首次注册时 `aclInit`（一次）→ 首个 plan 编译时 `GEInitialize`
  （引用计数）→ `new ge::Session`。**销毁顺序必须**为：sync stream → delete session →
  `GEFinalize`（ge.cpp:114-121；note-3 D6、note-5 §3 整改#6 约定，严于官方明文）。
- `graph_compute` 全程**禁止**内部同步：执行、staging 写回、副作用写回都是同 stream 上的
  异步 enqueue，顺序由 stream 保证；同步由 ggml 调 `ggml_backend_synchronize` 完成
  （ggml-cannge.cpp:492-507 注释）。
- 唯一允许的主动同步点：backend context 析构、plan 逐出（RemoveGraph 前）、
  session 销毁前——三者都必须先 `aclrtSynchronizeStream`。
- GEInitialize 失败的典型环境原因（远程）：缺 python 依赖导致 InitCannKB import 失败，
  处置见 note-8 §3。

## 4. 整图路径 env 门（GGML_CANNGE_GRAPH）

- `GGML_CANNGE_GRAPH=1` 才启用整图路径；环境变量进程内只读一次（static 缓存）。
  默认 off：supports_op 对所有 op 返回 false（含 OP_NONE），调度器不会给 CANNGE 分任何
  图；graph_compute 亦在入口直接拒绝（ggml-cannge.cpp:286-292, 329-332, 593-595）。
- 含义：默认形态下 CANNGE 是"零存在感"后端，与 GGML_CANN/CPU 共存无干扰；实验整图路径
  必须显式开 env。开 env 后拒绝粒度仍是逐 op 的（L0 门），不满足的图由调度器切走。

## 5. 与 JittorInfer 参考实现的差异（设计要点）

CANNGE 相对 JittorInfer GE/ES 路径的结构性修正（依据 note-1 §7、note-2 §9）：

| JittorInfer 做法 | CANNGE 做法 |
|---|---|
| 时间戳图名作缓存键 | 结构签名（§2.4） |
| 构图时绑死裸指针、命中缓存空转 | 每次执行按当前 `t->data` 重建 gert::Tensor（§2.3） |
| 私有 cgraph flags 魔数（bit0/bit1） | 无；默认 off + env 门 + 显式状态机 |
| 构建期强制依赖 OPP 自定义算子 | 无自定义算子（note-3 D2，hold） |
| 不支持即 GGML_ABORT | 返回 false / GGML_STATUS_FAILED，交上层 |
| AddGraph 裸调（GE 改传入对象） | AddGraphWithCopy |
| `ge.exec.deviceId` 单 session 全局 | 同（多 device 重构为 TODO M7） |

## 6. 待核实清单

| # | 条目 | 现状 |
|---|---|---|
| 1 | `ge.exec.reuseZeroCopyMemory=1` 是否为 310 实测最优（代码注释自标 re-tune） | 待 310 复测（note-7 §8 已列实测清单） |
| 2 | HBM 预算逐出（D6-2）与 `ge.exec.staticMemoryPolicy=2` 的启用决策 | 列入 310 实测清单，未落地 |
