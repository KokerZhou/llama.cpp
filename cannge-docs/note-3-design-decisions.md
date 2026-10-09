# CANNGE 研读笔记三：设计决策记录

> 本文记录开发过程中的关键设计决策及理由，随讨论更新。当前状态：2026-09-10。

## D1. 回退阶梯（fallback ladder，细粒度版 2026-09-10 定稿）

"回退"是三个正交选择轴的笛卡尔积，按优先级在积空间选点：

| 轴 | 取值 |
|---|---|
| A. 节点实现（某 ggml op 怎么落地） | A1 原生 GE 算子（含同义变体） / A2 ES 原语分解 / A3 自定义算子(hold，见 D2) |
| B. 图拓扑 | B1 整图一张 / B2 分段（>=2 张 GE 图） |
| C. 执行引擎 | C1 GE 编译产物 / C2 eager aclnn / C3 CPU |

**细粒度层级总表**：

| 级 | 名称 | 轴取值 | 触发条件 | 判定依据 | 状态变更 | 主要代价 | 失败去向 |
|---|---|---|---|---|---|---|---|
| L0 | 静态能力门控 | （元层） | 调度器问 supports_op / 构建 plan 签名 | per-SoC 静态文档表 x (op, dtype, shape类, 布局, broadcast) | 无 | 误拒（文档保守->能力浪费） | - |
| L1a | GE 整图·原生 | A1xB1xC1 | 所有节点静态支持 | 静态表全绿 | plan 缓存 +1 | 无（最优） | L1b / L2 |
| L1b | GE 整图·同义变体 | A1'xB1xC1 | 首选 GE 算子本 SoC 不可用但有同义算子（如 attention 的 FIA/PFA 变体、matmul 变体） | 算子变体表 per SoC | 同 L1a | 属性适配轻微开销 | L2 |
| L2 | GE 整图·ES 分解 | A2xB1xC1 | 原生缺失；分解式所有成分静态支持 | 分解策略表 + 成分递归走 L0 | 分解选择入签名 | 节点变子图（编译器可能再融合）；fp16 舍入差异 | L3 |
| L3 | GE 分段（段内 L1/L2） | A1/A2xB2xC1 | 整图 CompileGraph 失败且可定位/怀疑具体节点 | 编译错误解析（尽力）+ 黑名单 | 段 plan 各自缓存；分割点入签名 | 段边界 device buffer 往返；跨段融合丢失；每段一次 Compile | L4 |
| L4a | eager aclnn 单算子 | A1xB2xC2 | 段间填补节点有原生 aclnn | aclnn 可用性（静态表精神，aclnn 支持集 != GE 支持集） | 黑名单/降级记录 | 每步 kernel 启动开销 | L4b |
| L4b | eager aclnn 等价组合 | A2xC2 | L4a 不可用，组合成分全可用 | eager 组合策略表 | 同 L4a | 中间 buffer 访存；启动开销 xk | L5(不可用)/L6 |
| L5 | 自定义 AscendC | A3xB1xC1 | **hold**：仅当测量证明 b/d 都不够 | 人工决策 | - | 工程成本 | - |
| L6 | CPU 兜底 | xxC3 | 后端整体拒绝（supports_op=false） | 调度器 | 调度器 split | H2D/D2H 拷贝；图彻底碎片化 | - |

**层间转移控制**：

| 转移 | 触发事件 | 决策数据 | 谁决策 | 可逆性 |
|---|---|---|---|---|
| L1->L2 | 静态表：首选算子不支持 | 变体表->分解表 | 后端（构图期） | 可（改表即回） |
| L2->L3 | CompileGraph 返回错误 | 错误码/日志解析；失败节点候选集；黑名单 | 后端（编译期） | 可（新 plan） |
| L3->L4 | 段内节点无 GE 落地 | eager 组合表 | 后端（编译期） | 可 |
| L4->L6 | aclnn 也不可用 | 静态表 | 后端撤回 supports_op -> 调度器重切 | 可 |
| 任意->L6（图外） | 后端对整图 supports 失败 | 调度器 split 算法 | **调度器** | 可 |
| 数值回退（横切） | 降级实现过 test-backend-ops 容差失败 | 容差标注 | 后端（验证期） | - |

**决策主体三分**：构图期选择（后端静态）、编译期降级（后端动态）、图级拒绝（调度器）。

**横切控制变量**：SoC 身份（所有表的共同键）、静态能力表（L0 唯一权威）、算子变体表（L1b）、分解策略表（L2/L4b）、黑名单（运行时学习，per-device 持久化）、plan 签名（结构签名+SoC+实现选择，变体/分解/分段点都是签名一部分）、数值容差（每级标注精度风险）、性能预算（降级预估代价，供"留本后端降级 vs 切 CPU"比较）。

**时间线**：构建时固化静态表/变体表/分解表（无硬件可做）；容器编译验证只做语法/链接；**310 首次加载是能力学习真正发生的时刻**（worst-case 图逐级尝试，秒~分钟级一次性，结果固化为 plan 缓存+黑名单）；稳态 decode 纯执行+重绑，零降级开销。=> 黑名单/plan 缓存必须持久化。

**遗留问题（含 D6 后的裁决）**：
0. TODO(needs_staging)：非 dense IO（如 PERMUTE 收尾的边界输出，ggml nb 非稠密）当前在 CANNGE build 期确定性失败回退 CPU；待 staging 拷贝 + writeback 机制落地后支持（plan.h needs_staging 处同步标注）。
1. ~~L3 段数上限~~ -> 由 HBM 预算动态决定（见 D6-2）：加载新 plan 前内存预算检查，不足先驱逐冷 plan，仍不足放弃分段直降 L4。
2. L4 填补节点与 GE 段的数据流：填补算子输入输出必须落在 GE 段已绑定的 device buffer（zero-copy 衔接）-> 分段时边界张量显式导出，与"只导出真边界输出"原则打通。
3. 黑名单粒度：建议 per (SoC, op)，shape 相关失败进 plan 签名处理。
4. 数值容差执行方：分解式上线前用 test-backend-ops 离线验证（进 CI），运行时不重复检查。

## D2. 自定义算子（手写 AscendC kernel）：暂时 hold

**决策**：阶段 1-3 不实现任何手写算子；仅在 CMake 预留可选接入通道（开关 + 运行期检测 vendor 包）。

**理由**：
- 设备强相关：OPP vendor 包是 per-SoC 架构编译的二进制，兼容性差，违背"后端可独立编译"目标；
- 开发代价高：proto + tiling + kernel + 打包四件套；
- 现有需求可被 b（ES 分解）和 d（eager 组合）覆盖；rope 之类的性能瓶颈待真机测量后再定（justified by measurement）。

**遗留问题**：预留通道放在哪一层（图内节点 vs 图外 eager 插件）待商榷，暂不决定。

**事实备查**：自定义算子是 GE 图内节点，不是运行时回退层。GE 可对其做算子间调度/内存规划/边界融合（融合对自定义 op 默认保守，fusion 信息需额外注册），kernel 内部不受 GE 优化。JittorInfer 的 `RopeExtCustomV2` 即此类（且其为构建期强制依赖，我们不照搬）。

## D3. 算子能力判定：两层，静态为主

**决策**：op 是否受支持采用两层判定，以编译时静态层为准。

**L1 编译时静态表（权威层）**：
- 数据来源：官方文档的 per-SoC op 支持矩阵（CANN 各版本文档明确各 op 在 310P/910B 等上的支持情况）；
- 形态：构建期固化的能力表（按 SoC 版本参数化）；
- 语义：**文档说支持即支持**，不做运行时动态探测；文档未提及/存疑的按不支持处理。

**L2 运行时观测层（被动兜底，非主动 probe）**：
- 仅处理静态表未覆盖的例外：真实模型图上 CompileGraph/执行失败时，定位（尽力而为）→ 降级（D1 阶梯）→ 将结果持久化到 per-device 黑名单；
- 不主动做小样例图 probe 扫描（曾考虑过的 ge_check_op.json 自动扫描方案取消）；
- 与 plan 缓存绑定：黑名单随 SoC 身份持久化，同签名图下次直接走低级策略。

## D4. ES 与 GE 的关系（事实确认）

ES（`ge::es::EsGraphBuilder`）是 GE 的构图前端 API，产物与 raw GE API 同为 `ge::Graph` IR，进同一条 `AddGraph → CompileGraph` 链路。"能否被 GE 优化"对两者完全等价（融合/tiling/内存规划一视同仁）。选 ES 只是选写法。raw GE 通道（`GetCGraphBuilder`）保留为 ES 表达不了时的适配器。

## D5. 310P 事实备查

- llama.cpp 的 `USE_ACL_GRAPH`（capture/replay，非 GE 编译）在 310P 上被 CMake 显式禁用（`ggml/src/ggml-cann/CMakeLists.txt` FATAL_ERROR）；其实现为 `aclmdlRICaptureBegin/End + aclmdlRIExecuteAsync`，录制 eager kernel 序列原样回放，无编译器级融合。
- JittorInfer 整图 GE 已在 310 上验证（DeepSeek-V2-Lite 测试）。
- CANNGE 第一目标 310；910B3 远程环境仅做编译与 API 兼容确认，不作为 310 运行时证据。

## D6. 图资源模型（硬件约束，2026-09-10 文档核实）

**ACL Graph 与 GE 是两个独立子系统**（勿混）：
- ACL Graph = 运行时任务捕获/回放（`aclmdlRICaptureBegin/End` → 模型运行实例 `aclmdlRI` → `aclmdlRIExecuteAsync` 重放）。无编译器。约束：捕获期间 device 内存不变直至实例销毁、捕获中禁止 malloc/memcpy、消耗 stream 资源。试验特性，部分版本文档注明"不支持商用"，且部分版本产品支持不含 Atlas 推理系列（310P）。
- GE = 编译器+运行时（AddGraph/CompileGraph/LoadGraph/Execute）；ES 是 GE 的 C++ 构图前端（见 D4）。llama.cpp 的 USE_ACL_GRAPH 属前者，与本后端无关。

**GE 图的存储位置与数量约束**（昇腾社区文档）：
- Graph IR 与编译产物在 host 侧；`LoadGraph` 后**权重+工作内存占用 device HBM** 直至 RemoveGraph/session 销毁。→ **每张已加载图 = 一份 HBM 占用**。
- `graph_memory_max_size + variable_memory_max_size` 芯片相关（910 系 ≤31GB 默认 26GB；310P HBM 更小，约束更紧）。`ge.exec.staticMemoryPolicy=2`：同 session 多图按最大图复用内存。
- session 独占资源、不支持并发执行；多 session 各绑 device。
- **文档无硬性图张数上限**（graph_id 为用户自分配 uint32）；实际约束是 HBM + session 资源，上限内存驱动。

**对设计的修正**：
1. plan 缓存 = HBM 预算内的有界 LRU，**显式驱逐（RemoveGraph）为必需机制**，非可选。
2. L3 分段数上限由 HBM 预算决定（非固定常数）：加载新 plan 前做内存预算检查，不足先驱逐冷 plan，仍不足则放弃分段直接降 L4。
3. `ge.exec.staticMemoryPolicy=2`（同 session 多图内存复用）列入 310 实测评估清单，对 decode/prompt/分段图共存场景关键。
4. AddGraph 会直接修改传入的 Graph 对象；需要保留原对象时用 AddGraphClone（构图失败重试路径要注意）。
