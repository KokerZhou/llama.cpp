# note-14: CANNGE 算子接入规范

建立：2026-09-11。规范对象：向 CANNGE 后端新增/修改 ggml op 支持的流程与契约。
代码基线同 note-18（d85b6bd08 + F1/F2/F3 未 commit 修复）。出处分级引用遵循 note-11
的文档来源注册表（T1 官网 / T2 cann-ops 镜像 / T3 ge 源码仓 path:line@commit /
T3' 容器头文件 / T4 ops-info 配置）。

## 1. 总则

1. **禁止 abort**：任何"不支持"的情形必须表现为 supports_op 返回 false 或构建返回
   false，交上层调度器处理（note-3 D1、note-4 §2 的既定差异）。
2. op mapping 与每个非平凡的 GE 语义决策，代码注释必须带一行官方出处，格式三选一
   （note-11 §3）：
   - 官网：`// spec: <https://...aclnnXxx.md>`
   - 源码：`// ge 仓 <path>:<line> @00ecb5c`
   - 头文件：`// include/es/es_Xxx.h`
   注释保持 1-2 行（llama.cpp AGENTS.md 规范），出处附在语义句后，不占额外行。
3. T4-T6 来源（ops-info 配置、JittorInfer、.so 字符串）的结论若要进代码，必须先升级到
   T1-T3 依据；升级不了的，note-10 标 open item，注释写"未文档化，依据见 note-10 F#"
   （note-11 §3 第 3 条；现例：MUL_MAT 32B 对齐，ggml-cannge.cpp:658-664）。
4. 注释一律 ASCII、精简（llama.cpp AGENTS.md）。

## 2. 两层支持判定（note-3 D3）

### 2.1 L1 编译时静态表（权威层）

- 数据来源：官方文档的 per-SoC op 支持矩阵——"如果原型中有算子不在该手册中，则说明当前
  产品型号不支持该算子"（T1，Ascend IR 算子规格说明·规格清单 operatorlist_00095，
  note-7 §1）。
- **ops-info 配置（T4）不得作为"某 SoC 是否支持某算子"的判定依据**：逐目录核对证实它是
  TBE ai_core 实现注册表而非支持矩阵（910b 目录无 Add 等条目但 910B 实测全通过；
  Permute 16 目录零条目但 310P 实测通过），只在**有条目**时提供实现层 dtype/format/attr
  默认值细化（note-19 §1 三条硬证据、§6 落地建议）。
- **必须**在编译时（无硬件）固化判定结果；"文档说支持即支持、未提及按不支持"，
  **禁止**运行时动态探测（D3：曾考虑的 ge_check_op.json 自动扫描方案已取消）。
- 当前实现形态：判定逻辑直接编码在 `supports_op()` 的静态分支中（ggml-cannge.cpp:590-727），
  即"代码即静态表"；后续 op 增多后应当保持同样的保守方向。

### 2.2 L2 运行时观测层（被动兜底）

- 仅处理静态表未覆盖的例外：真实图上 CompileGraph/执行失败 → 尽力定位 → 按 note-15
  回退阶梯降级 → 结果持久化到 per-device 黑名单（粒度 per (SoC, op)，note-3 D1 遗留#3）。
- **禁止**主动小样例图 probe。
- 与 plan cache 绑定：黑名单随 SoC 身份持久化，同签名图下次直接走低级策略（D3）。
- 当前状态：黑名单机制未实现（代码只有 FAILED plan 缓存，见 note-18 §2.4），属预留。

### 2.3 supports_op 的保守校验契约

supports_op **必须**镜像构建期前置条件（note-10 F3 的修正方向；F1/F2/F3 修复后已部分
落地）。当前逐 op 的判定规则以代码为准，约束汇总见 §4。通用规则：

- dtype 白名单：F16/F32（`ggml_cannge_float_dtype`），I32 仅作 GET_ROWS 索引；
  映射函数 `ggml_cannge_ge_dtype`（graph-build.h:37-51）只支持 F16/F32/I32。
- shape 校验：elementwise 逐维"相等或为 1"（广播，T1/T3' 原文 "Support broadcasting
  operations"，elewise_calculation_ops.h CANN 9.1.0，note-10 F3）；matmul 收缩维严格
  相等（T2 aclnnMatmul.md：mat2 的 Reduce 维须与 self 相等）。
- 对齐校验：MUL_MAT 的 M/N/K 必须 32 字节对齐（`align_elems = 32/dtype_size`，
  F16→16、F32→8 元素），**产品无关**地保守拒绝（310P 与 910B 同样触发
  CheckDimsAligned/EZ9999；该约束不在 aclnn spec 中，属 GE 图编译 tiling 路径约束，
  "未文档化，依据见 note-10 F3"，spec 行引用 aclnnMatmul.md 作 dtype 依据）。
- layout 校验：VIEW 要求 `ggml_is_contiguous`；CONT 要求 src0 连续；CPY 要求 dst 非
  view 且连续；非 dense IO 由 analyze 置 `needs_staging`，构建期确定性失败（当前未实现
  staging，见 §5）。
- 禁止 in-place 变体（`op->src[i] == op`，任一 src）——已在 supports_op 通用段显式拒绝
  （ggml-cannge.cpp:602-608，2026-09-11 落地；in-place 节点上游无过滤、确实到达
  supports_op，拒绝理由是 GE 静默丢弃别名写回，note-10 F1）。

## 3. 特殊 op 的注册规则（plan.cpp analyze 的 produced 语义）

- **produced 集合**：每个节点产出自己的输出张量；**CPY 节点额外产出其写目的地
  `src[1]`**（F1 修复引入，plan.cpp:73-84）。把 CPY dst 注册为图输入会把 dst 变成
  只读外部缓冲，cast 输出端口与输入地址重叠被 GE 静默丢弃（910B 实测，note-10 F1）。
- 输入扫描必须跳过 `GGML_OP_CPY && j==1`（plan.cpp:104-115）。
- **边界输出**：无消费者或被 `GGML_TENSOR_FLAG_OUTPUT` 标记的节点。别名外部输入的
  **dense** view 不需要 GE 输出端口（消费方按 view 步长直接读输入缓冲）；**非 dense**
  view（PERMUTE/TRANSPOSE 步长）必须由 GE 物化，进输出端口（F1 修复引入的
  `is_dense` 条件，plan.cpp:134-147）。
- **GE 输出端口顺序**：同 dtype 且目的地独立的 CPY 在前（节点序），其后边界输出，
  去重（plan.cpp:165-185）。构建侧 `SetOutput` 必须按同序（graph-build.cpp:371-379），
  新增 op 时**禁止**破坏该顺序契约。
- 图输入顺序 = 首次引用顺序；构建侧 `CreateInput` 按下标绑定（graph-build.cpp:73-80），
  执行侧按同序重建 gert::Tensor（ggml-cannge.cpp:428-434）。两侧顺序必须一致。

## 4. 当前已支持 op 清单（以代码为准，修复后状态）

dtype 约束中"浮点"= F16/F32。"→"后为 ES 映射与出处。

| ggml op | 支持约束（supports_op） | ES 映射（graph-build.cpp） | 出处/备注 |
|---|---|---|---|
| ADD / SUB / MUL / DIV | 两 src 均浮点；逐维相等或为 1（广播） | `Add`/`Sub`/`Mul`/`Div` | es_math；广播 T1/T3' "Support broadcasting operations"（note-10 F3） |
| SCALE | src0 浮点 | `Mul(es0, CreateScalar(s))`，s 读 op_params | es_math |
| SOFT_MAX | 单 src 浮点；**仅无参形态**（scale==1.0 且 max_bias==0.0，读 op_params） | `SoftmaxV2(es0, {-1})` | es_math；参数化 softmax（mask/scale）预留 |
| UNARY(SILU) | 仅 SILU、浮点 | `Swish(es0, 1.0f)`（silu==swish beta=1） | es_nn/es_Swish.h（note-6 §5） |
| RMS_NORM | src0 浮点 | **元语分解**：`x * rsqrt(ReduceMean(Square(x), axes={nd-1}, keepdims=true) + eps)`，axes 常量用 `CreateVector(axes)` | F2 修复；ge 仓 all_ops.cpp:39019-39048 @00ecb5c + include/es/es_ReduceMean.h。融合 RmsNorm 核因 310P3 小 shape 间歇丢行弃用（note-8 坑表） |
| MUL_MAT | 两 src 同 dtype 浮点；收缩维 ne0 相等；M/N/K 32B 对齐；**仅 2D** | `BatchMatMulV3(es1, es0, nullptr, nullptr, false, true)` | MatMulV3 无 infer_datatype 注册（note-8 坑表），JittorInfer 同款映射（T5 对照）。精度语义见 note-16 |
| RESHAPE | src0 浮点 | `Reshape(es0, ge_shape(node))` | es_math |
| PERMUTE | src0 浮点 | `Permute(es0, order)`；ggml 是 move-dim 语义，先求逆再翻轴序 | es_math（note-6 §5：perm 用 C++ 属性版） |
| TRANSPOSE | src0 浮点 | `Transpose(es0, perm)`（交换最内两 ggml 维 = 最外两 GE 轴） | es_math |
| VIEW | src0 浮点且 view 自身连续 | 全等 alias；否则按元素偏移分解 `Slice(base, offsets, sizes)`（越界回绕/偏移越界即构建失败）；nd 变化再 `Reshape` | es_math；ggml-cannge.cpp:678-682 |
| CONT | src0 浮点且**src0**连续 | 纯 alias，无数据搬运 | graph-build.cpp:292-301 |
| CPY | 两 src 浮点、同 shape；**dst（src1）非 view 且连续**；同 dtype 写入 view 的别名拷贝暂拒 | 同 dtype：alias（dst 独立缓冲经 analyze 强制 GE 输出端口）；异 dtype：`Cast(es0, dt)` | F1 修复配套（§3） |
| GET_ROWS | src0 浮点；src1 必须是 I32 | `GatherV2(es0, es1, ndims-2)` | es_nn/es_GatherV2.h（note-6） |
| CONCAT | 同 dtype 浮点；op_params 轴合法；其余维相等 | `Concat(CreateScalar(ge_axis), {es0, es1}, 2)`，ge_axis = nd-1-ggml_axis | es_math |

其他 op（含 GGML_OP_SET）：一律 false。空张量（ggml_is_empty）与 OP_NONE 叶子：接受/跳过
（graph_compute 预检与 analyze 均跳过空张量）。

## 5. 已知限制与预留（接入新 op 前必读）

- **needs_staging 未实现**：非 dense IO（如 PERMUTE 收尾的边界输出）当前在构建期确定性
  失败（"non-dense IO requires staging, not implemented yet"，graph-build.cpp:350-352）。
  staging 拷贝 + writeback 落地后才允许放开（note-3 D1 遗留#0、plan.h:63-67）。
- 上述清单外 op 的"接入"动作 = 三处同时修改：supports_op 分支（L0 门）、graph-build
  翻译分支、（若涉 IO 语义）plan.cpp analyze 规则；只改一处视为不完整提交。
- 新增映射的语义基准：优先 T1/T2 算子规格 + T3 IR 定义；JittorInfer 的 handler 仅作
  对照（T5，note-11），**禁止**以其行为作为出处。

## 6. 待核实清单

| # | 条目 | 现状 |
|---|---|---|
| 1 | ~~in-place 变体（`op->src[0] == op`）的显式拒绝未落代码（note-10 F3 遗留）~~ | 已实现（ggml-cannge.cpp:602-608，见 §2.3） |
| 2 | 黑名单（L2 运行时观测层）的持久化格式与位置未定 | 预留，未实现 |
| 3 | 参数化 SOFT_MAX、batched(>2D) MUL_MAT 的接入排期 | 代码注释标注 lands later，无时间表 |
