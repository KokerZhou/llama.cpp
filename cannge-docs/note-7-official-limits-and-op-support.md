# note-7: CANN 官网文档调研——图资源约束、session 配置、算子支持矩阵

来源：昇腾官网 CANN 9.1.0 文档树（个别页面经 9.1.0-beta.3 镜像交叉核对）+
gitee.com/ascend/cann-ops 开源规格（与官网 9.x 同源）。整理日期：2026-09-10。
"未找到官方依据"处明确标注，勿当结论引用。

## 1. 算子支持矩阵的"宪法"依据

1. **"如果原型中有算子不在该手册中，则说明当前产品型号不支持该算子"**（Ascend IR 算子
   规格说明·规格清单，CANN 9.1.0，分产品附件：Atlas 推理系列=310P、Atlas A2=910B）：
   https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/aolapi/operatorlist_00095.html
   ——这是 D3"编译时静态表"的最直接官方依据：GE 图编译按 Ascend IR 原型 + 分产品清单判定，
   **不在清单 = 该 SoC 不支持，且无算子级自动分解承诺**。
2. 规格简介（含 Type Promotion 规则、确定性计算清单）：
   https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/aolapi/operatorlist_00094.html
3. Ascend Graph 开发指南：构图支持算子以规格说明为准，"如果不支持或不满足实际需要"只能
   走自定义算子（自定义算子入图流程 atlasgraphug_24_0007）——支持 L5 自定义算子作为官方
   认可的最后手段（当前 hold）。
4. aclnn 算子库简介："对于算子文档中未声明支持的场景（产品型号、数据类型、数据格式、
   数据维度等），不推荐开发者使用，当前版本不保证算子调用效果"。
   这条同时是 **L2 运行时黑名单**的官方依据：文档未声明 = 不保证，失败后按 (SoC, op) 拉黑
   并降级是符合官方口径的策略。
5. Ascend IR 算子头文件官方路径 `${INSTALL_DIR}/opp/built-in/op_graph/inc`，与容器内头文件
   互为印证；分产品逐 dtype 矩阵可进一步查容器内 opp 规格。

## 2. 图资源约束（D6 的官方答案）

- **每 Session/每进程最大图数量：未找到官方依据**（AddGraph 页只规定 graph_id 唯一、
  同 Graph 对象重复注册共享对象）。图数量上限只能靠 310 实测 + HBM 预算软限制。
- **图内存介质（HBM 还是 DDR）：未找到 GE 文档明确表述**。间接证据：异步执行示例中
  输入/输出/feature 内存由用户 aclrtMalloc（Device 内存）自行申请；CompileGraph +
  GetCompiledGraphSummary 可查"模型执行所需内存资源大小及内存是否可刷新、复用"。
  实践含义：加载的图占 device 内存至 RemoveGraph，plan 缓存必须有界（我们的 LRU+驱逐
  设计成立），上限自定。
- **ge.exec.staticMemoryPolicy**（官网内存管理页，9.1.0）：
  - 0（默认）动态分配，按实际大小分配，不支持扩展。
  - 2 静态 shape 内存动态扩展：**同 Session 多图内存复用，以最大图所需内存分配**；当前执行
    图所需内存超过前一张图时，释放前一张图内存并按当前图重新分配。
  - 3 仅动态 shape 支持内存扩展（省内存，可能有性能损失）。
  - 4 静态+动态 shape 同时支持扩展。
  - **官方约束：多张图并发执行时不支持配置 2 和 4**。配置 1 按 2 处理。
  - 对 CANNGE：默认 0；若确认同 session 内严格串行执行，2 可显著省 HBM（按最大图分配）。
    结论修正 note-3：staticMemoryPolicy=2 的启用前提是"同 session 多图串行"，列入 310 实测清单。
- 相关配置（官网内存管理页）：ge.exec.disableReuseMemory（0 默认开内存复用）、
  ge.exec.atomicCleanPolicy、ge.exec.inputReuseMemIndexes/outputReuseMemIndexes/
  outputReuseInputMemIndexes（graph 级 IO 内存复用声明）、ge.externalWeight（0/1/2，同 Session
  多模型权重复用，落盘优先级 externalWeightDir > ASCEND_WORK_PATH > 当前目录）、
  ge.constLifecycle（session=graph 同名常量内存复用）、ge.featureBaseRefreshable（静态 shape
  feature 地址可刷新）、ge.exec.input_fusion_size（Host->Device 合并拷贝阈值 128KB，仅
  RunGraphAsync 场景生效）。
- **ge.graphMaxParallelModeNum... 实为 ge.graphMaxParallelModelNum**：同一图在同一 device
  可并行加载执行的模型数，1~INT32_MAX，默认 8。

## 3. 精度策略（310P FP32 的官方语义）

来源：ATC/AOE --precision_mode 官方页（CANN 9.0.0 商用版，8.x 起同文）：
https://www.hiascend.com/document/detail/zh/canncommercial/900/devaids/atctool/atlasatcparam_16_0068.html

- `force_fp16`（ATC 默认）：原图 fp16/bf16/fp32 一律强制 fp16。
- `allow_fp32_to_fp16`（**我们 GEInitialize 的取值，但注意在线推理默认是 force_fp16**）：
  - 矩阵类算子：fp32 优先降 fp16；AI Core 不支持 fp16 则选 fp32；AI Core 不支持 fp32 则落到
    AI CPU；AI CPU 也不支持则**执行报错**。
  - 矢量类算子：优先保持原图精度，支持 fp32 则保留，不支持则降 fp16。
- precision_mode_v2 取值：fp16（在线推理默认）/origin/cube_fp16in_fp32out/mixed_float16/
  mixed_bfloat16/mixed_hif8/cube_hif8。官方建议用 v2，与 v1 互斥。
- MatMul 佐证（cann-ops 开源规格）：310P 上 FP32 输入 MatMul **不能** keep dtype，必然
  FP16 计算；910B 则 HF32。
- **官方语义是"优先/回退"策略，不承诺精度等价**。对 CANNGE 的含义：F32 图在 310P 上数值
  容差必须离线验证（纳入 CI 的想法成立）；支持判定时 FP32 MATMUL 在 310P 应标"支持但内部
  降精度"。
- "310P 整图级 FP32 策略"逐字官方表述：未找到官方依据。

## 4. Broadcasting 与 MatMul 方向

- dtype 层面：Add/Mul 等输入 dtype 不一致时算子内部自动 Type Promotion（规格简介含完整
  提升表；fp32xdouble 无法提升则落 AI CPU）。
- shape 层面：TBE 广播规则官方定义"每个维度大小与目标 shape 相等或为 1 即可扩展"
  （TBE&AI CPU 自定义算子开发指南）。GE 图模式 Add/Mul 自动广播、无需显式 ExpandDims——
  间接官方依据，GE Add 算子规格广播原文未直接抓取（置信中高）。图融合规则另有限制
  （如 MatMul 融合仅支持 batch 轴广播）。
- MatMul 方向：`y = x1 @ x2`，最后两维矩阵乘、前导维 broadcast；transpose_a/transpose_b
  控制**乘前**转置，`A^T x B` 即 transpose_a=True。BatchMatMul 仅 3 维，batch 轴可 broadcast。

## 5. 算子支持情况表（阶段 3 静态表初稿）

产品口径：310P = Atlas 推理系列；910B = Atlas A2。dtype 依据 cann-ops 开源规格。

| 算子 | GE 原生 | 310P dtype 限制 | 910B dtype 限制 | 置信度 |
|---|---|---|---|---|
| Add/Sub/Mul/Div | Ascend IR 原生，dtype 不一致自动 Type Promotion | 多 dtype；fp32 矢量算子按算子逐个查规格附件 | 同左，另支持 bf16 | 原生性高 |
| MatMul (V3) | 原生 | FP16/FP32；FP32 必降 FP16 计算 | BF16/FP16/FP32（FP32->HF32） | 高 |
| BatchMatMul (V3) | 原生，3 维 batch 可 broadcast | 同 MatMul | 同 MatMul | 高 |
| Softmax (V2) | 原生 | 逐 dtype 查规格附件 | 同左 | 原生性高 |
| RmsNorm | 原生（es_nn，x/gamma: bf16/fp16/fp32） | 支持 | 支持 | 高 |
| LayerNorm | 原生 | 支持 | 支持 | 高 |
| Silu | aclnn NN 库原生；GE 层无独立 Silu | 用 Swish(scale=1) 等价 | 同左 | 中 |
| Swish | 原生 | fp16/fp32 | 同左 | 高 |
| Cast | 原生 | 支持 | 支持 | 高 |
| Reshape/Transpose/Permute/Concat/Slice/Split/Squeeze/ExpandDims | 全部原生 | 支持 | 支持 | 高 |
| GatherV2 / Embedding | 原生 | 支持 | 支持 | 高 |

遗留：分产品规格附件直链未抓到，各算子在 310P 的完整 dtype 矩阵建议直接在容器
`/usr/local/Ascend/cann-9.1.0/opp/built-in/op_graph/inc` 及各算子 .ini 规格核对后填入
supports_op 静态表。

## 6. 异步执行与销毁的同步要求（官网原文要点）

- ExecuteGraphWithStreamAsync（07_0100，9.1.0）：调用前需完成 CompileGraph 与
  GeSessionLoadGraph、aclrtCreateStream 建流；"**得到输出运行结果前，需要通过
  aclrtSynchronizeStream 接口保证 Stream 上的任务已经执行完**"。
- RunGraphWithStreamAsync 异步流程官方 10 步（编程指南 atlasag_25_0035，9.1.0）：第 9 步
  aclrtSynchronizeStream 阻塞至该流任务完成，然后 D2H；释放顺序 aclrtFree/FreeHost ->
  GEFinalizeV2 -> aclFinalize。
- GEFinalize（07_0087，9.1.0）：接口内默认 2000ms 延时用于 Device 业务日志回传，可用
  环境变量 ASCEND_LOG_DEVICE_FLUSH_TIMEOUT=0 去除。
- "RemoveGraph/Session 析构/Finalize 与在途异步任务的显式同步约束"：官网 API 页均写
  "无"——同步要求只体现在上述示例流程的 synchronize 步骤。**我们的严格约定（整改#6）比
  官方明文更严，保留**。
- RunGraphAsync 回调 status==SUCCESS 时数据可用（唯一 GE 侧自带完成语义的异步接口）。

## 7. "整图编译失败"类官方表述

- 未找到"部分算子在某些芯片不支持导致整图编译失败"的逐字原句；等价依据是 §1 第 1、4 条
  （不在清单=不支持；未声明场景不保证效果）。
- 故障码佐证：ATC 故障处理含 **E19010 Unsupported Operator**；模型含 AI CPU 不支持 dtype
  的算子会编译失败。这直接支撑 D1 回退阶梯：编译失败 -> 定位失败算子 -> L1b 同义变体 /
  L2 分解 / L3 分段降级，而不是 abort。

## 8. 对既有设计决策（note-3）的修正汇总

| 项 | 原决策 | 修正 |
|---|---|---|
| staticMemoryPolicy=2 | 直接列入启用 | 仅当同 session 多图**串行**执行时可用（并发不支持 2/4），列入 310 实测清单 |
| RMS_NORM | ES 分解起步 | 改用 es_nn 原生 RmsNorm（note-6），分解仅作语义对照 |
| SILU | 未决 | 用 Swish(scale=1) |
| precision_mode | allow_fp32_to_fp16 | 保持；但记录在线推理默认实为 force_fp16，310 实测对比两种取值 |
| 图数量上限 | HBM 预算软限制 | 维持，官方无上限依据；GetCompiledGraphSummary 可用于查询每图内存占用 |
| session_device_id | 已实现 | 键名已修正为 ge.session_device_id |
