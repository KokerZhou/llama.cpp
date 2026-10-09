# note-19: CANNGE 逐 SoC 算子支持精细静态表（ops-info 全量取证）

建立：2026-09-11。纯静态取证（无硬件、无运行时探测），全部结论可在本机容器
（`quay.io/ascend/cann:9.1.0-310p-ubuntu22.04-py3.10-devel`）内复现。
出处分级遵循 note-11；本文是 note-17 待核实 #2（"16 SoC ops-info 矩阵尚未逐产品核对"）
的交付物，同时回答 note-10 F3 的遗留取证路径与 note-16 §7 的 #2/#3 两问。

分析对象：CANNGE 实际映射的全部 17 个 GE 算子（以
`llama.cpp/ggml/src/ggml-cannge/graph-build.cpp` 为准）：
Add / Sub / Mul / Div（es_math）、SoftmaxV2、Swish、Square / ReduceMean / Rsqrt
（RMS_NORM 元语）、BatchMatMulV3、Reshape、Permute、Transpose、Slice、Cast、
GatherV2、Concat。

## 1. ops-info 是什么：先纠偏，再使用

note-14 §2.1 把容器内 ops-info 配置（T4）预期为"分产品细化"的数据源。逐目录核对后的
**第一结论：ops-info 是 TBE ai_core 算子实现的注册表，不是算子支持矩阵**。"某 SoC 无某
算子的 ops-info 条目"**不能**推出"该 SoC 不支持该算子"。三条硬证据：

1. **Permute 在全部 16 个目录中零条目**，但 310P 实测 PERMUTE 探针通过
   （note-9 §2，310P3 x 9.1 轮 11/11）。
2. **ascend910b 目录没有 Add/Sub/Mul/Div/SoftmaxV2/Swish/Square/ReduceMean/Rsqrt/
   Reshape/Transpose/Slice/Cast/GatherV2/Concat 中任何一个的条目**（§3 矩阵），但 910B
   实测 ADD/RMS_NORM(元语)/SOFT_MAX/SILU/GET_ROWS/CONCAT 探针全通过
   （note-9 §2，910B3 x 9.1 轮），test-backend-ops CONCAT OK=4（note-16 §3）。
3. 反方向同样不严格：条目的 dtype 清单是"TBE 该实现的注册"，GE 图路径另有格式转换/
   精度回退通道（§5）。

因此 L1 静态表（note-14 §2.1）的主依据仍是 T1 operatorlist 分产品清单 + T3' IR 头；
ops-info 只在**有条目**时提供"实现层 dtype/format/attr 默认值"的细化证据（最完整的是
310p；950 次之）。本文以下矩阵的价值在于：把"哪些 SoC 有条目、条目的实现约束是什么"
固化下来，避免今后把"无条目"误读为"不支持"。

## 2. SoC 目录清单（16 目录，CANN 9.1.0 容器实测）

路径：`/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/config/<soc>/*.json`。
单文件为"算子名 -> 条目"的 JSON 对象；六类文件后缀 cv/legacy/math/nn/oam/transformer。
条目结构（见 §4 解剖）：`attr*`（属性+默认值）、`inputN`/`outputN`（dtype/format/name/
paramType/shape）、标志位（`dynamicFormat`/`dynamicShapeSupport`/`dynamicRankSupport`/
`needCheckSupport`/`precision_reduce`）、`slicePattern` 等。

| SoC 目录 | 随包文件 | 条目总数 |
|---|---|---|
| ascend310p | cv + **legacy** + math + nn + oam + transformer（最全） | 872 |
| ascend910b | cv + math + nn + oam | 264 |
| ascend910_93 | cv + math + nn + oam | 262 |
| ascend950 | cv + math + nn | 634 |
| kirin9030 | cv + math + nn | 141 |
| kirinx90 | cv + math + nn | 141 |
| ascend350 | nn | 12 |
| ascend910 | math + nn | 12 |
| ascend310b | cv + nn | 3（GridSample、UpsampleBicubic2d、RmsNormQuant） |
| ascend910_55 | nn | 2（AddRmsNormQuant、Conv3DTransposeV2） |
| mc62 | cv + nn | 4 |
| ascend031 / ascend035 / ascend610lite / ascend630 / mc62cm12a | **空目录** | 0 |

要点：
- **legacy 文件仅 ascend310p 随包**（794 条，CANNGE 用到的普通 math/数组算子几乎都在
  这里）；950 的 math（241 条）/nn（366 条）是新品里覆盖最广的，但仍无 legacy 集。
- 空目录的 5 个 SoC 不是"不支持"，而是本工具包版本未随包其 TBE 配置（§1 结论）。

## 3. 逐算子存在性矩阵（17 GE 算子 x 16 SoC）

图例：`L`=legacy 文件，`M`=math，`N`=nn；`-`=该目录无任何条目（**≠不支持**，§1）。

| GE 算子 | 310p | 910b | 950 | 910_93 | 350 | kirin9030 | kirinx90 | 其余 8 目录 |
|---|---|---|---|---|---|---|---|---|
| Add | L | - | M | - | - | - | - | 全 - |
| Sub | L | - | M | - | - | - | - | 全 - |
| Mul | L | - | M | - | - | - | - | 全 - |
| Div | L | - | M | - | - | - | - | 全 - |
| Square | L | - | M | - | - | - | - | 全 - |
| ReduceMean | L | - | M | - | - | - | - | 全 - |
| Rsqrt | L | - | M | - | - | - | - | 全 - |
| Transpose | L | - | M | - | - | - | - | 全 - |
| Slice | L | - | M | - | - | - | - | 全 - |
| Cast | L | - | M | - | - | - | - | 全 - |
| Concat | L | - | M | - | - | - | - | 全 - |
| Reshape | L | - | - | - | - | - | - | 全 - |
| SoftmaxV2 | L | - | N | - | - | - | - | 全 - |
| Swish | L | - | N | - | - | - | - | 全 - |
| GatherV2 | L | - | N | - | - | - | - | 全 - |
| BatchMatMulV3 | N | N | N | N | N | N | N | 全 - |
| Permute | - | - | - | - | - | - | - | 全 -（16/16 无条目） |

对照 CANNGE 两个目标 SoC 的实测（note-9）：310P 全 op 链路通过、910B 除 F1 别名写回外
全 op 链路通过——与"910b 目录大面积无条目"并存，再次印证 §1。

## 4. 代表算子条目解剖与跨 SoC 差异

### 4.1 BatchMatMulV3（MUL_MAT 的 GE 映射，nn 文件）

| 项 | ascend310p | ascend910b | ascend950 | kirin9030/x90 | ascend350 / 910_93 |
|---|---|---|---|---|---|
| x1/x2 dtype | **仅 float16** | f16+f32+bf16 | f16/f32/bf16 多组合 | 仅 float16（dynamicFormat=true） | 同 950（350）/ f16+f32+bf16（910_93） |
| y dtype | 仅 float16 | f16+f32/bf16→f32 | 多组合 | 仅 float16 | 同左 |
| format | x1: FRACTAL_NZ,ND；x2: 仅 FRACTAL_NZ | 全 ND | ND/FRACTAL_NZ 混合 | FRACTAL_NZ/ND | 950 混合；910_93 全 ND |
| enable_hf32 默认 | false | false | false | false | false |
| precision_reduce | 无 | 无 | **true** | true | 350 true；910_93 无 |
| shape | all | all | all | all | all |

出处（容器内行号，9.1.0）：310p 条目 `ascend310p/aic-ascend310p-ops-info-nn.json:896`
（attr 列表 :898、`attr_enable_hf32` 默认 false :908-912、input0 dtype "float16,float16"
:937-944、x2 format "FRACTAL_NZ,FRACTAL_NZ" :946-953）；910b 条目
`ascend910b/aic-ascend910b-ops-info-nn.json:2944`（input0 dtype
"float16,float16,float16,float32,bfloat16,bfloat16" :2987-2989、`attr_enable_hf32` :2960）；
950 `ascend950/aic-ascend950-ops-info-nn.json:4610`；MatMulV3 同构
（910b `aic-ascend910b-ops-info-nn.json:13159`）。

**这一条是 note-16 §7 #2（910B MUL_MAT F32 真 FP32 机制）的关键证据**：310P 的 TBE
BatchMatMulV3 不注册 F32，F32 图只能经精度回退降到 FP16 计算（实测 rel ~1.5e-3 吻合，
note-9 §3）；910B 注册了 F32，F32 直算（实测 rel 3.7e-7 吻合）。机制链完整论证见 §7 取证。

### 4.2 ReduceMean（RMS_NORM 元语，legacy/math 文件）

- 310p `aic-ascend310p-ops-info-legacy.json:32682`：x dtype "float16,float"（f16/f32）；
  axes 输入 dtype "int32,int32,int64,int64"；attr `keep_dims` 默认 false、
  `noop_with_empty_axes` 默认 true（CANNGE 显式传 keep_dims=true，note-14 §4）。
- 950 `aic-ascend950-ops-info-math.json`：x 增加 bfloat16；axes 同 int32/int64；两个 attr
  默认值与 310p 一致。
- 910b：**无条目**（910B 的 RMS_NORM 元语实现走的是非 TBE-python 通道，§1）。

### 4.3 SoftmaxV2 / Swish / GatherV2

- SoftmaxV2：310p `aic-ascend310p-ops-info-legacy.json:38270`——**dynamicFormat=true +
  needCheckSupport=true，无静态 dtype 清单**（dtype 由编译期 .so 回调决定）；attr
  `axes` 默认 [-1]、`half_to_float` 默认 false。950（nn）给出静态清单
  f32/f16/bf16 + precision_reduce=true，attr 默认相同。
- Swish：310p `aic-ascend310p-ops-info-legacy.json:41733`——f16/f32，attr `scale` 默认
  1.0（CANNGE 传 1.0 等价 SILU，note-14 §4）；950 增加 bf16。
- GatherV2：310p `aic-ascend310p-ops-info-legacy.json:19592`——x 支持 f16/f32/整型/bool
  等 11 种，indices int32/int64，axis 为**输入**（int32/int64）而非 attr；attr
  `batch_dims` 默认 0、`negative_index_support` 默认 false。950（nn）dtype 更宽
  （含 complex/double/fp8），attr 默认一致。

### 4.4 elementwise 与数组类（Add/Sub/Mul/Div/Square/Rsqrt/Cast/Transpose/Slice/Concat/Reshape）

- 310p（legacy）：Add/Mul/Slice **dynamicFormat=true、无静态 dtype**；Sub/Div/Square/
  Rsqrt/Cast/Transpose/Reshape 有静态清单（Sub 含 f16/f32/int32/int64/u8/int8/bool；
  Div f16/f32/int32/int8/u8；Square 含 int32；Rsqrt f16/f32）；Add 标
  `slicePattern=elemwiseBroadcast`（与 note-10 F3 的广播结论互证）。Concat 特殊：
  input0 是 `concat_dim`（int32）、input1 动态 `x` 仅注册 float——**清单显著窄于实测**
  （910B 无条目也能跑 CONCAT，§1），说明此类清单不是支持边界。
- 950（math/nn）：静态清单宽得多（Add/Sub/Mul/Div/Square/Rsqrt 含 bf16，部分含
  complex32/64、fp8；Cast/Transpose/Slice 含 hifloat8/fp8/fp4），且 Add/Sub/Mul/Div/
  Square/Rsqrt 标 precision_reduce=true（混合精度白名单语义，见 §7 取证引用的
  ge.exec.modify_mixlist 文档）。
- dtype 命名差异：310p legacy 用 "float" 表示 FP32，950 用 "float32" 明示；同义。

## 5. MatMul 对齐专项（note-10 F3 遗留核心问题）

**问题**：MUL_MAT 的 M/N/K 32 字节对齐约束是否能在 ops-info 中逐产品看到？各 SoC 间
是否有差异？

**结论：ops-info 无此约束记录，16 个 SoC 一致地无记录。静态表只能维持产品无关保守拒绝。**
（ggml-cannge.cpp:658-668 现状不动，注释"未文档化，依据见 note-10 F3"继续有效。）

取证链（全部可在容器内复现）：

1. **ops-info 层（T4）**：MatMul 系（310p legacy MatMul；6/7 SoC 的 MatMulV3；7 SoC 的
   BatchMatMulV3）在**所有有条目的 SoC 里输入 shape 全部 "all"，无任何对齐字段**；对全部
   config 树 grep
   `align|aligned|32-byte|CheckDims` 零命中（唯一命中是 CV 类算子的 `align_corners`/
   `aligned` 属性，如 GridSampler/RoiExtractor，与矩阵乘无关）。
2. **IR 头文件层（T3'）**：`opp/built-in/op_graph/inc/matrix_calculation_ops.h` 中
   GEMM/MatMul 只有**性能建议** "For better performance, The k-axis must be aligned to
   16 (input type is float16) or 32 (input type is int8)"（:343、:387）——是性能建议而非
   正确性约束，且只涉 K 轴、未提 32 字节。同文件 :1412 的 "Output param var last dim
   should be 32B-aligned" 属于 **QuantUpdateScatter**（量化稀疏更新，:1427 注册），与
   MatMul 无关，但说明 IR 规格**有能力**写明 32B 对齐——MatMul 系没写就是真没有。
3. **实现层（T6，note-10 F3 既有结论）**："shape of m/n should be 32-byte aligned"
   只出现在闭源 op_tiling 的 .so 字符串（matmul_v3_to_multi_mul_tiling.cpp），
   `CheckDimsAligned310P` 符号同时存在于 310P 与 A2 路径——ge 开源仓查不到
   （opp 闭源），无法升级出处分级。
4. **实测层**：310P 与 910B 都对非对齐 shape 硬失败（EZ9999，note-9 §3）——两 SoC
   行为一致，**无分产品差异可档**。

## 6. 对 supports_op 静态表的落地建议

可以产品化分档（有 T1/T3'/T4 依据链，且实测吻合）：

1. **MUL_MAT F32 精度分档**：310P = "支持但内部降 FP16 计算"（T4：310p 仅注册 fp16；
   T1：precision_mode=allow_fp32_to_fp16 语义；实测 rel~1.5e-3）；910B = "支持，真 FP32"
   （T4：910b 注册 f16/f32/bf16；T1：ge.exec.allow_hf32 对 Matmul 类默认不使能 HF32；
   T3'：enable_hf32 默认 false；实测 rel 3.7e-7）。note-16 §6 精度特性表可据此从
   "open item"升级为"已解释"。
2. **BF16 分档**：910B/950 的 BatchMatMulV3 及 950 的 elementwise 均有 bf16 TBE 注册
   （T4），但 CANNGE 的 `ggml_cannge_ge_dtype` 只支持 F16/F32/I32（note-14 §2.3），
   BF16 接入需另行验证，本轮不动。
3. **HF32 性能档预留**：ES 封装已暴露 `enable_hf32`（容器
   `x86_64-linux/include/es/es_nn/es_BatchMatMulV3.h:33`，默认 false）。未来若要在 910B
   上开 HF32 性能档，接口现成的；但 310P 无 HF32 能力（aclnn spec：USE_HF32 在推理/
   训练系列不支持），分档必须带 SoC 条件。

维持产品无关保守拒绝（依据不足或两 SoC 实测一致失败）：

1. **MUL_MAT 32B 对齐**：ops-info 无记录（§5），无法产品化分档；维持
   align_elems=32/dtype_size 拒绝。
2. 非连续输入 / 非 dense view / in-place 变体 / 参数化 SOFT_MAX：维持 note-14 §2.3/§6
   现状（这些与 ops-info 无关，属 GE 图构建契约）。
3. note-14 §2.1 的预期修正（建议下轮修订时落笔）：ops-info 是"TBE 算子实现注册表"，
   只在有条目时提供实现层细化；**不得**作为"某 SoC 是否支持某算子"的判定依据
   （§1 三条硬证据）。

## 7. 取证（note-16 §7 #2/#3，aclnnMatmul spec 与 cubeMathType 默认值）

抓取（本机代理 http://127.0.0.1:63145）：T2 gitee 镜像
`gitee.com/ascend/cann-ops/raw/master/src/matmul/mat_mul_v3/doc/aclnnMatmul.md`（86 行
全文）与 `.../batch_mat_mul_v3/doc/aclnnBatchMatmul.md`；T1 官网
`https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnMatmul.md`
（JS/cookie 墙，但正文以 JSON 内嵌形式完整可取，已解码核对，页面版本 9.2.0-beta.2）；
T1 options/试验参数页与容器内 `tbe/impl_mode/*.ini`。

### 7.1 aclnnMatmul 规格要点（T1/T2 同源互证）

- 产品支持（9.2.0-beta.2 页面"产品支持情况"）：950PR/950DT、A3、A2、310p、910 = 支持；
  **310b（Atlas 200I/500 A2 推理产品）= 不支持**。
- dtype：训练/推理系列 F16+F32（无 BF16）；A2 加 BF16（两版本同文）。
- **无 32B 对齐约束**，且明确"支持非连续的Tensor"；mat2 的 Reduce 维须与 self 相等
  （contraction 校验官方化，note-10 F3 补充已录）。
- cubeMathType 枚举（9.2.0-beta.2 比 9.1 时代镜像多第 5 项）：
  0 KEEP_DTYPE（训练/推理系列 F32 不支持）；1 ALLOW_FP32_DOWN_PRECISION（训练/推理
  F32→FP16；A2 F32→HF32）；2 USE_FP16（A2 的 BF16 不支持）；3 USE_HF32（训练/推理
  不支持；A2 F32→HF32）；**4 USE_FP32_ADD（支持使用高精度方式进行计算，新增）**。
- **规格未文档化 cubeMathType 的默认值**（参数表默认值列为 "-"）。结论：aclnn eager
  层的"默认值"问题官方未答。

### 7.2 GE 图路径 matmul 的 HF32 默认（note-16 #3 的落地答案）

GE 图路径没有 per-op 的 cubeMathType 选项；对应精度控制有三层，**三层一致表明
matmul 的 HF32 默认关闭**：

1. **图级 ge.exec.allow_hf32**（T1 原文，商用 7.0.0/8.1.RC1 同文，
   <https://www.hiascend.com/doc_center/source/zh/canncommercial/700/inferapplicationdev/atctool/atlasatc_16_0100.html>）：
   "**参数默认值：针对Conv类算子，使能FP32转换为HF32；针对Matmul类算子，不使能
   FP32转换为HF32。**"支持的型号：Atlas A2 训练系列（HF32 为 A2 系能力）。文档并
   指向随包 `impl_mode/allow_hf32_matmul_t_conv_t.ini`（使能清单）/
   `allow_hf32_matmul_f_conv_f.ini`（未使能清单）——容器内已核：后者 MatMul/MatMulV2/
   BatchMatMul/BatchMatMulV2/GEMM = `enable_float_32_execution`。
2. **算子级 attr enable_hf32**：IR 头 `opp/built-in/op_graph/inc/ops_proto_nn.h:42`
   `.ATTR(enable_hf32, Bool, false)`（BatchMatMulV3，:37 注册）；ops-info 各 SoC 同
   默认 false（§4.1）。ES 封装同名参数默认 false（es_BatchMatMulV3.h:33）。
3. **precision_mode 互斥语义**：allow_hf32 文档明确"若 --precision_mode_v2 默认 fp16
   把 F32 强制转 F16，allow_hf32 不生效"。

结论：**"GE 图路径 matmul HF32 默认关闭"是文档化（T1）+ 配置化（T4）+ IR（T3'）三重
确认的**。910B 实测 MUL_MAT F32 rel 3.7e-7（优于 HF32 典型值）与此完全吻合——默认
根本没走 HF32，是 F32 直算。note-16 §7 #2/#3 关闭。

### 7.3 残余矛盾（如实记录）

- precision_mode=allow_fp32_to_fp16 的官方句"**对于矩阵类算子，使用float16**"
  （T1 options 页，<https://www.hiascend.com/doc_center/source/zh/canncommercial/800/apiref/ascendgraphapi/atlasgeapi_07_0150.html>）
  与 910B 实测（F32 保留）字面冲突。实测行为与"按算子实现注册决定"
  （§4.1：910b 注册 F32 则保留）一致，推测该句是对模型编译路径的简化表述。
  GE eager 图路径的精确适用规则**待核实**（或按"文档简化表述"归档）。
- 310P 的 F32→F16 机制定性为"精度回退"：310p TBE 未注册 F32（T4），与
  allow_fp32_to_fp16 的回退链（AI Core 不支持 F32 则降/落 AI CPU，note-7 §3）相容；
  实测 rel~1.5e-3 只能是 FP16 计算（FP16 输入舍入下限 ~4.9e-4，实测与之同量级）。

## 8. 待核实清单

| # | 条目 | 现状 |
|---|---|---|
| 1 | precision_mode=allow_fp32_to_fp16"矩阵类算子使用float16"与 910B 实测 F32 保留的字面矛盾 | 待核实（§7.3）；以实测 + §7.2 默认值为准 |
| 2 | note-10 Q2 引用的 `aic-ascend910b-ops-info-legacy.json` 在 CANN 9.1.0 容器**不存在**（legacy 仅 310p 随包）；其结论（axes int32/int64、keep_dims=false、noop_with_empty_axes=true）已在 310p legacy 条目复核成立，910B 侧文件名待回溯（疑为 9.1.0-beta.3 环境差异） | 待核实 |
| 3 | 官方 9.1 社区版 options 页 page id 未定位（atlasgeapi_07_0133 在 9.1 版无 precision_mode 正文）；§7.2 引 7.0.0/8.1.RC1 商用版 + latest 试验参数页（2026-08-31 快照）互证 | 9.1 逐字文本待补 |
| 4 | kirin9030/kirinx90/350 的 BatchMatMulV3 标 dynamicFormat=true + 仅 fp16 静态清单，F32 是否经 cast 通道可用 | 无硬件，静态存档；这些 SoC 不在 CANNGE 目标范围（note-17） |
| 5 | 950 的 elementwise 静态清单（bf16/complex/fp8）仅作存档；950 已确认不做（note-17，2026-09-11 用户确认） | 存档 |
| 6 | ES BatchMatMulV3 的 enable_hf32 传入路径若启用，与 ge.exec.allow_hf32/precision_mode 的优先级 | 需真机验证，列入 910B 实测清单 |
