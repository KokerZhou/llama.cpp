# note-20: 310P GE 图路径 MatMul/BatchMatMul 形状对齐约束的文档出处专项调研

建立：2026-09-14。任务性质：纯文档定位（无硬件操作、无 repo 改动）。回答一个问题：
**CANNGE 的 MUL_MAT 走 GE BatchMatMulV3 时，310P 图编译路径的 matmul 形状判据
（"shape of m/n should be 32-byte aligned"，CheckDimsAligned310P）官方文档到底有没有写、
写在哪、原文是什么。** 背景见 note-10 F3/F7、note-19 §5/§7。

## 0. 最终裁决（TL;DR）

**GE 图路径（Ascend IR BatchMatMulV3/MatMulV3 → TBE op_tiling）对 M/N 维的
32 字节对齐形状判据，官方文档未覆盖。** 具体说：

- GE 图开发指南全文（CANN 9.1，89 子页）：**无任何 matmul 形状/对齐约束**。
- 分产品 Ascend IR 算子规格附件（CANN 9.1 operatorlist_00095 的 zip，T1 最高权威级）：
  310P（推理系列）与 A2 两份规格中 MatMul/MatMulV2/MatMulV3/BatchMatMul/BatchMatMulV2/
  BatchMatMulV3 **六个条目全部没有形状对齐约束**；同一规格通道里 GroupedMatmul 写了
  "N axis should align to 16"、GEMM 写了 K 轴性能建议——**该写的约束会写，MatMul 系
  没写就是真没有**。
- aclnn 规格（T1/T2 同源）：无对齐约束、明确支持非连续（note-19 §7.1 已证，本轮复核成立）。
- ops-info 配置（T4）：310p BatchMatMulV3/MatMulV3 条目 shape="all"、needCheckSupport=false、
  无任何对齐字段（完整 dump 见 §3.4）。
- GE 开源仓（T3，commit 00ecb5c）：无 CheckDimsAligned/32-byte matmul 字符串；
  32B 字样全部是**内存地址对齐**（device 指针 32 字节对齐）语义，与形状无关。
- 判据仅存在于闭源 opp：`CheckDimsAligned310P` 符号在 liboptiling.so
  （`optiling::matmul_stress_detect::MatmulV3BaseTiling::`），反汇编判据为
  **(dim × dtype_size) % 32 == 0（M、N 两路）**；"shape of m/n should be 32-byte aligned"
  字符串在 libophost_nn.so 的 `matmul_v3_to_multi_mul_tiling.cpp` 簇（T6 线索级，§3.5）。
- **最接近的官方文档**是 Ascend C 算子开发层的"矩阵乘输出的N方向对齐（ND_ALIGN）"
  （CANN 9.1，T1）：输出 C 矩阵 N 方向 32 字节对齐补齐。它解释了底层 cube 输出的
  N 向 32B 硬件要求，但：① 属 Ascend C 设备 API 层，不涉 GE 图路径；② 只有 N 向，
  无 M 向；③ 讲自动补齐（正常行为），不讲 GE 图路径的分解/回退/病态编译。
  **不能作为 GE 图路径判据的出处。**

结论给 supports_op：note-10 F7 的处置（C：实验后再定）是唯一正确路线；文档层无解，
L0 门判据只能靠 §5 的最小硬件实验收敛。注释规范（note-11 §3）不变：该约束在代码中
保持"未文档化，依据见 note-10 F7/note-20"的标注方式，不引 T6。

## 1. 搜索面总览

| # | 来源（分级） | 搜索对象 | 结果 |
|---|---|---|---|
| 1 | T1 GE 图开发指南 graphdevg 89 子页 | 对齐/matmul/约束关键词全文 | 无（§2.1） |
| 2 | T1 operatorlist_00095 分产品规格 zip ×2（推理系列/A2） | MatMul 系 6+2 条目 + GEMM/GroupedMatmul | MatMul 系无；GEMM/GroupedMatmul 有（§2.2） |
| 3 | T2 gitee cann-ops 镜像 | matmul 域全部 doc md + 仓内 docs/ | 无（§2.3） |
| 4 | T3 ge 仓 @00ecb5c（/tmp/ge-src，现存复核） | 全仓 CheckDimsAligned/32-byte/EZ9999/multi_mul | 无（§2.4） |
| 5 | T4 容器 9.1.0-310p | .so 字符串+反汇编、随包源码/文档、310p ops-info | 判据在闭源实现层（§3） |
| 6 | T1 Ascend C opdevg（补充路径） | Matmul 高阶 API 特性/数据排布格式 | N 向 32B 补齐（唯一近似官方文，§4） |
| 7 | T6 网络检索 | EZ9999/CheckDimsAligned/社区工单 | 无形状判据文档（§2.5） |

取证物存档：/tmp/doc-hunt/（graphdevg 89 页 html/txt、两份规格 zip 解压、
asc910_*.md Ascend C 页、llms.txt 全站索引 12MB）。

## 2. 逐来源记录（搜了什么、找到什么）

### 2.1 T1：GE 图开发指南全文（CANN 9.1）——无

- 抓取方法（note-19 §7 技巧的推广）：官网 `/llms.txt`（12MB 全站索引，直连可抓）+
  正文走 `https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/...`
  （免 JS 墙出正文；`/document/detail/...` 大页 NUXT payload 为空壳）。
- 枚举 `atlasag_25_0001~0130` 共 **89 个有效子页**（910 版），全部转文本后 grep：
  `MatMul|matmul|对齐|aligned|字节` 仅命中示例代码/融合 Pass 名单/量化层列表/
  Triton 结构体 `aligned(8)`；`EZ9999|CheckDims|32-byte|32 字节` **零命中**。
- 89 页标题清单核对过（构建Graph/编译运行/量化/AIPP/动态shape/Profiling/图编译缓存/
  附录/ES构图/AutoFuse/SuperKernel/自定义算子入图/自定义Pass/最佳实践），
  无"算子约束"类页面。graphdevg 是 GE 用法指南，不含算子形状契约。

### 2.2 T1：分产品 Ascend IR 算子规格（operatorlist 附件 zip）——MatMul 系无约束

- 入口：`.../910/API/aolapi/operatorlist_00095.html`（规格清单，note-7 §1 的"宪法"页）。
  该页 6 个分产品规格是 zip 附件。下载：
  - 推理系列产品（含 310P）：
    `.../aolapi/resource/算子规格说明(Altas 推理系列产品)_0000002694931955.zip`（1.1MB，
    1174 个算子 md）
  - A2 系列产品：`.../aolapi/resource/算子规格说明(Atlas A2系列产品).zip`（1.3MB）
- **BatchMatMulV3.md（推理系列）全文要点**：原型 F16/BF16/F32；Inputs "2D-6D.
  Has format [ND, NHWC, NCHW]"；分产品 Data Types 段："AI Core: input0 x1: float16 ...
  output0 y: float16"（**310P 只有 F16 上 AI Core**，与 note-19 §4.1/§7.3 一致）；
  **全文无 Attention Constraints 段、无任何对齐字样**。A2 版同文（AI Core 含 F32/BF16），
  同样无对齐约束。
- MatMul.md（推理系列）：Data Types 段是 "AI CPU: float16,float32,int32"——legacy
  MatMul 在 310P 落 AI CPU；无对齐约束。MatMulV2.md / BatchMatMulV2.md：Attention
  Constraints 仅 "if performances better in format NZ, please close
  \"MatmulTransdataFusionPass\""（融合开关提示，非形状约束）。MatMulV3.md：无约束段。
- **同规格通道内对照（证明"会写"）**：
  - GEMM.md Attention Constraints 原文："**For better performance, The k-axis must be
    aligned to 16 (input type is float16) or 32 (input type is int8).**"（K 轴、性能建议口径）
  - GroupedMatmul.md Attention Constraints 原文（含推理系列专条）：
    "Each axis for tensors in x and weight for each group of matmul should be less or equal
    to 2147483647 ... **after aligning to 32 byte**."；"When weight has format NZ,
    **N axis should align to 32 bytes**, i.e. if weight has data type int8, N axis align to 32;
    if weight has data type float16, N axis align to 16."；"**Atlas Inference Series Product
    only supports x, weight and y have data type float16, and N axis should align to 16.**"
- 结论：MatMul/BatchMatMulV3 的 32B M/N 约束在**契约层最高权威文档**中不存在；
  Ascend IR 规格有能力表达形状对齐（GroupedMatmul 即为例证），缺失是真实的。

### 2.3 T2：cann-ops 镜像（gitee）——无

- `src/matmul/` 下 7 个算子目录逐个查 doc/：
  - mat_mul_v3/doc/：aclnnMatmul.md（note-19 §7.1 已证无约束）+ 互推导关系.md
    （dtype type-promotion 表，与形状无关）。
  - batch_mat_mul_v3/doc/：aclnnBatchMatmul.md，无对齐（复核成立）。
  - quant_batch_matmul_v3、gemm_v2：无对齐字样（gemm_v2 命中的是 HTML 表格 align 属性）。
  - weight_quant_batch_matmul_v2/doc/aclnnWeightQuantBatchMatmulV2.md：**有**对齐约束
    （"k为antiquantGroupSize对齐，n为64对齐"；910B UINT64/INT64 模式"k和n要求64对齐"）——
    又一次证明"有约束会写进 spec"，但属量化算子特定模式，与 BatchMatMulV3 无关。
- 仓根 docs/ 只有 contributors/images/Contributing.md，无约束类总文档。

### 2.4 T3：GE 开源仓（/tmp/ge-src @00ecb5c 现存复核；未重克隆）——无

- `CheckDimsAligned|32-byte|32字节|shape of m|multi_mul` 全仓 grep：**零命中**
  （CheckDimsAligned 确属闭源 opp，note-10 Q3 维持原判）。
- `EZ9999` 命中多处，但都是通用内部错误上报（REPORT_INNER_ERR_MSG），无 matmul 形状语义；
  `te_fusion_error_code.h:75` 定义 EM_UNKNOWN_PROCESS_DIED_ERROR="EZ9999"——与网络检索
  一致：EZ9999 是 Inner Error 兜底码，不含形状判据。
- 全部 `32字节/32-byte` 命中均为**内存地址对齐**：acl_mdl.h:1679（devPtr need 32-byte
  alignment）、GeSession/RunGraph*.md（"Tensor在Device侧的存储地址，必须32字节对齐"）、
  model_args_layout_planner.cc（GM→UB 最低拷贝 32 字节）等。**地址对齐 ≠ 形状对齐**，
  这两类约束在文档/源码中从不混用。
- `BatchMatMulV3` 在 ge 仓的实质引用只有 base/common/op_tiling/op_tiling_rt2.cc:78,915
  （RT2 动态 tiling 把 AutoFuse 子图里的 MatMulV3/BatchMatMulV3 委托给二次 tiling），
  无形状检查逻辑。docs/zh 下无 matmul 约束文档。

### 2.5 T6：网络检索——无

- `"shape of m should be 32-byte aligned" Ascend matmul`：零命中（该字符串只存在于闭源 .so）。
- `CheckDimsAligned310P` / `matmul_stress_detect`：零命中。
- `EZ9999`：社区/官方故障处理文档一致定性为 Inner Error 兜底码
  （hiascend 故障处理 PDF、MindSpore 教程、掘金/51CTO 错误码解读均同），
  无 matmul 形状语义。昇腾FAQ A24 提到 Atlas 300I Pro "内存地址未32字节对齐或越界访问"
  导致 107002——又是地址对齐口径。
- 无任何社区工单/官方答复引用过"matmul 形状 32 字节对齐"文档条文。

## 3. 实现层证据（T4 容器深挖；均为"官方发布物但非文档"，线索级）

### 3.1 符号定位

含 `CheckDimsAligned310P` 的 .so（容器 9.1.0-310p，x86_64）：
`opp/built-in/op_impl/ai_core/tbe/op_tiling/.../liboptiling.so`、
`op_host/.../libophost_nn.so`、`libophost_transformer.so`、`libophost_oam.so`。
核心符号（liboptiling.so，nm 可见）：
`optiling::matmul_stress_detect::MatmulV3BaseTiling::CheckDimsAligned310P()` @0x128a920，
同族还有 IsCapable/UpdateNd2nzFlag/GetTilingKey/GetWorkspaceSize/PostTiling——
**"matmul_stress_detect"（应力探测）是 MatMulV3 的一条特殊 tiling 分支**。

### 3.2 反汇编判据（objdump，385 字节函数）

逻辑还原：
```
if (flag@0xd78 != 0 || flag@0xd79 != 0) return 0;      // 两类带 flag 场景跳过检查
if (dword@0xda8 == 2) {                                 // M 路
    if ((qword@0xdc0 * qword@0x1790) & 0x1f) {
        打印 "shape of m should be 32-byte aligned"(行460); return -1; }
}
if (dword@0xda0 == 2) {                                 // N 路
    if ((qword@0xdb8 * qword@0x1788) & 0x1f) {
        打印 "shape of n should be 32-byte aligned"(行466); return -1; }
}
return 0;
```
即 **(dim × dtype_size) % 32 == 0，覆盖 M、N 两路**。f32 m=16→64B 过、n=8→32B 过——
与 F7"过了 32B 门仍病态"自洽（门内还有 GE 分解路径的其它判据，文档全无）。

### 3.3 multi-mul 分解路径字符串簇（libophost_nn.so，偏移 0x1405548 邻域 ±3KB）

`matmul_v3_to_multi_mul_tiling.cpp` 文件名串 + 同簇关键串：
"shape of n/m should be 32-byte aligned"（与 liboptiling 同文）、"illegal value:
m[%ld], k[%ld], n[%ld]"、"invalid input dim num"、"unequal input kDim values"，
以及路径选择串："meet small block num or meet small shape situation, enable small
shape algorithm"、"Ascend310P currently doesn't support IncreShape."、
"MULTI_CORE_SPLIT_K only support fp322fp32, transA=false and transB=true"、
"N is exceeded to 256 to deal with aligning."。
→ F7 的病态编译/数值错发生在这条**小形状分解 tiling**里，其完整判据表只能在闭源
代码中，文档层不可达。

### 3.4 310p ops-info 完整 dump（隐藏字段核查）

`aic-ascend310p-ops-info-nn.json` 的 BatchMatMulV3/MatMulV3 条目全字段：
attr（adj_x1/adj_x2/offset_x/enable_hf32，默认 false）、coreType=AiCore、
dynamicCompileStatic=true、dynamicFormat=**false**、dynamicRankSupport=true、
dynamicShapeSupport=true、input0~3/output0 的 dtype="float16,float16"、
format="FRACTAL_NZ,ND"（x1）、shape=**"all"**、**needCheckSupport.flag=false**、
opFile/opInterface、prebuildPattern=Opaque、softsync=true。
**无任何对齐/support-check 字段**；MatMulV2/MatMul 不在 nn 文件（legacy 文件，note-19 §5 已证）。

### 3.5 随包物其它核查

- 工具包无随包文档：`find /usr/local/Ascend -iname "*.pdf" -o -iname "*.md"` 仅命中
  工具 README 与 pto README；无 docs/guide 目录。
- **意外收获——随包可读设备源码**：`opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/
  ascendc/detect_mat_mul/`（stress-detect 核的 Ascend C 头文件）。其中
  detect_mat_mul.h:39-41：`DATA_SIZE_FP32 = 4; ALIGN_BYTE = 256;`；
  mat_mul_kernel_stress_detect.h:752/787/1095/1165：`if (N * DATA_SIZE_FP32 %
  ALIGN_BYTE != 0)` → 走 `alignedN = ceil(N,64)*64` 的补齐 workspace 路径。
  即 stress-detect 核自身有"N×4B 不满 256B（N 非 64 倍数）时补齐"的非对齐处理——
  与 host 侧 32B 检查是**两个不同层的对齐口径**（32B host 判据 / 256B 核内补齐），
  进一步说明该实现是一组分层启发式，无单一文档化判据。
- TIK python API（te/tik/api/cube/matmul.py）：只有地址对齐检查（L1 512B、src 32B、
  L0C dst 1024B），无形状维对齐。
- 310p 预编译核二进制元数据（kernel/ascend310p/ops_nn/*/.o.json）：仅 FP16 变体
  （FP16_FP16_FP16_FP16），simplifiedKey 形状位为 -2（动态），无约束声明。

## 4. 最接近的官方文档原文（T1，CANN 9.1 社区版，均为 Ascend C 算子开发层）

> 这些是可引用的"32 字节 + 矩阵乘形状"官方表述，但**层级不对**（设备 API 层，非 GE
> 图路径/Ascend IR 规格层），不能升级为 supports_op 判据的出处，仅供机理理解。

1. **矩阵乘输出的N方向对齐（ND_ALIGN）**
   `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/guide/算子实践参考/SIMD算子实现/矩阵编程（高阶API）/特性场景/矩阵乘输出的N方向对齐.md`
   原文摘录：
   - 功能介绍："ND_ALIGN是矩阵的另一种数据格式，该格式一般用于**N方向非32字节对齐的
     矩阵乘计算**中，配置结果C矩阵为ND_ALIGN格式后，将按照**N方向32字节对齐的补齐规则**
     输出C矩阵"。
   - 使用场景："Matmul计算中N方向非32字节对齐，输出C矩阵的N方向要求32字节对齐的场景。"
   - 约束说明："若配置C矩阵为ND_ALIGN格式输出，则为C矩阵申请的Buffer空间为**N向上32字节
     对齐后的空间大小**。"
   - 示例：M=16, K=16, N=14, half（14×2B=28B 非 32B 对齐 → 自动补两列）。
2. 数据排布格式.md 的 ND_ALIGN 条目（同树 技术附录/概念原理和术语/神经网络和算子/）：
   "输出矩阵乘的结果矩阵C时，用于配置C矩阵按照N方向32字节对齐的规则进行输出。"
3. Matmul特性介绍.md（同目录）：特性表含"矩阵乘输出的N方向对齐……N方向32字节对齐的
   自动补齐及输出"；"多核对齐切分"仅是 M/N/K 被 singleCoreM/N/K 整除的 tile 语义。
   **全篇无 M 方向 32 字节约束。**
4. 分产品 Ascend IR 规格（§2.2 的 GEMM/GroupedMatmul 引文）——契约层里唯一写明了
   matmul 形状对齐的条目，但对象是 GEMM（K 轴性能建议）与 GroupedMatmul（N 轴对齐），
   不是 GE 图路径的 BatchMatMulV3。

## 5. 落地建议

### 5.1 supports_op L0 门（文档层无解 → 维持保守 + 实验收敛）

note-10 F7 的用户处置（C：记录+实验后再定）经本轮调研确认是唯一选项：
文档给不出 GE 图路径判据，静态表无法从 T1-T3 升级。当前 L0 门
（align_elems=32/dtype_size，产品无关拒绝）保持不变，代码注释维持
"未文档化，依据见 note-10 F7/note-20"（note-11 §3 第 3 条口径）。

**由 §3.2/§3.5 引出的待验假设**（实验要一 runs 判定的核心）：
310P 的 BatchMatMulV3 AI Core 只注册 F16（§2.2/§3.4），GE 对 F32 输入做精度回退
（F32→F16，note-19 §7.3 文档化）。若回退后形状判据按 **fp16 口径**（16 元素=32B）
执行，则 F32 的 L0 门也应按 16 元素拒绝（比现在 8 元素更严）——F7 的 f32 n=8
（8%8==0 现门放行；8%16≠0 新门拒绝→交回 CPU 得正确值）即可被解释。

### 5.2 允许的最小硬件确认实验（设计建议，未执行）

目标：一轮跑完即可判定"310P GE 图路径 matmul 有效形状判据"，产出可固化进静态表的边界。

- **形状矩阵**：m,n ∈ {8,16,24,32,40,48,64} × k ∈ {256,512,1024}，f32/f16 各一套
  （沿用 note-10 F7 设计）。关键判别形状：**f32 下 m 或 n ∈ {8,24,40}**（过 f32 8 元素门、
  不过 fp16 16 元素门）——这批形状一次性区分"原 dtype 口径"vs"回退 fp16 口径"。
- **对照**：已知好形状 16×16×16（f16/f32，两 SoC 均 92ms 级）；已知坏形状 F7 原例
  f32 16×8×{256,4096}。910B 同矩阵跑一遍作 SoC 对照（910B 支持 F32 直算，口径应不同）。
- **执行纪律**：每形状独立进程；GE 编译超时 120s 判"病态编译"（防 F7 型 >10min 卡死）；
  三态记录（数值正确/编译失败/病态编译）+ rel err（对 CPU 参考）。
- **日志**：ASCEND_GLOBAL_LOG_LEVEL=0 收集编译期 CheckDimsAligned310P/32-byte 警告，
  与三态结果逐形状关联——验证"警告残留⇒分解路径⇒病态/数值错"链条。
- **产出**：310P matmul 可形状边界表（dtype × m/n/k × 三态）；若 f32 {8,24,40} 形状族
  病态而 f16 同形正常 → 回退 fp16 口径成立 → L0 门 f32 改 align_elems=16（仅 310P；
  910B 以同批实验结果决定是否产品化分档）。
- **前置**：310 长连接恢复后先清理 F7 残留进程（PID 37148，note-10 F7 收尾项）。

## 6. 证据强度声明（按 note-11 分级）

| 结论 | 依据级别 |
|---|---|
| GE 图指南/分产品 Ascend IR 规格/aclnn 规格/ops-info 均无该约束 | T1+T2+T4 多源一致，可进注释作"未文档化"断言 |
| 判据 (dim×dtype_size)%32==0、M/N 两路 | T6（.so 反汇编，容器随包物）——线索级，不进注释 |
| N 向 32B 输出对齐是官方文档化行为（Ascend C 层） | T1（CANN 9.1 原文），但层级不适用 GE 图路径 |
| EZ9999 = 通用 Inner Error，无形状语义 | T1（故障处理 PDF）+T6 社区一致 |

## 7. 待核实清单

| # | 条目 | 现状 |
|---|---|---|
| 1 | f32 {8,24,40} 判别形状族的 310P 实测（§5.2 实验） | 等 310 会话；产出后回填 note-10 F7 |
| 2 | A2(910B) 侧同批实验，判是否需产品化分档 | 同上 |
| 3 | 910B 容器的 liboptiling 是否有 CheckDimsAlignedA2 等价符号 | 310P 镜像只有 310P 变体；910B 镜像可顺手核（非阻塞） |
| 4 | 若实验证实"回退 fp16 口径"，L0 门改 align_elems=16 的代码落地需用户批准（note-10 F7 处置 C 的流程约束） | 待实验 |
