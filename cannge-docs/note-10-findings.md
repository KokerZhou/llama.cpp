# note-10: CANNGE 真机 findings 台账（逐项解决中）

建立：2026-09-10。工作规则：**每项先查最新 ES/GE 官方文档（容器头文件 doxygen + hiascend 官网算子规格），
明确官方语义后给结论，禁止无脑穷举**。证据来源：三轮真机验证（note-9 矩阵）。
解决一项勾一项，解决方案附官方出处。

## F1 [P0] 910B 静默丢弃"输出-输入别名"写回

- **现象**：910B3 上 PERMUTE/CPY-cast 探针失败，310P3 全过；9.0/9.1 表现一致（SoC 差异，与版本无关）。
- **证据**：金丝雀实验——绑定输出到写入前值为 99 的 buffer，ExecuteGraphWithStreamAsync 返回成功后
  读回仍 99，GE 根本没写。CPY 链中间张量全 0 同因：dst 被 analyze 误注册为图输入，
  输出端口地址与输入缓冲重叠。reuseZeroCopyMemory=0/1 无影响。
- **待查文档**：gert::Tensor 绑定规则、GE 对输入/输出内存重叠的官方约束（是否明文"输入输出不可重叠"）。
- **修复方向（暂定，待文档确认）**：analyze 把 CPY dst 标记为产出（非输入）；输出绑定前做
  地址重叠检测，命中则 GE 写 plan 持有的临时 buffer，execute 后 writebacks 列表 D2D 拷回真实地址。
- **状态**：调研完成（agent-12，2026-09-10）。**官方结论：未找到任何明文**（9.1 头文件、
  9.1.0-beta.3 API 页、6.3-8.x options 文档均无"输入输出地址不得重叠"条款）——重叠是
  **未文档化的未定义行为**：310P"恰好能跑"与 910B"静默不写"都是 UB 的不同表现。GE 默认契约
  （推断）：输入内存只读、输入/输出生命周期独立；改写输入是显式 opt-in 的
  `inputReuseMemIndexes`（方向相反，不能用来修）。
- **修复设计（官方认可方向）**：绑定前做地址区间重叠检查，命中则输出绑后端自持 staging
  buffer，execute 后在同一 stream 上 aclrtMemcpyAsync 拷回——与官方"先分配内存、执行后
  aclrtSynchronizeStream"模型完全兼容。另配合 analyze 修 CPY dst 误注册为输入的 bug。
  长期可用 RegisterExternalAllocator（官方机制，等价）。

## F2 [P0] RMS_NORM 4D 数值错（SoC 无关，占 26 个 test-backend-ops FAIL）

- **现象**：4D shape（如 [64,5,4,3]）数值错（ERR~1.03），3D 及以下稳定 0 误差。
- **证据**：两 SoC（310P3/910B3）x 9.1 同错 → 后端 bug，非核问题。当前实现：元语分解
  `y = x * rsqrt(mean(x^2, axes, keepdims) + eps)`（310 融合核丢行问题的替代）。
- **待查文档**：es_math ReduceMean 的 axes/keep_dims 官方语义（axes 是属性还是张量？负轴支持？
  keep_dims 默认？4D 下是否有 rank 相关限制）；对照我们 ge_shape 反轴序后 axes 的换算是否正确。
- **状态**：调研完成（agent-13，2026-09-10）。**根因定位：graph-build.cpp:221-222**——
  `EsCreateConstV2(builder, axes.data(), axes.data(), size, DT_INT64)` 第 3 参应是 dims 指针，
  却传了 axes.data()（数据缓冲）→ Const 的 shape 变成 [nd-1] 而非 [1]，axes 内容越界读。
  与实测完全吻合：nd=2 碰巧对；nd=3 越界值恰为 0 时多归约尺寸-1 轴成恒等（"3D 全对"是
  假象）；nd=4 多归约外轴 → ERR~1.03。
- **官方语义**（es_ReduceMean.h:31，CANN 9.1.0）：axes 是张量输入；keep_dims 默认 false
  （我们显式传 true 正确）；`CreateVector(std::vector<int64_t>)`（es_graph_builder.h:~399）
  是建 int64 向量常量的正确封装（dims 不会错位）。
- **修复**：axes 常量改用 `ctx.builder.CreateVector(axes)`，或 EsCreateConstV2 第 3 参传
  单独的 dims={1} 数组。修复时一并复核文件内所有 EsCreateConstV2 调用点（SOFT_MAX 等）。

## F3 [P1] supports_op 未镜像构建前置条件 → 该回退的用例变 FAIL

- **现象**：test-backend-ops 中大量用例 FAIL 而非 not-supported，回退阶梯 D1 在此断裂。
- **已知前置条件（代码侧已确认，缺静态表镜像）**：
  - matmul 维度 32B 对齐：310P **和** 910B 都强制（EZ9999，50 条）——是否官方文档化待查
  - 输入连续性/dense 检查（非连续输入在 build 期 "no ES tensor" ~880 条）
  - in-place `op->src[0] == op` 未拒
- **待查文档**：Ascend IR MatMul/BatchMatMul 规格的分产品 shape 约束（32B 对齐是否写在规格里），
  各 elementwise 算子的 broadcast/shape 约束原文。
- **状态**：调研完成（agent-14，2026-09-10）。官方结论：
  1. **elementwise 自动广播是文档化行为**（elewise_calculation_ops.h CANN 9.1.0 原文
     "Support broadcasting operations"），维持 supports_op 现有逐维"相等或为 1"判定即可；
  2. **BatchMatMul 仅 batch 维自动广播**（IR 头原文），矩阵维 M/K、K/N 必须严格相等；
  3. **32B 对齐未在 Ascend IR 规格文档化**——只在 host 实现 .so 字符串（matmul_v3_to_multi_
     mul_tiling.cpp: "shape of m/n should be 32-byte aligned"）与 GEMM IR 头性能建议
     （K 轴 fp16 对齐 16 元素/int8 对齐 32 元素）中出现；`CheckDimsAligned310P` 符号同时存在于
     310P 与 A2 路径。GE 有 unalign fallback 通道但 310P 部分路径无 fallback 直接 FAIL——
     故 supports_op 按**产品无关**保守拒绝：align_elems = 32/dtype_size（F16→16、F32→8），
     M/N/K 任一对齐不过即 false 交回 CPU。
  4. dtype 白名单按 IR 头原文：F16/BF16/F32（窄集现状一致）。

## F4 [P2] 310 格需按新口径重跑

310P3 轮的 test-backend-ops（134/216）是在 supports_op 未镜像前跑的，F3 修完后 FAIL 口径会变，
需要新凭证重建 310 长连接后重跑，与 910 同口径对比。

## F5 [P2] 9.0 向前兼容决策（hold）

后端最低版本 = 9.1（缺 Swish/Permute/Reshape/EsCreateConstV2 封装）。9.0 shim 代码未落库。
等 F1-F3 收敛后再决定是否做 try_compile 特性探测的正式兼容层。用户指示：先 9.1，再考虑向前兼容。

## F6 [P2] 融合 RmsNorm 核在 910B 大 shape 的可用性

310P3 小 shape 间歇丢行（→ 已改元语）；910B 未复现但样本小。若 F2 元语实现稳定，
融合核可长期不用；留一项 310 实测确认。

## F7 [P1] MUL_MAT f32 m=16 n=8 形状族：310 上数值错 + GE 编译病态（2026-09-14 F4 实测）

- 现象：f32 m=16 n=8 k=256/4096 两例，910B（d85b6bd08）OK → 310（28558340a）数值错
  （ERR 96.7/1415.4）；该形状 GE 编译 >10 分钟不完成（对照 16x16x16 仅 92ms）；
  编译期残留 CheckDimsAligned310P（"shape of m"）警告。
- 定性：L0 门判据（32B/dtype_size，f32→8 元素）与 GE BatchMatMulV3 内部映射口径
  不一致——过了门的形状 GE 内部仍判不对并产生病态编译。32B 对齐约束只存在于闭源
  op_tiling .so 字符串（Q3 维持原判），无法从文档升级判据。
- 处置（2026-09-14 用户定：**C，记录+实验后再定**，不改代码）。文档定位已做（note-20，
  2026-09-14）：**官方文档未覆盖该判据**——GE 图指南 89 子页零命中、分产品 Ascend IR
  规格 zip 里 MatMul 系六条目全无形状约束（同规格 GroupedMatmul 写了 N 向对齐，反证
  "没写即真没有"）；判据仅存在于闭源 liboptiling.so（反汇编还原为
  (dim×dtype_size)%32==0、M/N 两路，与我们 L0 门一致）→ F7 的病态更可能在
  **f32→fp16 精度回退层**按 fp16 口径二次判对齐（待验假设，note-20 §5.1）。
- 下轮 310 会话执行 note-20 §5.2 最小判别实验（非 392 格盲扫）：关键判别形状 f32
  m/n ∈ {8,24,40} 一族（过 f32 8 元素门、不过 fp16 16 元素门），一 runs 区分两种口径；
  已知好 16×16×16、已知坏 F7 原例对照；每形状独立进程 + 120s 超时判病态；
  若 {8,24,40} 病态而 f16 同形正常 → f32 L0 门改 16 元素（仅 310P，910B 同批实验定）。
- 收尾时 310 连接中断，远程复现进程（PID 37148）未清理——恢复后先 kill + npu-smi 确认。

## F8 [P1] in-place 拒绝不完整：view_src 型 in-place 绕过 L0 门（2026-09-14 F4 实测）

- 现象：`op->src[i]==op` 检查（28558340a）未覆盖 rms_norm_inplace / scale_inplace：
  ggml 的 *_inplace 形态是 `view_src=src[0]`（非 src[i]==op），仍被 supports_op 放行、
  GE 执行后数值错（rms_norm_inplace ERR=0.685，scale inplace 同）。
- 根因两处：① supports_op 只查 src[i]==op；② plan analyze 的 F1 is_dense 别名捷径把
  "带 view_src 的计算 op"误当纯 view——非变换别名只对 VIEW/PERMUTE/TRANSPOSE 成立，
  计算 op 的 in-place 结果必须由 GE 物化，不能靠别名。
- 修法（已定方向，**2026-09-14 用户决定暂缓**：先 review F4 完整报告
  ~/.kimi-private/remote-results-310-f4/report-f4.md 再批准落地）：supports_op 拒
  "view_src 非空且 op 不在 {VIEW,PERMUTE,TRANSPOSE} 白名单"；analyze 别名捷径按同一
  白名单限制。

## 权威来源指令（2026-09-10 用户指定）

F1-F3 及后续所有 GE 语义问题，以下来源为最高权威，优先级高于头文件 doxygen、.so 字符串和推断：
1. 官网文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/programug/graphdevg/atlasag_25_0081.html（GE 图开发指南，CANN 9.1）
2. GE 官方源码仓：https://gitcode.com/cann/ge（CANN GE 实现源码，语义终极依据）

## 源码级复核结论（agent-15，2026-09-11，依据 gitcode.com/cann/ge 源码仓 00ecb5c + 官网文档）

### Q1/F1 修正：重叠 IO 有官方契约，且找到静默丢弃的代码级机制
- **契约文档化**（推翻"无任何明文"）：docs/zh/api/.../memory_management.md:139-167 ——
  `ge.exec.outputReuseInputMemIndexes` 显式声明"输出与输入共享同一内存地址"（原地语义），
  声明后 GE 按原地编译（tensormove_delete.md:230-232 删除 Data->NetOutput 的 TensorMove），
  且运行时强制校验地址精确相等（base/common/util.cc:560-596 CheckIoReuseAddrPairs，
  不等返回 PARAM_INVALID；davinci_model.cc:7213、model_v2_executor.cc:459 空声明即跳过校验）。
- **未声明的重叠 = 未校验 UB**（确认此前推断）。**静默不写输出的机制**：
  runtime/v2/kernel/common_kernel_impl/memory_copy.cc:574-579 TensorToOut —— src==dst 时
  "Zero copy takes effect" 跳过对输出的拷贝。910B(v2 路径)与 310P 行为差异与此一致。
- **修复定案**：维持 staging buffer + writeback 方案。官方原地语义通道
  (outputReuseInputMemIndexes) 要求地址精确相等，view 带 offset 场景不满足，不采用。
- /tmp/ge-src 留有完整浅克隆可继续查。

### Q2/F2 确认 + 补充
- IR 层确认（tests/.../eager_style_graph_builder/all_ops.cpp:39019-39048）：ReduceMean 的
  x/axes 为 required 输入，keep_dims/noop_with_empty_axes 为 optional attr。
- 910B ops-info 配置（aic-ascend910b-ops-info-legacy.json:38387-38434）：axes dtype
  int32/int64（我们 int64 合法）；keep_dims 默认 false；ReduceMean 空 axes 默认 noop=true
  （ReduceMeanD 变体默认 false）。负轴支持源码无 verify，维持未确认。
- F2 修复（CreateVector(axes)）维持不变。

### Q3/F3 维持原判 + 后续取证路径
- ge 开源仓无 CheckDimsAligned/multi_mul/unalign（属闭源 opp：libopsintf/liboptiling），
  无法从源码升级结论；维持"32B 对齐按产品无关保守拒绝"。
- 后续取证：本地容器 /usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/config/
  有逐 SoC ops-info（ascend310p/ascend910b 等 16 个目录），可逐产品比对 MatMul 约束做精细静态表。

### F3 补充证据（2026-09-11 用户提供权威入口：aolapi 算子规格）
- 入口：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/aolapi/context/ops-nn/aclnnMatmul.md
  （官网页 cookie 墙，内容同 gitee 官方仓镜像
  gitee.com/ascend/cann-ops/raw/master/src/matmul/mat_mul_v3/doc/aclnnMatmul.md）
- 官方确认：分产品 dtype——Atlas 推理/训练系列 F16+F32（无 BF16），A2(910B) 加 BF16；
  窄集 F16/F32 两 SoC 均合法。
- cubeMathType 官方语义（对应我们实测的精度差）：KEEP_DTYPE 在推理/训练系列对 F32
  **不支持**；ALLOW_FP32_DOWN_PRECISION 在推理/训练系列 F32→FP16、A2 F32→HF32。
  → 310P F32 matmul 必须降精度是官方文档化的；910B 实测 rel 3.7e-7 优于 HF32 典型值，
  留作 open item（可能 GE 图路径 cubeMathType 默认不同）。
- **该 spec 无 32B 对齐约束**（且明确"支持非连续的Tensor"）→ 32B 硬 FAIL 是 GE 图编译
  tiling 路径约束，不在 aclnn eager 层，维持"产品无关保守拒绝"。
- 同 spec：mat2 的 Reduce 维须与 self 相等（contraction 维校验官方化）。
- URL 规律：底层算子规格 = aolapi/context/ops-{nn,math,...}/aclnn<Op>.md，
  镜像在 cann-ops 仓 src/<域>/<算子>/doc/ 下，后续按此查其他算子。
