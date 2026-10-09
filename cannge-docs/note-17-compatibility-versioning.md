# note-17: CANNGE 兼容性与版本策略

建立：2026-09-11。规范对象：CANN 版本基线、SoC 支持范围、外部代码来源的兼容性边界。

## 1. CANN 版本基线：9.1

- **最低版本要求 = CANN 9.1.0**，依据（note-9、note-10 F5）：同一份 d85b6bd08 代码在
  9.1.0 上 310P3 与 910B3 均**零改动构建通过**；在 9.0.0 上缺 4 个 ES 封装
  （`Reshape` / `Swish` / `Permute` / `EsCreateConstV2`），需临时 shim 才能编译。
- 9.0 的 shim（CMake try_compile 特性探测 + graph-build 条件回退 + supports_op 拒
  RESHAPE）曾用于 910B3 x 9.0 轮验证，**未落库**（note-9 §5：远程副本在
  `/home/developer/cannge-ggml`，恢复时先 diff 落回再 amend）。
- **向前兼容（9.0）决策：hold**（note-10 F5）：用户指示"先 9.1，再考虑向前兼容"；
  等 F1-F3 收敛后再决定是否把 try_compile 探测做成正式兼容层。在此之前，
  代码可以继续依赖 9.1 的 ES 封装，**禁止**为兼容 9.0 引入未决的条件编译。
- 构建系统行为（CMakeLists.txt）：ES 头检测（`es_graph_builder.h` + `es_math_ops.h`）
  失败即 FATAL_ERROR——ES 是本后端的硬依赖；GE 头采用新旧布局双探测
  （`<ge/ge_api.h>` 优先、平铺布局回退，ge.cpp:34-48）；SOC_TYPE 优先 `-DSOC_TYPE=`，
  回退 npu-smi 自动探测，均无则告警继续（纯编译环境无 NPU 不判失败）。

## 2. SoC 支持范围

- **Ascend 310（310P3）先行**：第一目标 SoC（note-3 D5、note-4 §2）；GE 整图链路已在
  310P3 x 9.1 全链路验证通过（note-9）。
- **910B3 为第二阶段验证环境**：编译与 API 兼容确认 + 跨 SoC 差异取证；其运行时证据
  不作为 310 行为结论（note-3 D5）。
- **Ascend 950 不做**（2026-09-11 用户确认）：决策范围**仅限 CANNGE 后端**，GGML_CANN
  不在此决策内。
- 多 SoC 能力矩阵（ops-info 配置）：容器内
  `/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/config/` 有 **16 个 SoC
  目录**的逐产品算子配置（note-10 F3、note-11 T4），用于精细静态表取证（如逐产品
  比对 MatMul 约束）。**该矩阵是 note-9（多 SoC x 多 CANN 版本验证矩阵）的附件，
  本文不复制其内容，引用即可。**
- 当前已实测矩阵（310P3 / 910B3 × CANN 9.0 / 9.1）与其余空白格的填充路径，见 note-9。
- 已知跨 SoC 差异清单（310 vs 910B）：引用 note-9 §3，不在此复述。

## 3. JittorInfer 的引用边界（架构参考，非代码来源）

- JittorInfer 三个版本（pre#11=master GE 版、post#11=PR#11 ES 版、master 演进版）的
  定位是**工程参照**（已验证的 mapping、workaround、机制设计），可信度等级 T5
  （note-11）：其"正确"可能含巧合，**禁止**作为注释出处或语义依据。
- 具体借用边界（note-4 §1 选型结论）：
  - 借构图**模式**：普通 op 用 ES 构图（PR #11 的 24 个 handler 形态）；
  - 借 executor/session/plan 缓存的**机制设计思想**（note-2 §9 的落地分工表），
    实现全部自研；
  - op 语义基准对照 master 的 GE handler（div/sum_rows/allreduce/rope），对照时仍以
    T1-T3 官方来源复核；
  - **禁止照搬**：私有 op 枚举、flags 魔数、时间戳图 id、地址绑死+空转复用、构建期
    强制 OPP 依赖、不支持即 abort（逐条依据见 note-1 §7）。
- 源码引用纪律：JittorInfer 的路径/行号只允许出现在笔记（note-1/note-2/note-4）中，
  不允许写进 ggml-cannge 代码注释。

## 4. 其他兼容性契约

- **与 GGML_CANN 共存**：两后端独立（设备发现、buffer、stream、GE session 各自一套），
  可同时注册；CANNGE 默认 env 门 off、零存在感（note-18 §1、§4）。
- **平台**：仅 Linux，x86-64 与 arm64（CMakeLists.txt FATAL_ERROR 口径）；x86 host 运行
  无需求（CANNGE 只跑 device 侧，note-9 §4）。
- **多 device**：受 `ge.exec.deviceId` 全局量限制，当前实质仅支持单 device 构图
  （TODO M7，note-18 §2.2）；设备数上限 GGML_CANNGE_MAX_DEVICES=16（ggml-cannge.h:35）。
- **ABI**：CANN 库按旧 C++ ABI 构建，后端必须 `_GLIBCXX_USE_CXX11_ABI=0` +
  `GE_FUNC_VISIBILITY=`（CMakeLists.txt:102-107，JittorInfer workaround 沿用）。

## 5. 待核实清单

| # | 条目 | 现状 |
|---|---|---|
| 1 | 9.0 向前兼容是否做正式兼容层的重启条件与时间点 | hold 中（note-10 F5），等 F1-F3 收敛 |
| 2 | 16 SoC ops-info 矩阵尚未逐产品核对（精细静态表未建，note-10 F3 取证路径） | 待办 |
