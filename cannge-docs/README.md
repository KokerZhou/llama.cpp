# CANNGE 工作笔记索引

Ascend GE 整图编译后端（ggml-cannge）的研究笔记目录。代码仓：`../llama.cpp`（分支
cannge）；开发环境运维见 note-12。

## 研读笔记（note-1 ~ note-11，建立于 2026-09-09 ~ 09-11，权威附件，勿改动）

| 文件 | 内容 |
|---|---|
| note-1-es-graph-pipeline.md | JittorInfer PR#11 的 ES/GE 构图与执行链路精读 + 移植禁止照搬清单 |
| note-2-inference-mechanisms.md | JittorInfer 可借鉴推理机制（persistent graph/warm-up/plan cache/KV scatter 等）与分工建议 |
| note-3-design-decisions.md | 设计决策记录：D1 回退阶梯、D2 自定义算子 hold、D3 两层判定、D4 ES/GE 关系、D5 310P 备查、D6 图资源模型 |
| note-4-version-comparison.md | JittorInfer 版本谱系（pre#11/post#11/master）与四方执行模型对照 |
| note-5-official-ge-api.md | CANN 9.1 GE C++ API 官方参考（ge_api.h v1/v2、Session 全表面、ge.exec.* 配置） |
| note-6-official-es-ops.md | CANN 9.1 ES 构图接口官方参考（容器头文件精读、阶段 3 窄 op 集清单） |
| note-7-official-limits-and-op-support.md | 官网调研：支持矩阵宪法依据、图资源约束、精度策略、广播规则、静态表初稿 |
| note-8-remote-test-server.md | 310P3 远程真机使用手册（长连接纪律、环境初始化、标准工作流、已知坑）——凭证纪律唯一权威来源 |
| note-9-multi-soc-cann-matrix.md | 多 SoC x 多 CANN 版本验证矩阵（310P3/910B3 × 9.0/9.1，三轮实测） |
| note-10-findings.md | 真机 findings 台账（F1-F6）+ 源码级复核结论，含权威来源指令 |
| note-11-doc-sources.md | 文档来源注册表（T1-T6 分级、按问题选源、注释引用规范）——所有出处的分级权威 |

## 规范文档（note-12 ~ note-18，建立于 2026-09-10 ~ 09-11）

| 文件 | 内容 |
|---|---|
| note-12-dev-environment.md | 开发环境运维手册：310 长连接获取与纪律、910 npussh/npursync 快通道、本地容器与双构建纪律、凭证总则 |
| note-13-code-style.md | 代码风格与工程纪律规范：AGENTS.md 硬规则、注释出处规范、命名、禁止 abort、提交规范 |
| note-14-op-integration-spec.md | 算子接入规范：两层支持判定、supports_op 保守校验契约、CPY/VIEW 的 produced 注册规则、当前 op 清单及约束与出处 |
| note-15-fallback-ladder-spec.md | 回退层次规范：L0-L6 层次职责/触发/状态、层间转移记录要求、aclnn 等价组合（L4b）与手写算子（L5）hold 条款 |
| note-16-validation-spec.md | 验证规范：双构建流程、11-op 探针、test-backend-ops 基线口径（OK=24/FAIL=83/NS=2304）、双 SoC 流程、归档格式、已知精度特性 |
| note-17-compatibility-versioning.md | 兼容性与版本策略：CANN 9.1 基线（9.0 hold）、16 SoC ops-info 矩阵引用、310 先行、JittorInfer 引用边界 |
| note-18-architecture-overview.md | 架构总览：ES 构图 → GE 编译 → session 执行 → plan cache 四层、关键抽象、与 GGML_CANN 切分、GE 生命周期与异步安全、env 门 |
| note-19-static-op-table.md | 逐 SoC 算子支持精细静态表：16 SoC ops-info 全量取证（ops-info 是 TBE 实现注册表而非支持矩阵）、17 GE 算子存在性/dtype/attr 差异矩阵、MatMul 32B 对齐专项（ops-info 无记录，维持保守拒绝）、note-16 两问取证（cubeMathType 默认值/910B F32 机制） |
| note-20-matmul-constraint-docs.md | 310P GE 图路径 matmul 形状对齐约束的文档出处专项：graphdevg 89 页/分产品 Ascend IR 规格 zip×2/cann-ops/ge 仓/容器 .so 反汇编/网络全搜面无契约文档；闭源判据 (dim×dtype)%32 还原；最接近官方文（Ascend C ND_ALIGN N 向 32B）；文档层无解，给出最小硬件实验设计（f32 {8,24,40} 判别形状族） |

编号说明：note-13 已被代码风格篇占用，架构篇编号为 note-18（内容相互独立，按主题查阅即可）。

## 使用约定

- note-1~11 是决策与调研的权威台账，规范文档（note-12~18）引用而非复制其内容；
  冲突时以 note-3/note-11 的决策与出处分级为准。
- 每篇 note-12~18 文末自带"待核实清单"，待用户逐项确认。
- 凭证一律不入库；流程与路径可以写。
