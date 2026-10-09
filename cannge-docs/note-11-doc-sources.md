# note-11: CANN 文档来源注册表（分层、可信度、注释引用规范）

建立：2026-09-11。用途：写代码/注释时"每个 GE 语义决策注官方出处"的选源依据。
规则：注释只引本表 T1-T3 来源（URL 或 仓内路径+commit）；T4 仅线索，不进注释；T5-T6 禁止引用。

## 1. 分层与可信度

| 级 | 来源 | 覆盖层面 | 可信度 | 备注 |
|---|---|---|---|---|
| T1 | CANN 官网文档树 `hiascend.com/document/detail/zh/CANNCommunityEdition/<ver>/...` | 对外契约：GE API（ascendgraphapi/）、算子规格（aolapi/context/ops-*/）、支持矩阵（operatorlist_00095）、options（atlasgeapi_07_08xx）、开发指南（programug/graphdevg/atlasag_25_*） | ★★★★★ 契约层最高权威 | JS/cookie 墙，部分页抓不到时用 T2 镜像同文 |
| T2 | gitee.com/ascend/cann-ops（raw 抓取） | 与 T1 的 aolapi 算子规格**同源**（src/<域>/<算子>/doc/aclnn*.md）+ 各算子 README 规格 | ★★★★★ 与 T1 同源 | 绕墙首选；URL 规律：`src/{matmul,norm,math,...}/<op>/doc/aclnn<Op>.md` |
| T3 | GE 官方源码仓 `gitcode.com/cann/ge`（浅克隆 /tmp/ge-src，commit 00ecb5c） | 实现层语义终极依据：API 契约（docs/zh/api/）、设计文档（docs/zh/design/features/）、runtime 行为（runtime/）、IR 定义（tests/.../all_ops.cpp） | ★★★★★ 实现即真相 | 阅读成本高；注释引用格式：`ge 仓 <路径>:<行> @<commit>`；ES 无公开文档，ES 语义以 IR 定义（all_ops.cpp）+ 头文件为准 |
| T3' | CANN 容器头文件（本地镜像 quay.io/ascend/cann:9.1.0-310p） | 精确签名、ES 函数原型、gert::Tensor 成员 | ★★★★ 签名级权威，语义级不足 | 生成代码（gen_esb）无深语义；注释引用格式：`include/es/es_Xxx.h` |
| T4 | 随包 ops-info 配置（`opp/.../tbe/config/<soc>/*.json`，16 个 SoC 目录） | 逐 SoC 算子输入 dtype/shape 校验、分产品约束 | ★★★★ 官方发布物，未文档化 | 用于精细静态表取证（如 F3 逐产品对齐）；注释引用格式：文件路径 |
| T5 | JittorInfer（pr11/tip/master） | 工程参照（已验证的 mapping、workaround） | ★★★ 非权威 | 其"正确"可能含巧合；只作对照，不作注释出处 |
| T6 | .so 字符串、错误码、社区 issue | 线索 | ★★ 仅定位用 | 不进注释；结论须回溯到 T1-T4 确认 |

## 2. 按问题类型选源（查什么先查哪）

| 问题类型 | 首选 | 备选 |
|---|---|---|
| GE API 行为/生命周期/配置项 | T1 ascendgraphapi API 页 + options 页 | T3 源码 + docs/zh/api |
| 算子 dtype/shape/对齐/广播约束 | T1 aolapi 算子规格页 | T2 同文镜像；T4 逐产品细化 |
| 某 SoC 是否支持某算子 | T1 operatorlist 分产品规格清单 | T4 ops-info 配置 |
| 输入输出内存契约/零拷贝/复用 | T3 docs/zh/design/features/memory_*.md + runtime/ | T1 options 页 |
| ES 构图函数语义 | T3 IR 定义 all_ops.cpp + T3' 头文件 | —（无公开文档，note-6 结论） |
| 融合行为/图优化副作用 | T1 图融合和UB融合规则参考 | T3 compiler/ |

## 3. 注释引用规范（写代码时执行）

1. 每个 op mapping / 每个非平凡 GE 语义决策，注释带一行出处，格式三选一：
   - 官网：`// spec: <https://...aclnnXxx.md>`（tensor 约束、dtype、对齐）
   - 源码：`// ge 仓 <path>:<line> @00ecb5c`（行为契约类）
   - 头文件：`// include/es/es_Xxx.h`（仅签名事实）
2. 注释保持 1-2 行（llama.cpp AGENTS.md 规范），出处不占额外行，附在语义句后。
3. T4-T6 的结论若进代码，必须先升级到 T1-T3 依据；升级不了的在 note-10 标 open item，
   代码注释写"未文档化，依据见 note-10 F#"。
4. 链接选该决策对应的层：契约类选官网/源码仓，算子约束选 aolapi/cann-ops 规格。
