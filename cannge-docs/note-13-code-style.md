# note-13: CANNGE 代码风格与工程纪律规范

建立：2026-09-10。用途：所有 CANNGE 后端代码改动必须遵守的格式、命名、注释、结构与提交规范。
优先级：本规范与 llama.cpp/AGENTS.md 冲突时，以更严格者为准；两者都管不到的，以本规范补充。

## 1. 基本盘：llama.cpp/AGENTS.md 硬性规则（必须全部遵守）

- **ASCII only**：禁止 em dash `—`、unicode arrow `->` 之外的箭头、`×`、`…` 等任何非 ASCII 字符；用 `-`、`->`、`x`、`...` 代替。字符串、注释、commit message 一视同仁。
- **注释 1-2 行精简**，ASD-STE100 简式英语（怎么短怎么写），不写"做了什么"的复述注释，只写非显而易见的约束/不变量。
- **禁止句中断行折行**、禁止固定列宽硬折注释和代码。
- **先写代码后补注释**：禁止先铺一堆注释再填代码；只在真正需要处补注释。
- **复用现有设施**，不引入新组件/新抽象，不做入侵式大改；改动必须与周边代码风格融合。大改或新模式必须先 pause 向用户确认（见 llama.cpp/AGENTS.md）。
- **不在 tests/ 新增文件**（除非用户批准）；验证复用 test-backend-ops 等现有设施。

## 2. 注释出处规范（用户硬性要求，细节见 note-11）

1. 每个 op mapping、每个非平凡 GE 语义决策，注释必须带一行出处，三选一格式：
   - 官网：`// spec: <https://...aclnnXxx.md>`
   - 源码：`// ge 仓 <path>:<line> @00ecb5c`
   - 头文件：`// include/es/es_Xxx.h`（仅签名事实）
2. 出处只允许引 note-11 的 T1-T3 来源；T4（ops-info 配置）结论进代码前必须升级到 T1-T3 依据，升级不了的标"未文档化，依据见 note-10 F#"；T5-T6 只作线索，禁止进注释。
3. 出处附在语义句后，不额外占行，总注释仍保持 1-2 行。

## 3. 命名

- 公开符号（ggml-cannge.h 导出）：`ggml_backend_cannge_*`。
- 内部符号：`ggml_cannge_*`（函数、结构体如 `ggml_cannge_plan`）。
- ES 构图辅助：namespace `cannge_es`（见 graph-build.cpp）。
- 目录/目标名：`ggml-cannge`；CMake 选项 `GGML_CANNGE`；编译定义 `GGML_USE_CANNGE`；显示名 `CANNGE`。
- 禁止出现 `cannge` 之外的新自创缩写；变量名与 ggml 惯例一致（ne/nb/src/dst）。

## 4. 结构与行为纪律

- **禁止 abort/exit**：任何不支持、校验失败、GE 报错，必须返回 false 或错误码交上层调度（CPU 兜底），绝不终止进程。
- **CANNGE 与 GGML_CANN 完全独立**：禁止链接、包装、复用 ggml-cann 内部符号；两者可分别开启、可同时存在。
- 实验性路径必须 env 门控（先例：`GGML_CANNGE_GRAPH=1`，默认 off）。
- 部分失败必须清理：编译/加载/绑定中途失败的图，禁止泄漏 session、graph、staging buffer、已绑 device 内存。
- 新增 op 必须先按"算子接入规范"（见规范文档集算子篇）过 supports_op 两层判定，再写 mapping；先测试后扩展。

## 5. 提交规范

- **未获用户明示批准，禁止 git commit / push**（用户全局规则）。
- 批准后的提交并入单 commit `d85b6bd08` 工作流（amend），commit message 简洁一行 + 空行 + `Assisted-by: Kimi Code`；不用 `Co-authored-by:`。
- commit message 与代码注释同标准：ASCII、无废话。

## 6. 文档联动

- 行为语义变化（含 GE 行为新发现）必须同步：note-10 findings 台账（新问题）、note-9/算子静态表（支持矩阵变化）、相关规范篇。
- "待核实"类结论不进代码注释正文，进 note-10 open item。
