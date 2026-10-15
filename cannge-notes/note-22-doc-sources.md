# note-22: 文档来源整合注册表（本机 310P3 实测可达性，2026-10-08）

note-11 的分层规范继续有效（T1-T6、注释引用规范）。本文件是"本机可用来源全集"，
含 CrispASR 项目沉淀的来源与 URL 转换规则。**注释出处仍只允许引 T1-T3**，
T4 及以下只作取证线索。

## 0. 本机连通性实测（2026-10-08）

| 源 | 状态 | 备注 |
|---|---|---|
| hiascend.com detail 页 | 200 | curl 可达，正文 JS 渲染，用 FetchURL 抓正文 |
| hiascend.com source 直链 | 200 | **可直接 curl 下载**（MD/PDF） |
| gitcode.com/cann/ge | 200 | GE 源码仓可克隆 |
| gitee.com raw | 302 | `curl -L` 跟随后可下 |
| github.com | 超时 | 间歇性全不通 |
| gh-proxy.com | 200 | clone/fetch 前缀 `https://gh-proxy.com/https://github.com/...` |
| huggingface.co | 超时 | 模型走 `https://hf-mirror.com` 全量镜像 |

## 1. T1 CANN 官网（契约层最高权威）

- 文档树：`https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/<ver>/...`
  - 版本段：`910` = CANN 9.1.0 文档集（本机 cann-9.1.0 对应）；`latest` = 滚动版
  - 算子规格：`.../API/aolapi/context/ops-{nn,norm,math,transformer,...}/aclnn<Op>.md`
  - 支持矩阵：`.../API/aolapi/operatorlist_00094.html` / `_00095.html`（分 SoC）
  - GE API：`.../API/ascendgraphapi/...`；options：`atlasgeapi_07_08xx`
  - 图开发/融合规则：`.../programug/graphdevg/atlasag_25_*`
- **URL 转换下载规则（CrispASR 时期验证）**：
  `document/detail/zh/<path>` 换成 `doc_center/source/zh/<path>` 即可直链下载。
  例：
  - 查看页 `.../document/detail/zh/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnFusedInferAttentionScore.md`
  - 下载 `.../doc_center/source/zh/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnFusedInferAttentionScore.md`
  - 算子库全量 PDF：`.../doc_center/source/zh/CANNCommunityEdition/910/API/aolapi/CANN%209.10%20%E7%AE%97%E5%AD%90%E5%BA%93.pdf`（即 "CANN 9.10 算子库.pdf"）

## 2. T2 cann-ops 同源镜像（绕墙首选）

- `https://gitee.com/ascend/cann-ops`（raw：`/raw/<branch>/<path>`，需 `curl -L`）
- 与 T1 aolapi 算子规格**同源**：`src/{matmul,norm,math,...}/<op>/doc/aclnn<Op>.md`

## 3. T3 GE 源码仓（实现层终极依据）

- `https://gitcode.com/cann/ge`（浅克隆即可；note-11 引用基准 commit `00ecb5c`）
- 关键路径：`docs/zh/api/`（API 契约）、`docs/zh/design/features/`（memory_* 等设计文档）、
  `runtime/`（runtime 行为，如 memory_copy.cc F1 证据）、IR 定义 `tests/.../all_ops.cpp`（ES 算子语义，ES 无公开文档）
- 本机尚未克隆，需要时克隆到 /tmp/ge-src 或 /workspace（用完即删）
- GitHub 镜像存在但本机不通，一律走 gitcode

## 4. T3' 本机 CANN 9.1.0 头文件（签名级权威）

- `/usr/local/Ascend/cann-9.1.0/include/`：`ge/`（ge_api、es_graph_builder、graph/types）、
  `es/`（ES 构图函数原型，gen_esb 生成）、`exe_graph/runtime/`（gert::Tensor）、`acl/`、`graph/`
- 引用格式：`include/es/es_Xxx.h`

## 5. T4 随包 ops-info（逐 SoC 校验配置，未文档化）

- `/usr/local/Ascend/cann-9.1.0/opp/.../tbe/config/<soc>/*.json`（16 个 SoC 目录）
- 用途：F3 式逐产品对齐取证；进注释前须升级 T1-T3

## 6. 工程参照（T5，不进注释）

- **vllm-ascend**：`github.com/vllm-project/vllm-ascend`（本机走 gh-proxy 前缀抓文件）。
  价值：FusedInferAttentionScore/PA 在 Ascend 上的成熟接法、AscendGraph 用法。
  注意：旧机器上的本地拷贝 `/workspace/user_data/vllm-asend` 已不存在（未随项目迁移）。
- **JittorInfer**（pr11/tip/master）：已验证 mapping/workaround 对照
- **本 fork 自带的原生 ggml-cann 后端** `/root/cannge-ggml/ggml/src/ggml-cann/`：
  CANNGE 的"近亲"参照（同一 GE/acl 生态、op 映射决策、NZ 处理），随仓库本地可得，**无需网络**
- CrispASR 调优笔记 `/workspace/user_data/notes/`：msprof 方法学、NZ/TransData 结论、
  F16 约束、310 无 FA 硬件等**已实测结论**（T5 级经验，结论引用仍须回溯官方源）

## 7. 模型/数据集源

- `https://hf-mirror.com`（hf 直连不通；API/resolve 全兼容，只需换域名）
- ggml-org 部分仓库镜像缺失时，备选 unsloth/lmstudio-community/bartowski/Qwen 官方仓库

## 8. 检索纪律（继承 note-11 §2）

GE API 行为 -> T1 ascendgraphapi/options + T3 docs/zh/api
算子 dtype/shape/对齐 -> T1 aolapi 或 T2 同文；逐 SoC 细化 -> T4
SoC 算子支持性 -> T1 operatorlist_00094/95 + T4
内存契约/零拷贝/复用 -> T3 memory_*.md + runtime/
ES 构图语义 -> T3 all_ops.cpp + T3' 头文件（无公开文档）
融合副作用 -> T1 图融合规则 + T3 compiler/
