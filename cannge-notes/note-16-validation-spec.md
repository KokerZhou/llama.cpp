# note-16: CANNGE 验证规范

建立：2026-09-11。规范对象：CANNGE 后端变更的验证流程、基线口径、双 SoC 验证与结果归档。
当前基线数字来自三轮真机验证（note-9 矩阵、report.md、report-9.1.md），**重跑后必须以
新数字刷新本文**。

## 1. 本地双构建（每次变更的必做项）

1. **容器构建**：在本地 CANN 容器（note-12 §4）内
   `cmake -B build -DGGML_CANNGE=ON -DGGML_CANN=OFF && cmake --build build --target ggml-cannge
   test-backend-ops -j`，必须零错误完成（语法/链接级验证，含 ES 头检测）。
2. **本机构建**：`llama.cpp/build-native/`（x86_64 原生，GGML_CANNGE=OFF）构建通过，
   验证上游路径无回归。
3. 两构建都绿才算完成；只绿一侧禁止进入真机阶段。

## 2. 11-op 探针（真机冒烟）

- 探针程序（ggml_init no_alloc + `ggml_backend_alloc_ctx_tensors`，env `GGML_CANNGE_GRAPH=1`）
  覆盖 11 项：add、add(2nd)（plan 缓存命中+重绑定）、mul_mat f32、rms_norm、soft_max、silu、
  get_rows、permute（消费于 add）、view(offset)、cpy cast 往返、concat。
- 通过标准（310P3 x CANN 9.1 基线）：除 mul_mat f32 与 soft_max 外全部 0 误差；
  mul_mat f32 max_rel ~1.5e-3（310P 内部 FP16 计算）；soft_max max_abs ~1.5e-8
  （report.md §3）。
- 探针全绿是进入 test-backend-ops 阶段的前置条件。

## 3. test-backend-ops 基线口径

- 命令（真机）：
  `GGML_CANNGE_GRAPH=1 ./build/bin/test-backend-ops test -b CANNGE0 -o
  ADD,MUL,MUL_MAT,SCALE,RMS_NORM,SOFT_MAX,SILU,GET_ROWS,CONCAT`（-o 过滤名大写）。
- **当前基线（F4 同口径解析器，总用例 2571；旧解析器漏计，旧数字作废）**：

  | op | 310P3 x 9.1 (28558340a) OK/FAIL/NS | 910B3 x 9.1 (d85b6bd08, 同解析器复解) OK/FAIL/NS |
  |---|---|---|
  | ADD | 52/13/36 | 52/13/36 |
  | MUL | 52/11/28 | 52/11/28 |
  | MUL_MAT | 2/4/1667 | 12/5/1656 |
  | SCALE | 1/3/0 | 1/3/0 |
  | RMS_NORM | 25/1/25 | 0/26/25 |
  | SOFT_MAX | 14/5/193 | 12/7/193 |
  | SILU | 4/0/4 | 4/0/4 |
  | GET_ROWS | 2/8/229 | 2/8/229 |
  | CONCAT | 8/0/184 | 8/0/184 |
  | **合计** | **160/45/2366** | **143/73/2355** |

  旧基线 OK=24/FAIL=83/NS=2304（report-9.1.md §3）系旧解析器漏计，已作废；两格均需以
  本表口径为准。归档：~/.kimi-private/remote-results-310-f4/（含解析器 parse_tbo_lib.py）。
  310 格 FAIL 明细：45 个数值错（含 F7 的 MUL_MAT f32 m=16 n=8 家族 2 个、F8 的
  rms_norm_inplace 1 个），清单在 tbo_fails_f4_vs_910b.json。

- NS（not supported）口径：L0 门拒绝的用例（量化 dtype、参数化 softmax、非对齐 matmul、
  非连续/view 输入等），**NS 不是失败**；对比时只看 OK 与 FAIL 的迁移。
- F1/F2/F3 修复后的首轮重采已由 F4（310，2026-09-14）完成，即上表 310 格；F2 方向吻合
  （RMS_NORM 26→25 OK，残余 1 个即 F8 inplace），F3 方向基本吻合（EZ9999 75→0，3 个
  旧 f16 未对齐数值 FAIL 转 NS；但引入 F7 过门家族与 9 个过拒，见 note-10 F7/F8）。
  910 格仍是 d85b6bd08 口径，待 F7 处置后重跑刷新。
- 失败三分法（归档时逐条归类）：构建期 "no ES tensor"（view/非连续输入未注册）、
  编译期对齐失败（EZ9999，310P 与 910B 均有）、数值错误（真 bug）。

## 4. 双 SoC 验证流程

1. **310P3**（第一目标 SoC，长连接，note-12 §2）：同步代码 → 构建 → 11-op 探针 →
   test-backend-ops 全量窄集 → 结果拉回本地归档。
2. **910B3**（第二阶段验证，npussh 快通道，note-12 §3）：同流程；910 额外关注跨 SoC
   差异项（见 §6）。
3. 两 SoC 都必须跑同一 commit；环境差异（toolkit 路径、pip 依赖）按 note-8 §3 与
   note-12 §3 处置。
4. GE 首调编译为秒级（310 实测 ~12.7s、910B ~28.3s，单次测量），plan 缓存命中 ~0.2ms；
   性能数字只作量级参考，不作回归门槛（未预热同条件对比）。

## 5. 结果归档（report 落盘格式）

- 所有真机产物**必须**拉回本地，禁止只留远程容器：
  - 310 轮：`~/.kimi-private/remote-results/`（report.md、探针源码与日志、
    test-backend-ops-run.log、remote-vs-local.diff、远程 ggml-cannge 全量拷贝）。
  - 910 轮：`~/.kimi-private/remote-results-910b/`（report-9.1.md、
    test-backend-ops-910b-91.log、探针与日志）。
- 报告必须含：构建结果、远程相对本地的文件改动清单+理由（附本地路径）、11-op 探针
  逐项数值、test-backend-ops 逐 op OK/FAIL/NS 表、失败三分法归类、GE 关键日志、
  远程/本地环境差异、跨 SoC 对比结论。
- 验证结束必须做远程 vs 本地逐文件 diff（910 轮纪律：验证后远程 ggml-cannge/ 与本地
  完全一致，report-9.1.md）；远程临时改动要么落回本地要么还原。

## 6. 已知精度特性（对比判读时必须扣除）

| 特性 | 310P3 | 910B3 | 出处 |
|---|---|---|---|
| MUL_MAT F32 计算精度 | 降 FP16 计算，max_rel ~1.5e-3 | 真 FP32，max_rel ~3.7e-7——**机制已查明**（note-19 §4.1/§7.2）：310p TBE 仅注册 fp16、F32 经精度回退降 FP16；910b 注册 f16/f32/bf16，且 GE 图路径 matmul HF32 默认关闭（ge.exec.allow_hf32 对 Matmul 类默认不使能 = T1 原文，三重确认） | note-7 §3 precision_mode 官方语义；cubeMathType 语义见 note-19 §7.1（aclnn eager 层默认值仍未文档化，属 aclnn spec 缺口而非 open item） |
| 输出-输入别名写回 | GE 允许 | **静默丢弃**（F1 staging 修复后由后端规避） | note-10 F1；ge 仓 memory_copy.cc:574-579 @00ecb5c |
| matmul 32B 对齐 | 强制（EZ9999） | 同样强制 | note-10 F3，"未文档化，依据见 note-10 F3" |
| RMS_NORM 融合核 | 小 shape 间歇丢行（故用元语实现） | 未复现（元语实现下通过） | note-8 坑表、note-10 F6 |

- GEInitialize 配置 `ge.exec.precision_mode=allow_fp32_to_fp16`（ge.cpp:67）；注意在线推理
  官方默认实为 force_fp16（note-7 §3），两取值列入 310 实测对比清单。
- 数值容差：F32 图在 310P 上必须按"内部降精度"判读容差；分解式（L2）与原生核之间的
  fp16 舍入差异属已知代价（note-3 D1），验证时按 op 标注。

## 7. 待核实清单

| # | 条目 | 现状 |
|---|---|---|
| 1 | F1/F2/F3 修复后的新基线（310 与 910B 两格） | 待修复合入并重跑（note-10 F4） |
| 2 | ~~910B MUL_MAT F32 真 FP32（rel 3.7e-7 优于 HF32）的机制~~ | 已查明（note-19 §7.2）：GE 图路径 matmul HF32 默认关闭（allow_hf32 默认不使能 Matmul 类 = T1 原文 + enable_hf32 attr 默认 false + impl_mode ini 三重确认） |
| 3 | ~~GE 图路径 cubeMathType 默认值~~ | 已答（note-19 §7.2）：GE 图路径无 per-op cubeMathType，精度由 allow_hf32/enable_hf32/precision_mode 三层控制且默认 HF32 关；aclnn eager 层默认值未文档化（spec 缺口） |
| 4 | precision_mode=allow_fp32_to_fp16"矩阵类算子使用float16"与 910B 实测 F32 保留的字面矛盾 | 待核实（note-19 §7.3）；以实测 + 默认值为判读基准 |
| 4 | precision_mode 两种取值在 310 的对比测量 | 列入 310 实测清单（note-7 §8） |
