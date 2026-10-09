# note-9: 多 SoC x 多 CANN 版本验证矩阵

整理日期：2026-09-14（F4 后更新）。实测数据来自四轮真机验证：310P3 轮（`~/.kimi-private/remote-results/`）、
910B3 x 9.0 轮与 910B3 x 9.1 轮（`~/.kimi-private/remote-results-910b/`）、310P3 F4 重跑轮
（`~/.kimi-private/remote-results-310-f4/`，28558340a 修复后口径）。
代码基线：commit `d85b6bd08`（含 310 轮 7 个修复；**该代码最低要求 CANN 9.1**——9.0 缺 4 个 ES 封装，
当时用临时 shim 验证，shim 未落库，向前兼容留待后续决策）。

## 1. 总览矩阵

|  | CANN 9.0.0 | CANN 9.1.0 |
|---|---|---|
| **Ascend 310P3** | 无环境，未测（代码不兼容，需 shim） | ✅ d85b6bd08：探针 11/11，TBO 窄集 134/216；✅ 28558340a（F4 重跑）：探针 11/11，TBO 160/45/2366 新口径，新发现 F7/F8 |
| **Ascend 910B3** | ⚠️ shim 后构建通过，探针 9/11 | ⚠️ 零改动构建通过，探针 9/11（与 9.0 同口径） |

图例：✅ 全链路验证通过；⚠️ 构建通过、运行层有未决项；空白 = 无环境。
注：910B 的两个探针未过项（PERMUTE/CPY cast）在 9.0 与 9.1 上失败方式完全一致、而 310 全过——
**定性为 SoC 差异**（见 §3），与 CANN 版本无关。

## 2. 各格明细

### 310P3 x CANN 9.1.0（driver 26.0.rc1，2 卡，44GB HBM/卡）
- 构建：cmake SOC 自动识别，零结构改动；aarch64 Ubuntu 22.04。
- GE 链路：ES 构图 → AddGraphWithCopy → CompileGraph → LoadGraph → ExecuteGraphWithStreamAsync 全通。
- 探针 11/11：ADD/RMS_NORM/SILU/GET_ROWS/PERMUTE/VIEW(offset)/CPY(cast 往返)/CONCAT 误差 0；SOFT_MAX 1.49e-8；MUL_MAT F32 max_rel 1.47e-3（**内部降 FP16 计算**，符合 allow_fp32_to_fp16 官方语义）。
- 性能：GE 首调编译 12.7s；plan 缓存命中后 0.2ms。
- test-backend-ops 窄集 134/216；FAIL 三类：view/非连续输入未注册（~880 条 build 期 "no ES tensor"）、MUL_MAT 非 32B 对齐编译失败（EZ9999）、82 个数值失败（含 4D RMS_NORM）。
- 关键坑：310P matmul 维度须 32B 对齐；MatMulV3 无 infer_datatype 注册（→ BatchMatMulV3）；RmsNorm 融合核小 shape 间歇丢行（→ 元语实现）；GEInitialize 需 pip 装 numpy<2 等依赖。

### 310P3 x CANN 9.1.0 — F4 重跑（2026-09-14，commit 28558340a，F1/F2/F3+in-place 后）
- 构建：全量 2m17s 零错误；同步前后 ggml-cannge/ 8 文件 md5 全同。
- 探针 11/11，与基线逐项一致（首调 12.8s、命中 0.1ms；mul_mat f32 max_rel 1.465e-3、
  soft_max max_abs 1.49e-8，其余 0 误差）。
- test-backend-ops 新口径（同解析器总用例 2571）：OK=160 / 数值 FAIL=45 / NS=2366。
  F2 方向吻合（RMS_NORM 26→25 OK）；F3 基本吻合（EZ9999 75→0）；三分法：数值错 45、
  "no ES tensor" 851、requires-staging 44、EZ9999 0、CheckDims 残留 2、E13025 执行期 5。
- **新发现**：F7（MUL_MAT f32 m=16 n=8 家族数值错 + GE 编译 >10min 病态）、F8
  （view_src 型 in-place 绕过 L0 门，rms_norm_inplace/scale_inplace 仍错）——详见
  note-10 F7/F8。
- 归档：~/.kimi-private/remote-results-310-f4/；收尾时连接中断，远程复现进程
  PID 37148 待清理。

### 910B3 x CANN 9.0.0（driver 25.5.0，1 卡，64GB HBM）
- 构建：✅ 通过，但需 **ES 兼容性回退**（try_compile 特性探测）：9.0.0 缺 `Reshape`/`Swish`/`Permute`/`EsCreateConstV2` ES 封装 → Silu 回退 Mul+Sigmoid、Permute 回退 Transpose、RMS axes 用 CreateConst。对 9.1 路径零影响（条件编译）。
- GE 链路：同 310，全通（Session + ExecuteGraphWithStreamAsync）。
- 探针 9/11：ADD 0/0、**MUL_MAT F32 max_abs=1.19e-07 / max_rel=3.7e-07（真 FP32，HF32 路径未触发）**、RMS_NORM 0/0、SOFT_MAX 5.96e-8、SILU 0/0、GET_ROWS 0/0、VIEW 0/0、CONCAT 0/0。
- 性能：首调编译 28.3s；缓存命中 0.2ms。
- 未决 2 项：
  1. **PERMUTE 输出=输入（线性拷贝）**：GE 图 dump 证实 Transpose 节点接线与 perm 常量 [1,2,0] 均正确，但输出未变——疑似 **9.0.0 EsTranspose 静默不生效**（与 310 RmsNorm 丢行同类的 GE 静默错误）。复现程序在远程 `~/es_transpose_repro.cpp`。
  2. CPY cast 全 0：疑似探针自身 bug + dst 被 analyze 注册成图输入，待查（310 同款代码通过）。

### 910B3 x CANN 9.1.0（driver 25.5.0，1 卡，64GB HBM；2026-09-10 环境升级后实测）
- 构建：✅ **零改动通过**（同一份 d85b6bd08），坐实后端最低版本要求 = CANN 9.1。
- 探针 9/11：与 9.0 轮完全同口径，**PERMUTE/CPY cast 两项失败方式与 9.0 完全一致**。
- test-backend-ops：OK=24 / 数值 FAIL=83 / NS=2304（NS 多为窄集外的 dtype/shape 组合）。
- 金丝雀实验：GE 执行后输出缓冲仍是写入前的值（写 99 读回 99）——**910B 静默丢弃"输出绑定地址与图输入缓冲重叠"的写回，310P3 允许**。CPY 链中间张量全 0 同因（dst 被 analyze 误注册为输入，地址重叠）。`reuseZeroCopyMemory` 0/1 无影响。
- 跨 SoC：MUL_MAT F32 真 FP32（rel 3.7e-7）；**910B 同样有 310P 的 matmul 32B 对齐限制**（50 条 EZ9999）；RMS_NORM 4D 两 SoC 同错（后端 4D 轴 bug，SoC 无关，占 26 个 FAIL）；GE 首调 28.3s vs 310 的 12.7s，缓存命中均 0.2ms。

## 3. 跨 SoC / 跨版本关键差异（实测确认）

| 维度 | 310P3 (CANN 9.1) | 910B3 (CANN 9.1) | 出处/含义 |
|---|---|---|---|
| 输出-输入别名写回 | 允许 | **静默丢弃** | SoC 级 GE 行为差异；后端必须避免输出端口绑定到与图输入重叠的地址（analyze 标记 CPY dst 为产出 + 重叠检测，下阶段修） |
| MUL_MAT F32 精度 | max_rel ~1.5e-3（内部 FP16） | max_rel ~3.7e-7（真 FP32） | note-7 §3 precision_mode 语义实锤 |
| matmul 32B 对齐 | 强制（EZ9999） | **同样强制**（50 条） | 非 310 特有，supports_op 静态表两个 SoC 都要查 |
| RmsNorm 融合核 | 小 shape 间歇丢行 | 未复现（元语实现下通过） | 元语实现两头安全 |
| EsTranspose 静默不生效 | 未发现（PERMUTE 通过） | PERMUTE 未过但根因是别名写回丢弃（非 Transpose 本身） | 9.0 轮的怀疑被推翻 |
| RMS_NORM 4D | 数值错 | 同错 | **后端 4D 轴 bug，SoC 无关**，优先修 |
| ES 封装完整性 | 全 | 全（9.0 缺 4 个，故最低版本 9.1） | 版本差异 |
| GE 首调编译耗时 | ~13s | ~28s | 图规模/芯片不同，量级参考 |

## 4. 已知空白与填充路径

| 空白格 | 填充路径 |
|---|---|
| 310 x CANN 9.0 | 910 容器镜像装 310P 工具链，或等 310 机器降版本（优先级低） |
| 910 x CANN 9.1 | 910 机器用户级再装 cann-9.1（~/Ascend 下并行安装，不动 9.0），重点验证 9.0 两个未决项是否随版本消失 |
| x86 host 运行 | 无需求，CANNGE 只跑 device 侧 |

## 5. 待落盘的代码资产

- 910 轮 ES 兼容性修复（远程 `/home/developer/cannge-ggml`：CMakeLists try_compile 探测 + graph-build.cpp 条件回退 + supports_op 拒 RESHAPE）——**未落回本地**，下次恢复时 diff 落回并 amend。
- 310 轮 7 个 bug 修复已落回并包含在 d85b6bd08。
