# CANNGE 服务器侧交接包（HANDOFF）

整理：2026-10-08，工作站侧 Kimi Code → 服务器侧 Kimi Code。

## 一句话项目状态

在 `ggml-org/llama.cpp` 的 fork（`KokerZhou/llama.cpp`）上实现 CANNGE 后端：
Ascend GE 整图编译后端（ES 构图 → GE 编译 → session 执行 → plan cache），
与 GGML_CANN（eager ACLNN）完全独立、可共存。分支 **`cannge`**，两个 commit：

- `d85b6bd08` ggml : add CANNGE backend（12 文件 +2105）
- `28558340a` ggml : fix CANNGE correctness findings（F1/F2/F3 + in-place；message 注明可 squash）

## 第一动作（按序，不要跳）

1. 通读本包 `cannge-docs/README.md`（索引）→ `note-10-findings.md`（F1-F8 台账）→
   `note-20-matmul-constraint-docs.md` §5（F7 实验设计）。
2. 用 PAT 对齐仓库：`GH_TOKEN=<PAT> bash scripts/01_clone_repo.sh`（token 只进
   `~/.git-credentials`，600 权限；**禁止**把 token 写进任何文件、笔记或 git remote URL）。
3. 构建冒烟：`bash scripts/02_build_and_test.sh <SOC_TYPE>`。
   SOC_TYPE：310 用 `Ascend310P3`，910 用 `Ascend910B3`。
4. 之后等用户指派任务。当前热点：F7（判别实验，设计已好在 note-20 §5.2）、
   F8（修法已定在 note-10 F8，等批准）。

## 构建/测试命令速查

```bash
# toolkit 环境（两个候选路径，哪个存在 source 哪个）
source /usr/local/Ascend/cann-9.1.0/set_env.sh 2>/dev/null || source ~/Ascend/ascend-toolkit/set_env.sh

# 构建
cmake -B build -DGGML_CANNGE=ON -DGGML_CANN=OFF -DSOC_TYPE=<SOC> -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ggml-cannge test-backend-ops -j$(nproc)

# 整图路径必须开 env 门（默认 off，零存在感）
GGML_CANNGE_GRAPH=1 ./build/bin/test-backend-ops test -b CANNGE0 -o ADD,MUL,MUL_MAT,SCALE,RMS_NORM,SOFT_MAX,SILU,GET_ROWS,CONCAT
```

基线口径（310P3 x 9.1，28558340a）：探针 11/11；TBO 160 OK / 45 FAIL / 2366 NS（总 2571）。
对比/判读规则见 `note-16-validation-spec.md`。

## 纪律（不可违反，违反即返工）

1. **未获用户明示，禁止 git commit / push**；commit message 简洁 +
   `Assisted-by: Kimi Code`；合并前 squash。
2. **禁止 abort/exit**：不支持即返回 false 交上层调度。
3. 注释规范（`note-13-code-style.md` + `note-11-doc-sources.md`）：每个 op mapping /
   GE 语义决策带一行官方出处（T1-T3 来源）；T5（JittorInfer）/T6（.so/issue 线索）
   只作线索、不进注释；注释 ASCII、1-2 行。
4. 后端基线 **CANN 9.1**（9.0 兼容 hold，见 note-17）；**950 不做**（仅限 CANNGE）；
   手写算子 hold（note-15 L5）。
5. llama.cpp/AGENTS.md 的项目规矩全部有效（先读）。
6. 凭证不落盘、不进 repo、不进笔记；共享环境（910）不动他人目录与全局配置。

## 文件地图（cannge-docs/）

| note | 内容 |
|---|---|
| note-1~8 | 前手调研：ES 管线、推理机制、设计决策、版本对照、官方 API 精读、远程测试纪律 |
| note-9 | 多 SoC x 多 CANN 版本验证矩阵（310/910 × 9.0/9.1 四轮实测） |
| note-10 | **findings 台账 F1-F8（当前活跃：F7 判据实验、F8 in-place 漏网）** |
| note-11 | 文档来源注册表（T1-T6 分级、注释引用规范） |
| note-12 | 环境运维手册（注意顶部 2026-10-08 更新：归档根已统一、910 keyring 坏） |
| note-13 | 代码风格与工程纪律 |
| note-14 | 算子接入规范（两层判定、CPY/VIEW 注册规则、已支持 op 清单） |
| note-15 | 回退层次规范（L0-L6，L3/L4b/L5 均 hold） |
| note-16 | 验证规范（基线口径、探针构成、归档格式） |
| note-17 | 兼容性与版本策略（9.1 基线、310 先行、JittorInfer 边界） |
| note-18 | 架构总览（四层管线、plan 缓存、GE 生命周期） |
| note-19 | 逐 SoC ops-info 精细静态表（16 SoC 取证） |
| note-20 | **matmul 形状约束文档裁决 + F7 最小判别实验设计（§5.2）** |

## 已知悬挂（接手前读完再动）

- **F7**：f32 m=16 n=8 家族数值错 + GE 病态编译。文档已裁决"官方无此文"（note-20），
  待做 §5.2 判别实验（关键形状 f32 m/n ∈ {8,24,40}），假设：f32→fp16 回退后按
  fp16 口径二次判对齐。
- **F8**：view_src 型 in-place（rms_norm_inplace/scale_inplace）绕过 L0 门。修法已定
  （note-10 F8：supports_op 白名单 {VIEW,PERMUTE,TRANSPOSE} + analyze 别名捷径同步限制），
  等用户批准后实现+验证。
- **归档事故**：F4/910 原始归档（report-f4.md 等）已丢失，只有 note-9/10/16 的摘要
  数字；如需原始日志需重跑。
- **910 独有**：`/home/developer/cannge-ggml` 有未入库的 9.0 shim（F5 hold，
  note-9 §5）；npussh keyring 当前 EACCES。
- `probes/` 下是幸存的 310 首轮探针源码（F4 探针已丢失，可据此重建）。
