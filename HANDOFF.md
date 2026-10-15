# CANNGE 服务器侧交接包 v2（HANDOFF）

整理：2026-10-14（v2），工作站侧 Kimi Code → 910 服务器侧 Kimi Code。
v1（2026-10-08, 310）由你的前任（310 服务器侧 Kimi）执行完毕，产出已合入：
代码 `cannge` 分支前进到 `d3a3ef81f`（2 新 commit），笔记 note-15 更新 + note-21~29 八篇
（本包 `cannge-docs/` 已是合并后的完整树）。

## 你的身份与工作模式

你在 910 开发机（1×910B3，CANNLab 环境 `DevEnvC_Abwb9`）上**自主推进** CANNGE 项目。
工作站上另有一个 Kimi Code（协调者），负责文档/git/收敛；你负责 910 上的构建、
测试、调查与修复。纪律相同：**commit/push 必须用户明示**（batch 模式：改码→构建→
TBO/冒烟验证→落盘 note，一批一验，控制变量）。

## 第一动作（按序）

1. 通读 `cannge-docs/README.md` → `note-27-failure-chain.md`（17 环根因链 + 当前状态）
   → `note-24-fix-log.md`（F9-F13+）→ `note-25-model-integration.md`（模型集成，含 #8
   病态编译复发）。
2. 仓库对齐：`GH_TOKEN=<PAT> bash scripts/01_clone_repo.sh`（预期 HEAD=`d3a3ef81f`）。
   注意：旧目录 `/home/developer/cannge-ggml.rsync-era-backup` 是 rsync 时代裸副本
   （内含未入库的 9.0 shim，note-9 §5，F5 hold），不要直接用它构建。
3. 构建冒烟：`bash scripts/02_build_and_test.sh Ascend910B3`。
   **910 特有口径**：f32 matmul 门走 `#else` 分支（8 元素对齐，非 310P 的 16）；
   病态编译（note-25 #8）在 910B 的形状口径可能不同于 310P，结论须标注 SoC。
4. 向用户汇报环境与基线，然后按"当前工作队列"领任务。

## 构建/测试命令速查

```bash
source /home/developer/Ascend/ascend-toolkit/set_env.sh
cmake -B build -DGGML_CANNGE=ON -DGGML_CANN=OFF -DSOC_TYPE=Ascend910B3 -DCMAKE_BUILD_TYPE=Release \
     -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF   # UI 下载 github 不通会拖死构建（note-27）
cmake --build build --target ggml-cannge test-backend-ops -j$(nproc)
GGML_CANNGE_GRAPH=1 ./build/bin/test-backend-ops test -b CANNGE0 -o ADD,MUL,MUL_MAT,SCALE,RMS_NORM,SOFT_MAX,SILU,GET_ROWS,CONCAT
```

冒烟（模型级）注意事项（全部来自 note-27/note-25 的教训）：
- llama-cli 冒烟 stdout 必须截断防 `"> "` EOF 洪泛写爆磁盘；
- 对拍参考文件先验证"同配置自比=100%"再采信；KL 解析用 C++ 工具，不信 python 快写解析器；
- GE 编译卡死判别：看 `ge_davidmdlrun` 线程是否全闲（全闲=编译未下发，病态在 host 侧）。

## 当前工作队列（按优先级，等用户确认后开工）

1. **冒烟续跑（note-27 #13/#14）**：d3a3ef81f 已含 set_rows host/dst 修复，重跑 Qwen3-4B
   冒烟过 execute 阶段；乱码则按算子二分（embedding GET_ROWS / f16 matmul Cast / RMS_NORM）。
2. **病态编译二分定位（note-25 #8）**：先加编译起点日志（图签名/节点数），命中后按
   ubatch/子图二分定位触发形状。**这是 310P 家族问题在 910B 上的对照观察点**。
3. **RMS-in-graph 根因调查（note-27 #15，主破坏者）**：孤立 RMS 精确、进图即坏、
   DISABLE_ALL 也在场即破坏。`GGML_CANNGE_NO_RMS` 旋钮现成。怀疑方向：图集成
   （绑定/别名/ staging 交互），不是 RMS kernel 本身。
4. TBO 910 基线对拍（310 口径 324/333，note-27）→ 填 note-9 矩阵 910×9.1 格（新代码）。
5. 以上收敛后跑 note-29 覆盖矩阵（36 次生成）。

## 纪律（与 v1 相同，摘录）

- 未获用户明示禁止 commit/push；message 简洁 + `Assisted-by: Kimi Code`；
- 禁止 abort；注释按 note-11/note-13 带官方出处、ASCII、1-2 行；T5/T6 不进注释；
- 凭证不落盘文件、不进 repo、不进笔记（`~/.git-credentials` 600 除外）；
- 记录格式沿用 note 体系（note-30 起编号，别覆盖 1~29）；每轮结束把成果推
  `cannge-handoff-notes` 式笔记分支或交给协调者合入。

## 文件地图

见 `cannge-docs/README.md`（note-21~29 为 310 轮产出，note-1~20 为更早轮次）。
