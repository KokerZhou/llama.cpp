# note-29: Qwen 全量覆盖实验矩阵（CANNGE，执行方案设计稿）

> 只读设计任务，**当前不构建/不运行/不碰 NPU**。本文件是一份可落地的执行方案，用于在修复 note-27 #13 之后，系统性地验证 CANNGE 对 4 个 Qwen 模型的覆盖情况。
>
> 矩阵规模：4 模型 × 3 配置 × 3 条固定 prompt = **36 次实际生成运行**（可复用 page cache 把加载次数降到 12 次）。

---

## 1. 环境与固定参数

| 环境项 | 说明 |
| --- | --- |
| 二进制 | `/root/cannge-ggml/build/bin/llama-cli` |
| 后端使能 | 必须 `export GGML_CANNGE_GRAPH=1`，否则 L0 gate 对所有 op 返回 false（`ggml-cannge.cpp:298-303`） |
| 设备 | 310P 单卡 44 GB HBM（MoE 27 GB 文件可单卡装载） |
| 固定参数 | `-c 512 -ub 512 --temp 0.6 -fa off --no-warmup --single-turn --no-display-prompt` |
| VL 额外 | `-c 2048`（预留 image token），`-mm /workspace/models-cannge/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf --image <测试图>` |
| 输出捕获 | `> run-<id>.log 2>&1`，避免 `>` EOF 洪泛（note-27） |

参数名来源：`tools/cli/README.md` / `common/arg.cpp`。

- `-ngl, --gpu-layers N` / `all`：offload 层数
- `-nkvo, --no-kv-offload`：KV cache 留在 CPU
- `-c, --ctx-size`：上下文长度
- `-n, --predict`：生成 token 数
- `-ub, --ubatch-size`：物理 batch 大小
- `--temp, --temperature`：采样温度
- `-m, --model`：模型路径
- `-mm, --mmproj`：VL projector 路径

---

## 2. 当前 L0 门支持的 op 集合

来源：`/root/cannge-ggml/ggml/src/ggml-cannge/ggml-cannge.cpp:860-1149`。

**已支持（F16/F32 且满足各自约束）**：

- `NONE` / 元数据：`GGML_OP_NONE`
- 四则/缩放：`ADD`, `SUB`, `MUL`, `DIV`, `SCALE`
- 归一化：`NORM`, `RMS_NORM`
- 激活/一元：`UNARY` 子集 `SILU/ABS/SGN/NEG/TANH/RELU/SIGMOID/GELU/GELU_ERF/GELU_QUICK/EXP/FLOOR/CEIL/ROUND/TRUNC`
- 矩阵乘：`MUL_MAT`（要求 src0/src1 同 dtype、M/N/K 按 16/32 对齐、稠密、batch 维匹配）
- 形状/视图：`RESHAPE`, `PERMUTE`, `TRANSPOSE`, `VIEW`（稠密）, `CONT`（稠密）
- 拷贝/重复：`CPY`（F16/F32 同形）, `DUP`, `GET_ROWS`（索引 I32）, `SET_ROWS`（I32/I64 索引）
- 拼接/GLU：`CONCAT`, `GLU`（单输入 `REGLU/GEGLU/SWIGLU/...`）
- 注意力元语：`SOFT_MAX`（无 attention sink、无 ALiBi bias）, `ROPE`（仅 `NORMAL`/`NEOX`、无 freq scale/ext/YaRN、无 src2）

**与本矩阵直接相关的未支持 op（默认返回 false）**：

- `MUL_MAT_ID` / `ADD_ID`：MoE 专家路由矩阵乘
- `ARGSORT`：MoE top-k 专家选择
- `CONV_2D` / `POOL_2D`：VL 图像 patch embed
- `ROPE` mode `VISION`：VL 视觉 RoPE
- 任何 BF16 输入：dtype gate 仅认 F16/F32（`ggml_cannge_float_dtype`，L853）

> 关键推论：**decode 阶段 `MUL_MAT` 的 token 维 `ne[1]=1`，不满足 16 元素对齐门，会落 CPU；只有 prefill/ubatch token 数 ≥16 时才可能上 CANNGE。**

---

## 3. 模型清单与关键属性

| 模型 | 文件 | 大小 | dtype | 显存单卡可行性 | 特殊结构 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-4B-Instruct | `Qwen3-4B-Instruct-2507-F16.gguf` | 7.5 GB | F16 | 轻松 | Dense decoder |
| Qwen1.5-MoE-A2.7B | `qwen1.5-moe-a2.7b-chat-f16.gguf` | 27 GB | F16 | 44 GB 单卡可装（≈28.6 GB 理论） | MoE：MUL_MAT_ID + ARGSORT |
| Qwen2.5-VL-7B | `Qwen2.5-VL-7B-Instruct-f16.gguf` | 15 GB | F16 | 可装 | VL：需 `mmproj` + `--image`，视觉侧含 CONV_2D + VISION RoPE |
| Qwen3-0.6B | `Qwen3-0.6B-BF16.gguf` | 1.2 GB | BF16 | N/A | **基线对照**：CANNGE 仅支持 F16/F32，预计全 CPU |

---

## 4. 固定 Prompt 集

| ID | 类型 | Prompt 文本 | -n |
| --- | --- | --- | --- |
| P1 | 中文短句 | 你好，请用一句话介绍自己。 | 40 |
| P2 | 英文短句 | Hello, describe the flavor of a fresh apple in one sentence. | 40 |
| P3 | 长文段 | 下面是一段关于人工智能发展的介绍。人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。近年来，深度学习和大规模语言模型取得了显著进展，推动了自然语言处理、计算机视觉和语音识别等领域的发展。请用简洁的中文总结上述内容。 | 120 |

VL 模型使用相同 3 条 prompt，但均附加 `--image /workspace/shared_assets/datasets/test-image.jpg`（测试图需提前准备，建议 512×512 JPG）。

---

## 5. 模型 × 配置矩阵

配置定义：

- **基线**：`-ngl all`，KV offload 默认开启
- **-nkvo**：`-ngl all -nkvo`，KV cache 强制 CPU 对照
- **CPU 参考**：`-ngl 0`，纯 CPU（预期极慢）

| 编号 | 模型 | 配置 | 关键 CLI 差异 | 预期主要执行路径 | 关键观察点 |
| --- | --- | --- | --- | --- | --- |
| 1 | Qwen3-4B | 基线 | `-ngl all` | 权重上 HBM；decode MUL_MAT 因 M=1 对齐门拒落 CPU；prefill MUL_MAT 可能上 CANNGE；其余 op 尽量走 GE | 首 token 编译时间、是否乱码、是否触发 #13/#7 家族 |
| 2 | Qwen3-4B | -nkvo | `+ -nkvo` | KV 缓存全在 CPU，SET_ROWS 回写方向变为 H2D/D2H | 对照验证 SET_ROWS 修复后是否 still 乱码 |
| 3 | Qwen3-4B | CPU 参考 | `-ngl 0` | 全 CPU，极慢 | 仅跑 P1 即可，作为“能跑但慢”下限 |
| 4 | Qwen1.5-MoE | 基线 | `-ngl all` | MoE 路由（ARGSORT/MUL_MAT_ID/ADD_ID）整体落 CPU；dense attention 部分同 Qwen3 | 观察大量 CPU fallback 子图切分、首 token 编译是否更长 |
| 5 | Qwen1.5-MoE | -nkvo | `+ -nkvo` | KV CPU + MoE CPU，权重仍上 HBM | 验证 MoE 在 CPU fallback 下的正确性 |
| 6 | Qwen1.5-MoE | CPU 参考 | `-ngl 0` | 全 CPU | 文件 27 GB，加载慢；可只做 P1 |
| 7 | Qwen2.5-VL-7B | 基线 | `-ngl all -mm <mmproj> --image <图>` | 视觉 encoder（CONV_2D、VISION RoPE）落 CPU；语言 decoder 同 Qwen3 | 验证 mmproj 加载、图像 token 注入、视觉侧 fallback 是否导致 SMMU/拷贝错误 |
| 8 | Qwen2.5-VL-7B | -nkvo | `+ -nkvo` | 视觉 CPU + KV CPU + 语言 decoder 部分 CANNGE | 对照 KV 路径 |
| 9 | Qwen2.5-VL-7B | CPU 参考 | `-ngl 0` | 全 CPU | 加载 15 GB + mmproj；可只做 P1 |
| 10 | Qwen3-0.6B | 基线 | `-ngl all` | **全部 CPU**：权重 dtype BF16 不被 L0 接受 | 验证 prompt/CLI 正确性、记录 CPU 参考速度 |
| 11 | Qwen3-0.6B | -nkvo | `+ -nkvo` | 同基线（-nkvo 对全 CPU 无意义） | 与基线结果应一致 |
| 12 | Qwen3-0.6B | CPU 参考 | `-ngl 0` | 同基线 | 作为最小模型的 CPU 基线 |

每个编号需依次跑 P1/P2/P3，共 **36 次生成**。

---

## 6. 各模型预期 op 覆盖差异

| 模型 | 关键 op | L0 当前支持？ | 预期落点 | 影响 |
| --- | --- | --- | --- | --- |
| Qwen3-4B dense | `MUL_MAT`（decode M=1） | 否（对齐门） | CPU | decode 阶段主矩阵乘在 CPU，t/s 受限 |
| Qwen3-4B dense | `MUL_MAT`（prefill N≥16） | 可能 | CANNGE | 长 prompt prefill 可能受益 |
| Qwen3-4B dense | `ROPE`, `SOFT_MAX`, `RMS_NORM`, `SET_ROWS` | 是（满足约束） | CANNGE | 已修复 F16/F16，注意 #13 execute 方向 |
| Qwen1.5-MoE | `MUL_MAT_ID`, `ADD_ID` | 否 | CPU | MoE FFN 整体 CPU 计算，权重即使上 HBM 也要反复拷入/拷出 |
| Qwen1.5-MoE | `ARGSORT`（top-k 路由） | 否 | CPU | 路由图在 CPU，与专家权重拷贝形成边界 |
| Qwen2.5-VL | `CONV_2D`（patch embed） | 否 | CPU | 视觉 encoder 全部 CPU |
| Qwen2.5-VL | `ROPE` mode `VISION` | 否 | CPU | 视觉 self-attention 位置编码 CPU |
| Qwen2.5-VL | 语言 decoder `MUL_MAT` 等 | 同 Qwen3-4B | 部分 CANNGE / 部分 CPU | 与 dense 模型类似，但输入来自视觉 CPU 输出，增加跨后端拷贝 |
| Qwen3-0.6B | 所有含 BF16 的 op | 否 | CPU | **全量 CPU 基线**，用于验证 prompt 与 CPU 路径正确性 |

---

## 7. 记录项与失败分类

每次运行必须记录以下字段（可用下方模板）。

| 字段 | 说明 |
| --- | --- |
| RunID | 如 `q3-4b-base-p1` |
| 生成文本（前 200 字） | 用于判断是否乱码 |
| 乱码判定 | 是/否；标准是连续出现无意义字符、中英文混杂异常、重复单字 |
| 首 token 时延 | 从 `llama_decode` 到首个 token 输出（秒） |
| 总耗时 | 进程 wall time（秒） |
| t/s | 生成速度（`n / (总耗时 - 首 token 时延)` 或 llama-cli 自带 timings） |
| stderr 一级错误 | 第一个非 INFO/WARN 的 ggml/ACL/GE 错误 |
| stderr 二级错误 | 后续级联错误（可能有毒化现象，按 note-27 抓首犯） |
| note-27 对应 ID | F14-F17 / #1-#13 中的编号 |
| 备注 | 编译卡死、超时、OOM 等 |

### 失败分类（按 note-27）

| 分类 | 典型症状 | 对应 note-27 ID | 处理建议 |
| --- | --- | --- | --- |
| A. 拷贝/指针方向 | `sdma copy error`、D2D dst=host、SMMU Terminate | F14, #12, #13 | 立即停跑，先修 set_rows / cpy 方向 |
| B. Offload 配置 | `-ngl` 被忽略、`no usable GPU` | F15a | 检查 device type=GPU、`-ngl` 参数 |
| C. 宿主内存/OOM | 加载期 OOM 137、staging pinned 页暴涨 | F15b | 降低并发、检查同步化是否生效 |
| D. 算子门拒 | `abort: cache_k_l0 cannot run SET_ROWS` | F16 | 检查 L0 条件（dtype/idx/连续） |
| E. view/reshape 边界 | `no ES tensor`、`view wraps a base dim`、GE E19999 | F17a/b, #8-#10 | 检查 view 链、范围盒算法 |
| F. GE 病态编译/卡死 | 单线程 100% CPU 空转、无 GE runtime 活动、EH0008/EZ9999 | F7 家族, #8 | 设 20-30 min 超时，打印图签名/节点数定位 |
| G. GE 执行期错误 | execute 阶段报错、graph 执行失败 | #13 等 | 按时间线抓首犯 |
| H. 输出质量 | 生成可解码但语义混乱/重复/乱码 | note-25 第 4 点 | 按算子二分：embedding GET_ROWS / f16 matmul Cast / RMS_NORM |

---

## 8. 执行顺序与总耗时估算

### 建议顺序

| 阶段 | 运行编号 | 目的 | 前置条件 |
| --- | --- | --- | --- |
| 0. 校验基线 | 10-12（0.6B 三组配置，仅 P1） | 验证 CLI/参数/prompt 不报错；拿到 CPU 速度基线 | 无 |
| 1. CPU 对照 | 3（Qwen3-4B -ngl0 P1） | 确认 dense 模型 CPU 路径可出 token | 无 |
| 2. Dense 首通 | 1（Qwen3-4B 基线 P1） | **首个 CANNGE 端到端冒烟**：验证 #13 修复后 execute 不卡 | note-27 #13 已修复 |
| 3. Dense 对照 | 2（Qwen3-4B -nkvo P1） | 验证 KV CPU 对照 | 阶段 2 通过 |
| 4. Dense 完整 | 1/2/3 × P2/P3 | prompt 长度与 decode 覆盖 | 阶段 2-3 通过 |
| 5. MoE 首通 | 4（MoE 基线 P1） | 验证 MUL_MAT_ID/ARGSORT CPU fallback 不毒化会话 | 阶段 2 通过 |
| 6. MoE 对照 | 5/6 P1 | -nkvo / CPU 参考 | 阶段 5 通过 |
| 7. MoE 完整 | 4/5/6 × P2/P3 | 完整覆盖 | 阶段 6 通过 |
| 8. VL 首通 | 7（VL 基线 P1） | 验证 mmproj + image + 视觉 fallback | 阶段 2 通过 |
| 9. VL 对照/完整 | 8/9 及 7/8/9 × P2/P3 | 完整覆盖 | 阶段 8 通过 |

### 耗时估算（单条 prompt，含加载+编译+生成）

> 以下按“未命中病态编译”估算；若命中 F7 家族，单次可能 >30 min 且不收敛，需要手动超时。

| 模型 | 单次加载 | 单次 GE 编译 | 单次生成（40/120 tokens） | 单配置 3 prompts | 3 配置合计 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B BF16 | 5 s | N/A（全 CPU） | 5-15 s | ~30 s | ~1 min |
| Qwen3-4B F16 | 60-90 s | 3-15 min | 30-90 s | 5-18 min | 15-60 min |
| Qwen1.5-MoE F16 | 3-4 min | 10-30 min | 1-3 min | 15-45 min | 1-2.5 h |
| Qwen2.5-VL-7B F16 | 2-3 min | 10-25 min | 1-3 min | 12-35 min | 1-2 h |

**全矩阵保守总耗时：约 6-12 小时**；若发生 GE 病态编译，需按 20-30 min 超时截断，实际可能更长。

### 效率优化

- 同一模型同一配置的 P1/P2/P3 建议放在同一脚本里串行跑，利用 OS page cache 减少重复加载。
- 每个进程必须带 `--single-turn -n <N>`，避免进入交互模式产生 `>` 洪泛。
- 对 CPU 参考配置（编号 3/6/9），可以只做 P1，节省大量时间；P2/P3 在基线通过后再补。

---

## 9. 风险与前置

| 风险 | 说明 | 缓解 |
| --- | --- | --- |
| note-27 #13 未落地 | 当前 execute 阶段卡在 `SET_ROWS` dst 宿主分支；若未修复，所有基线跑都会复现 D2D dst=host | **必须先合入 #13 修复再启动矩阵** |
| GE 病态编译（F7） | Qwen3-4B 已观察到首模型图单线程空转 18 min；MoE/VL 图更大，风险更高 | 每跑设 20-30 min 超时；打印编译起点日志；优先跑最小模型 |
| BF16 全 CPU | Qwen3-0.6B 的权重 dtype 不被 L0 接受，不要误报为“CANNGE 失败” | 明确标记为“CPU 基线对照” |
| MoE CPU fallback 毒化 | MUL_MAT_ID/ARGSORT  unsupported 会生成大量 CPU 子图，边界拷贝可能触发 F14/#13 类错误 | 先跑 0.6B/Qwen3-4B 验证基础拷贝通路 |
| VL 视觉侧 unsupported | CONV_2D / VISION RoPE 在 CPU，视觉输出到语言 decoder 的跨后端拷贝是新的边界 | 使用小图 512×512，降低视觉子图规模 |
| 磁盘/stdout 洪泛 | 若 `-n` 未设置或进入交互，会秒级写爆磁盘 | 每次命令必须显式 `-n` 并重定向到文件 |

---

## 10. 命令模板

### Qwen3-4B 基线（P1）

```bash
export GGML_CANNGE_GRAPH=1
/root/cannge-ggml/build/bin/llama-cli \
  -m /workspace/models-cannge/Qwen3-4B-Instruct-2507-F16.gguf \
  -ngl all \
  -c 512 -ub 512 --temp 0.6 -fa off --no-warmup \
  --single-turn --no-display-prompt \
  -p "你好，请用一句话介绍自己。" -n 40 \
  > run-q3-4b-base-p1.log 2>&1
```

### Qwen3-4B -nkvo

```bash
/root/cannge-ggml/build/bin/llama-cli \
  -m /workspace/models-cannge/Qwen3-4B-Instruct-2507-F16.gguf \
  -ngl all -nkvo \
  -c 512 -ub 512 --temp 0.6 -fa off --no-warmup \
  --single-turn --no-display-prompt \
  -p "你好，请用一句话介绍自己。" -n 40 \
  > run-q3-4b-nkvo-p1.log 2>&1
```

### Qwen3-4B CPU 参考

```bash
/root/cannge-ggml/build/bin/llama-cli \
  -m /workspace/models-cannge/Qwen3-4B-Instruct-2507-F16.gguf \
  -ngl 0 \
  -c 512 -ub 64 --temp 0.6 -fa off --no-warmup \
  --single-turn --no-display-prompt \
  -p "你好，请用一句话介绍自己。" -n 40 \
  > run-q3-4b-cpu-p1.log 2>&1
```

### Qwen1.5-MoE 基线（P1）

```bash
/root/cannge-ggml/build/bin/llama-cli \
  -m /workspace/models-cannge/qwen1.5-moe-a2.7b-chat-f16.gguf \
  -ngl all \
  -c 512 -ub 512 --temp 0.6 -fa off --no-warmup \
  --single-turn --no-display-prompt \
  -p "你好，请用一句话介绍自己。" -n 40 \
  > run-moe-base-p1.log 2>&1
```

### Qwen2.5-VL 基线（P1）

```bash
/root/cannge-ggml/build/bin/llama-cli \
  -m /workspace/models-cannge/Qwen2.5-VL-7B-Instruct-f16.gguf \
  -mm /workspace/models-cannge/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf \
  --image /workspace/shared_assets/datasets/test-image.jpg \
  -ngl all \
  -c 2048 -ub 512 --temp 0.6 -fa off --no-warmup \
  --single-turn --no-display-prompt \
  -p "描述这张图片的内容。" -n 40 \
  > run-vl-base-p1.log 2>&1
```

### Qwen3-0.6B BF16（CPU 基线，P1）

```bash
/root/cannge-ggml/build/bin/llama-cli \
  -m /workspace/models-cannge/Qwen3-0.6B-BF16.gguf \
  -ngl 0 \
  -c 512 -ub 64 --temp 0.6 -fa off --no-warmup \
  --single-turn --no-display-prompt \
  -p "你好，请用一句话介绍自己。" -n 40 \
  > run-q3-06b-cpu-p1.log 2>&1
```

---

## 11. 附录：记录模板

```markdown
| RunID | 模型 | 配置 | Prompt | 生成文本(前200字) | 乱码 | 首token(s) | 总耗时(s) | t/s | 一级错误 | 二级错误 | note-27 ID | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q3-4b-base-p1 | Qwen3-4B | 基线 | P1 | ... | 否/是 | ... | ... | ... | ... | ... | ... | ... |
```

