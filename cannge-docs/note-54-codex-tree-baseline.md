# note-54: Codex 重构树基线 + 7427 op 分解 + no-op cast 跳过（c5b849a30）

## 背景

用户指示基于 codex 代码继续开发。`cannge` 分支 tip 已由 Codex 重构推进到
`9224022f5`（ES 封装细分为 common/context/cpy/elementwise/fattn/ge/graph-build{,-io,-priv,-view}/
mul-mat/norm/ops/options/plan/plan-key/rope/supports/tensor/view + tests/，架构见 tests/HARDWARE.md）。
本会话此前全部关键优化在重构中存活：f16 gemv 产物、GQA g-major、lm head 分片、双输入 GLU、
视图替换、TSU 直连、is_weight 修复、plan 签名修复。env 变量名未变。

## 新树性能基线（910B，Q3_K_S 4B，-n 100，tg tok/s）

| 配置 | tg | 稳态后端 ms/token | 备注 |
|---|---|---|---|
| 默认（无 env） | 2.4 | — | 小图/逐 op |
| `GGML_CANNGE_GQA=all`（分裂图） | **9.4** | ~33 | bind 已被 Codex 修复；连贯 |
| +`GGML_CANNGE_CONT_NC=1`（合并图） | 2.6 | ~141 | 连贯但慢，见结构墙 |

-n 100 数字含一次性 GE 编译摊销（新树 ~23s）；稳态数字为 DBG-TIME 剔除编译后的实测。
目标参考：原始 CANN（aclrt 捕获图）71.7 tg。

## 合并图 7427 op blocks 分解

| 类别 | 数量 | 成本性质 |
|---|---|---|
| Const | 2335 | 编译期折叠，运行时免费 |
| Reshape | 1332 | 每层 ~37 个 relabel——**下一个待验证杠杆**（若 GE 不折叠则是最大单片开销） |
| Cast | 1107 → **786** | 本次砍掉 321 个 no-op（下节） |
| 计算（MatMul/RmsNorm/RoPE/GLU/Attn…） | ~2000 | 真实工作 |

结构墙结论（note-53 已立，本批数据继续吻合）：GE 逐 op 地板 ~48µs vs aclnn eager ~10µs；
op 数 5 倍 × 单 op 成本 5 倍 ≈ 25 倍差距，与观测（141ms vs 33ms、以及两者对 71.7 的倍数）吻合。
到 71.7 需 GE 每 op ≤6µs 或层内融合核——平台能力问题，非封装 bug。

## no-op cast 跳过（c5b849a30）

- 现象：mul-mat 构建在每个 gemv 周围无条件发 `Cast(→f32)`；f16 gemv 产物路径下
  输入已是 f32 或 Mul 已隐式对齐，`f16→f16`/`f32→f32` 的 no-op cast GE 不折叠，
  在合并图上每个都是真实 kernel。
- 改动：`cast_if_needed` 帮助函数——dtype 相同则直通；替换 3 处调用点
  （M==1 gemv、batched GQA gemv、fresh-row 拼接 cur_prod）。+16/-3 行，单文件。
- 效果：Cast 1107→786（-321，-29%）；分裂配置 tg 9.4 不变
  （砍掉的 cast 挂在合并路径上，分裂路径本来无此开销）；双配置冒烟连贯，探针通过。
- 提交纪律：工作树中另有一批**来历不明的 1871 行大改**（graph.cpp/plan.cpp/tensor.cpp/tests/*，
  疑似其他 Agent 半成品重写），本次仅提交 mul-mat.cpp（先 stash 验证独立编译通过后提交再 pop）。

## 下一步（按预期收益排序）

1. **Reshape 1332 验证**：每层 ~37 个 relabel 是否被 GE 折叠。验证法：临时删掉 gemv
   可省 reshape 或合并连续 reshape，数 op 变化 + 测 tg。若不动则合并路径在结构墙内
   上限 ~12 t/s，应转向向用户汇报平台能力边界。
2. 启动探针 M 对齐：省 30-50s 启动编译（一次性成本，优先级低）。
3. 分裂路径残余调度边界税：CPU v_cont 36 段。

## 方法论复读

子 Agent 最小提示词独立诊断仍是命中根因的最短路径；GE 无罪结论继续成立——
本批 cast 浪费也是我侧 ES 封装多发的 op，不是平台缺陷。
