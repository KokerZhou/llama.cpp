# note-48: GQA 模型级修复（g-major 映射 + 回写竞态）与 =v 3.8 t/s

## 1. 根因一：GQA 头映射探针自证循环（批E-1/2/3）

旧结论"探针精确、模型级失败、差异在 cache 视图 staging"**全部错误**。真相：

- 探针 `want` 参考用了 `g = h % G`（r-major），与 build 的 r-major reshape 自洽——cannge 探针"PASS"只证明 cannge 算了自己参考的东西。
- **ggml-cpu 真值 = `g = h / R`**（ggml-cpu.c:1221 `i02 = i12 / r2`）。探针 CPU 端执行一直 FAIL（bad=2065/4096），是 fitter 没跑在 cpu 侧而被忽视。
- 修复 = build 纯重标改为 g-major：`e0 {G,1,N,K} x e1 {G,R,1,K}`（1 维可自由移动，无 Permute），输出 `{G,R,N}`→`{H,1,N}`。探针参考同步改 `g = h/(H/G)`，双端 fitter `h=g*R+r bad=0` 实证。
- 探针 14/14 PASS（dense + permuted cache-view 双 case）。冒烟：GQA=kq 连贯 1.1、GQA=v 连贯 3.1。

教训：**参考实现必须与 ggml-cpu 对拍，且 fitter 要跑双端**；自洽的探针参考能"证明"错误代码正确。

## 2. 根因二：回写 bounce 竞态（=all 乱码的直接原因）

症状：=kq/=v 各自单开连贯，=all 全乱且 temp 0 输出不确定（"!"/"你好"随机）；加 DEBUG_IO（内含 `aclrtSynchronizeStream`，line 1224）后变正确——竞态被同步掩盖。

定位链：
1. 转置回写对 cache 是**恒等写入**（T 的 strided buffer 别名 cache，scatter 后字节恰好原位；`out S` 校验和 = `in` 校验和实证）——排除"缓存污染"假说。
2. 注意力链探针（kq→softmax→v 三输出）：kqv 输出 1.5e-8 精确（probe 注意：ggml_build_forward_expand 不设 OUTPUT flag，`ggml_set_output` 才设；无消费者又无 flag 的张量无 GE 端口，读到的是陈旧 buffer——探针的 kq/kq_s "FAIL" 是探针假象）。
3. 同步后的 `out S` 出现跨层重影（l2 = l1 的校验和）→ bounce 路径 `aclrtMemcpy`（host 阻塞，**跑默认流不等 ctx->stream 上挂起的 GE execute**）读到了上一层残留的共享 staging。

修复（ggml-cannge.cpp strided_copy bounce 分支）：device 侧源（dense_to_strided 的 staging、input 的 device view）先 `aclrtSynchronizeStream(stream)` 再走阻塞拷贝。

## 3. 顺手修复：转置回写跳过（plan.cpp）

"纯重排视图（PERMUTE/TRANSPOSE/RESHAPE/VIEW 链）的边界输出 + base 是 plan 输入"→ 不注册 GE 端口（与 dense-view 同理，bytes 已正确）。消除 36 层/步的 host 往返 + drain。io_views 本就只登记不被消费。

## 4. CONT_NC 实验（重要阴性结果）

`GGML_CANNGE_CONT_NC=1` 让 v_cont 上 GE：=all 飙到 **5.0 t/s** 但全乱。根因：**SET_ROWS 是 post-execute host scatter**，同 fragment 内 GE 视图链读到 scatter 前的缓存（缺当前行）；CPU 片段（#1/#2/#3 结构）恰好把"写后读"排到正确的序。要让注意力并成一片，必须先把 SET_ROWS 改成图内 ScatterUpdate（JittorInfer 同款，note-39）——下一步主线。

## 5. 当前数字（-n 100 --ignore-eos，F16 Qwen3-4B-2507，910B3）

| 配置 | tg | 备注 |
|---|---|---|
| 默认（无 GQA） | 1.7 | 连贯（def-regress 冒烟过） |
| GQA=v | **3.8** | 连贯，v 收编、kq 留 CPU；当前推荐配置 |
| GQA=v + SCHED_EVENTS | 3.4 | 事件开销>收益，维持关 |
| GQA=all（修后） | 1.8 | 连贯；K 视图 bounce 是大头 |
| GQA=all + CONT_NC | 5.0 | 乱码（缺 ScatterUpdate） |
| 原始 CANN graph 参照 | 71.7 | note-47 基线 |

=v 成为当前最优：v 收编无 bounce（v_cont 是 host 张量 H2D 便宜），kq 在 CPU 由 sched 同步 D2H。all 的 K 视图 staging（nb0=2 不合 4B 对齐）走 host 往返 + drain，36 次/步，吃掉收益。

## 6. 下一步（按预期收益）

1. **K 侧 in-GE 重排**（~all 1.8→4+）：gemv build 检测到 src0=纯重排视图且 base 为 dense device 输入时，Data(base) + 图内 Reshape/Permute 链替代 staging bounce。
2. **SET_ROWS → 图内 ScatterUpdate**：解锁全注意力一片（5.0 潜力已实测）+ 消掉 fragment #1/#2/#3 结构。
3. prefill FA 在 CPU（2.5 t/s）→ GE FlashAttention（note-42/43 有探针）。


---

# 批F补充（同 note-48 文件续记）

## F1. 双输入 SwiGLU 上 GE

模型 ffn 是 split 形式（`ggml_swiglu_split(gate, up)`，两个 mul_mat 输出喂 GLU 两个 src）。门放开：src1 同形同 dtype SWIGLU + swapped=0；build = `Mul(Swish(gate,1.0), up)`。每层 ffn_swiglu CPU 片消失。**=v 3.8→5.9 t/s**（-n 100）。

## F2. 重排视图输入替换（消 K/Q bounce）

plan.cpp：非 dense 纯重排视图（且 offset=0、视图元素数 ≤ base 且按 base 内维对齐）→ 注册 BASE 为输入。graph-build.cpp `get()` 回退：holder=base → Slice(GE dim0 线性前缀) → 逐级 relabel（PERMUTE/TRANSPOSE 先 Reshape 到 src0 形状再换序）。**=all 1.8→4.7 t/s**。

**坑实录（组合错误，单算子探针抓不到）**：ggml 折叠 view 链（permute->view_src 直指 cache），折叠视图可能只覆盖 base 前缀（KV=256 of 512）→ 首版构建把 rank3 的 permute order 打在 rank2 holder 上，GE E89999（`perm value[2] >= input rank[2]`）。按"节点前打印类型/形状"方法（DBG-RELABEL，DEBUG_BUILD=1）一步定位。

## F3. 批F后基准（-n 100 --ignore-eos，prompt≈10 token）

| 配置 | tg | 连贯 |
|---|---|---|
| 默认 | 2.0 | 冒烟过 |
| GQA=v | 6.0 | 冒烟过 |
| GQA=all | **6.3** | 冒烟过（K bounce 已消，=all 反超 =v） |

## F4. 与原始 CANN（71.7 tg）的剩余差距分解（诚实账目）

每 token 的墙钟 ≈ 片段数 × (GE 下发+执行+同步) + 边界拷贝 + CPU 算子：
- =v：~110 片/token（36 kq CPU + 36 v 片 + 36 ffn 片 + 边界）→ 170-190ms。kq 在 CPU（36 个 GQA gemv）是硬下限之一。
- =all：K bounce 已消，但 v_cont 仍 CPU（#1/#2/#3 结构由 SET_ROWS 语义强制）。
- 端到端要接近 71.7（13.9ms/token）必须：整层一片（kq+v+ffn 无 CPU 段）+ 层间合并 + KV scatter 图内化，把每 token 下发压到个位数——即 JittorInfer 架构（note-39 三前提的最后一块：O(1) 次同步）。
