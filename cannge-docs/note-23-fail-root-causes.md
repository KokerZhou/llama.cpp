# note-23: TBO 45 FAIL 根因分类（本机复现 @28558340a，2026-10-08）

调查：explore agent 对 /workspace/user_data/cannge-notes/tbo-310p3-28558340a.raw.log
逐例字节级核对 + CANNGE 源码定位。前提：stdout/stderr 有缓冲竞态，仅经字节算术验证的
配对才可信；nf=1 用例逐节点切图、nf>1 整图路径（ggml-backend.cpp:2318-2339）。

## 分桶统计（45 = 47 - 2 双归属）

| 桶 | 数 | 说明 |
|---|---|---|
| staging 已知缺口 | 23 | note-14 §5；均伴 F3 门缺口（该 NS 未 NS） |
| GE 执行期 E13025（GET_ROWS 映射错，NEW） | 5 | GatherV2 批维拼接错 |
| 数值语义新 bug | 12 | SCALE bias/sinks/f16 静默零/RMS inplace |
| build 期新 bug（哨兵 OP_NONE 输出，NEW） | 4 | 整图 nf>1 必失败 |
| GE 编译错（F7，已结案） | 2 | f32 m=16 n=8 |
| F8 已知 | 1 | rms_norm_inplace |

## 新发现（台账未覆盖，建议入 note-10 为 F9-F13）

- **F9 哨兵输出**（4 例）：OP_NONE 哨兵被 analyze 注册为边界输出（plan.cpp:134），
  build 跳过 OP_NONE（graph-build.cpp:363）→ "no ES tensor for sent_N" → 整图 nf>1 全灭。
  修复一行级：analyze 跳过 OP_NONE 输出或 build 注册零拷贝端口。
- **F10 GET_ROWS**（8 例）：①batched 时 graph-build.cpp:326 axis=nd-2 批维被 splice 两次，
  GE 输出比 ggml 大 → E13025（5 例）；②vs0=1 非 dense view → staging（2 例）；
  ③f16 输入：ggml_get_rows 强制 F32 输出（ggml.c:3963-3964），GE 输出随输入 f16，
  未插 Cast（#6/#8）。索引无越界。
- **F11 SCALE bias**（3 例）：graph-build.cpp:185-187 只读了 op_params[0]=scale，
  bias 整体丢弃；inplace 例叠加 F8。
- **F12 SOFT_MAX sinks**（5 例）：sinks 挂 src[2]（ggml.c:4209-4223），L0 门不查
  src[2]（ggml-cannge.cpp:635-645）、build 裸 SoftmaxV2（graph-build.cpp:191-198）。
  在参数化 softmax 接入前应 L0 拒 src[2]≠null 转 NS 止血。
- **F13 MUL_MAT f16 静默零**（2 例）：f16 m=16 n=16 k=256 / m=64 n=32 k=80，
  编译执行无报错但输出全零（ERR=1.0）——过了门的两个 f16 二维 matmul 全灭。
  GE 静默丢写（F1 同族症状）或错写零，需设备级判别实验。

## F7 结案证据（note-20 §5.2 实验被本轮日志替代完成）

GE 编译错误链实锤：F32 输入被 GE 转 float16（Inputs dump dtype:'float16'）→
16B 不过 32B 判据 → CheckDimsAligned310P → 编译失败且 GE 清理路径空指针（EH0008 病态）
→ plan 缓存 FAILED → 输出为 HBM 垃圾。修法维持 note-20 §5.1：f32 门改 16 元素对齐。

## 修复优先级（agent 建议 + 本机意见）

1. F9 哨兵（一行级，解锁全部 nf>1 整图用例的测试能力）
2. F8 落地（台账已定修法；解锁 rms_norm_inplace，防 #30/#34）
3. F11/F12 的 L0 补拒（立即止血转 NS，数值正确性优先）
4. F7 f32 门改 16 元素（结案）
5. F10 GET_ROWS 重做（GatherV2 axis 语义 + f16→F32 Cast）
6. staging 落地（23 例；ADD/MUL perm1=1 家族 + GET_ROWS vs0）
7. F13 f16 matmul 设备判别实验

## 其他自洽性确认

- F2 已修：graph-build.cpp:225 CreateVector，4D 非 inplace RMS_NORM 全 OK
- ADD/MUL perm1=1 家族 21 个（不是 22：ADD nf=2 perm1=1 实为 F9 哨兵）
- "no ES tensor for a"（set_param 叶子，ggml.c:7252-7271）多属逐节点模式良性噪音
