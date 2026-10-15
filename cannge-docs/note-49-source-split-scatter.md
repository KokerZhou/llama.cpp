# note-49: 源码拆分 + 图内 KV 写原语普查（阴性结果存档）

## 1. 源码适度拆分（用户要求，编译/查错）

`ggml-cannge.cpp`（2126 行）→ 三个 TU + 私有头：
- `ggml-cannge.cpp`（~1300）：设备/buffer/后端接口 + graph_compute + 事件 + 注册。
- `ggml-cannge-io.cpp`（~225）：dbg 计时、`repack_2d`、`strided_copy`（bounce/drain 修复在这）。
- `ggml-cannge-gates.cpp`（~555）：`supports_op` 全套门 + `float_dtype`。
- `ggml-cannge-priv.h`：跨 TU 声明（dbg 结构体、strided_copy、shared gates）。

`graph-build.cpp`（1190 行）→ `graph-build.cpp`（903，op dispatch）+ `graph-build-io.cpp`（215：build_inputs/build_view/重排链）+ `graph-build-priv.h`（build context 结构 + 内联 get/fail）。

改门/改 IO /改 build 现在各自只重编一个 TU。行为零变化（拆分后冒烟对比一致）。

## 2. 图内 KV 写：三种 GE 原语全部静默无效（本批核心阴性结果）

目标：SET_ROWS 图内化 → 注意力全并片（前提已实测：碎片全塌缩 5.0 t/s）。逐一经单算子单图探针（`probes/ggml_probe_setrows.cpp`）验证，**三种原语全部静默 no-op**（图编译通过、数据流正确、写出值=原值）：

1. `ScatterUpdate(var, idx, updates)`：no-op（idx i32/i64 都试过）。
2. `TensorScatterUpdate(x, idx2d, updates)`：no-op（idx {1,1}、updates {1,R}、dtype 强校验已通过——报过错说明 op 真实存在，修正 dtype 后编译过但写不发生）。
3. 算术选择 `var*(1-mask) + row*mask`（relu(1-|arange-idx|) one-hot，全 elementwise）：**输出仍等于原值**——且 `GGML_CANNGE_SR_ADD1` 实验（out 强制 +1）证明**该节点的 GE 输出端口回写根本没发生**（读回的值无 +1）。

第 3 条的发现链（按"节点前打印类型/形状"方法）：
- GE dump 确认数据流正确（Permute←Slice←Add←TSU←Data）。
- 探针最初的"输出"读的是 PERMUTE 视图的 **别名 buffer**（view->data == cache->data），读到的是原始缓存字节，从未读到 GE 结果——**探针读法修正**：视图输出必须经 `ggml_cont` 拿到独立 buffer；写是否生效要直接读被写 buffer。
- E13025（输出尺寸 1MB vs 2MB）暴露 Add 的 f16+f32 类型提升。
- 即使 var 先 Cast 成算子产物再散射（排除"输入只读"理论）依然 no-op。

结论：**本栈（CANN 9.1.0 / 910B）GE scatter 族不可用**（至少经 ES 路径）。全并片的正确性前提不成立。

处置：`GGML_CANNGE_SCATTER=1` opt-in 保留全部实现（plan 标记 in_graph / build 原语 / execute 跳过 host scatter）；`GGML_CANNGE_CONT_NC` 恢复 opt-in（其正确性依赖图内写）；默认路径 = host scatter，**生产零回归**（setrows 探针 cache 读 0 误差 PASS）。

## 3. 批G 最终验证

- 探针：gemv 14/14、chain 1.5e-8、setrows-cache PASS（host 散射路径）。
- 冒烟：默认 / =v / =all 全连贯。
- 基准（-n 100 --ignore-eos，与批F 一致，拆分+回退零回归）：

| 配置 | tg |
|---|---|
| 默认 | 2.0 |
| GQA=v | 6.0 |
| GQA=all | 6.3 |

## 4. 下一步候选（全并片的前提解决后）

1. GE 散射替代原语调研：aclnnScatterNd/aclnn 直通（绕过 ES）；或 GE 版本升级验证；或 InplaceAdd 类。
2. 或接受 ~6 t/s 架构，转向 prefill FA 上 GE（note-42/43 探针在，prefill 2.5→15+）。
3. 层间合并的余量：现 ~110 片/token 中 swiglu 已收编，剩余 CPU 段只有 kq/v 边界与 lm head。
