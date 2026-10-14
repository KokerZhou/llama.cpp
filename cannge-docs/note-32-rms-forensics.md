# note-32: #20 深度取证——值级验证与方法论修正（2026-10-14）

接 note-31。#20 调查过程中的关键修正与排除。

## 重大方法论教训：strided 张量上裸 fletcher 无效

DEBUG_IO 的 fletcher 是从 t->data 起连续读 ggml_nbytes 字节。对**非连续
视图**，该跨度包含无关张量的数据 → 跨配置 fletcher "差异"是布局假象。
**只有 DENSE 张量的 fletcher 可比**。 e/i 值转储（前若干元素）才是
跨配置可比口径。此前"h1 vs g5 fletcher 差异"的部分结论需按此重审。

## #20 值级排除（h1：全默认+nkvo，token1 即乱）

- **Qcur_normed-0（RMS+MUL，CANNGE）值级验证通过**：输入/权值/输出同图
  六元素采样（含 head0/head1 边界 idx 127/128），两 head 各自反推的
  rms 完全一致（h0: 0.2025、h1: 0.2355），y = x/rms*w 逐项成立
- RMS 形状矩阵探针 72 例全过：2D（[2560,x]、[128,x]…）+ 3D
  （[128,32,2/15/1]、[128,8,2/15/1]，即模型 q/k per-head 形状）×
  设备/host 输入两变体 → **RMS kernel 与 staging 路径全部无辜**
- RMS+[128,1] 权值广播链探针（ggml_probe_rmsmul）也过
- attn_norm-0（[2560,15] dense fletcher）h1==g5 → 残差流入口一致
- **真实分歧点**：ffn_norm-0 / l_out-0 / result_norm（均 dense，
  fletcher 有效）在 prompt 阶段即分歧 → 腐化进入 layer-0 attention
  内部（rope/FA/o_proj 均 CPU + CANNGE 残差 ADD），而 Q/K normed
  已验证正确 → 嫌疑收敛到 **V 路径 / FA 输入装配 / 残差 ADD 的
  跨子图输入拷贝**
- 竞态假说（input_staging H2D 无同步）**证伪**：补
  `aclrtSynchronizeStream`（ggml-cannge.cpp staged_any 块）后 h1 仍乱。
  同步保留作防御（注释含 F15b 出处），无害

## 工具增量

- DEBUG_IO in/out 值转储六元素采样（0,1,2,127,128,255，跨 head 边界）
- 探针：ggml_probe_rms（形状矩阵 72 例）、ggml_probe_rmsmul、
  ggml_probe_mmrms（初版权重超 256K 门+参考 bug，待重做）

## 下批切片

1. h1 vs t1（RMS off）DEBUG_COMPILE 子图普查对比：RMS-on 拉进来的
   新增/变形子图 = 腐化载体候选；对新增子图输出做值级验证
2. V 路径取证：v_cur 从 CPU matmul 到 FA 的完整链路（为何过 CANNGE
   边界、D2H 是否值正确）——V 无 norm 未被任何探针覆盖
3. 若 V 路径无辜 → 残差 ADD 子图的输入拷贝（l_out 链）逐值对拍
