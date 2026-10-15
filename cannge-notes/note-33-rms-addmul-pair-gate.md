# note-33: #20 锁定为 RMS×ADDMUL 合并子图 + 生产者门止血（2026-10-14）

接 note-32。#19/#20 收敛，#21（KV 路径）成为全默认冒烟的最后阻塞。

## #20 定案：RMS×ADDMUL 是唯一腐化配对

最小配对矩阵（全 -nkvo，big3 实际被 M 门/FA 挡在 CPU）：

| 配置 | 结果 |
|---|---|
| r1 仅 RMS | ✅ |
| m1 RMS+GET_ROWS | ✅ |
| **m2 RMS+ADDMUL** | ❌（token1 即乱） |
| m3 RMS+UNARY | ✅ |
| m4 RMS+ROPE | ✅ |

t2(R+Ad+U 无 RMS) ✅、h1(全默认) ❌、g5(无 RMS 全默认) ✅ —— 全部矩阵与此判定自洽。

## 单元探针全灭（关键负结果）

ADD→RMS、RMS→MUL([128,1]/[2560,1])、ADD→RMS→MUL 3-node、RMS 形状矩阵
72 例（2D/3D/设备/host 输入）——**全部 PASS**。模型内值级验证（embd 输入
8 chunk 与 g5 逐字节一致、Qcur_normed 六元素 rms 关系成立）也证明 GE
输入输出值正确。**腐化是模型级独有现象**（疑 alloc/流交互），单元层无法
复现——与 310 note-27 #15 "图集成问题" 的定性一致。

## 止血：RMS 生产者门（已验证）

`ggml-cannge.cpp` RMS_NORM 门：src0 为 ADD/SUB/MUL/DIV 时拒（注释含
m2 复现出处）。ADD→RMS→(权 MUL) 合并子图被拆去 CPU，独立 RMS（r1 干净）
保留在 CANNGE。验证：
- n1 (m2+门) ✅、**n2 (全默认+nkvo+门) ✅ 完美输出**、n3 (r1 复测) ✅

## 全默认冒烟现状

- 全默认 + -nkvo + 门：✅ 连贯中文（n2）
- 全默认 + KV-on-CANNGE + 门（f5）：轻度 salad → **#21 KV 路径仍腐化**
  （与 g1/g4 同型：首 token 对、随 cache 增长劣化）

## #21 已知情报（下批直接用）

- c0 探针：SET_ROWS 散射 ✓、VIEW 解析 ✓（单元层）；模型内 cache 全量读回
  显示写后稳定、各层 fletcher 各异
- KV 读路径形态：K = 全 cache 扁平 D2H（in cache_k_l0）；V = 视图经
  PERMUTE+TRANSPOSE 物化（"out cache_v_l0 (view)(permuted)(transposed)"）
- 边界拷贝全 SAME（436/436）
- 嫌疑：V 视图物化子图 / K 全量读的服务对象 / SET_ROWS 与读取的时序

## 工具增量

- DEBUG_IO out 行带 D/S(dense)+nb strides + 分块 fletcher；in 行分块 fletcher
  + 六元素值采样 —— 定位类案的标准取证套装
- 探针集：ggml_probe_{kv,getrows,gr0,rms,rmsmul,rmsmul2,addrms,mmrms}

## 待办

1. #21：c0 配置 + 分块 fletcher 定位 V/K 读路径分歧区；值级验证 V 物化输出
2. #21 修后回归 f5 全默认 → 达成冒烟判据
3. TBO 对拍（批C）；#18（-fa off）排期
4. 止血门（GET_ROWS 单行、RMS 生产者）的最终根因替换
