# note-41: 共享集成层根因实锤——SET_ROWS/SET 散射读 GE 内部缓冲（2026-10-14）

用户指令"必须排查"的答卷。批 B0 收案：一类根因实锤+修复，#20 判为另一类。

## 根因（实锤链）

1. rope-on + KV-on 乱码 / rope-on + **-nkvo 干净** → 腐化限定在 K 路径：
   rope(K) 输出 → SET_ROWS 散射 → cache → CPU FA 读取
2. **机制**：rope 与 SET_ROWS 落入同一 CANNGE 子图时，GE 把 rope 结果留在
   **GE 自己的工作区**（图内中间张量从不写到 ggml 地址）；而 SET_ROWS
   散射是 execute 后的 host 侧循环，读的是 `sr.data->data`（ggml 缓冲区）
   → 复制陈旧字节进 cache。孤立探针全过是因为探针的 data 是 host 张量
   （data_host=true 走真实宿主拷贝）——从未覆盖"图内生产 + 设备散射"
3. **修复**（plan.cpp analyze）：SET_ROWS/SET 的 data/src 由图内生产时，
   强制加入 io.outputs → GE 输出端口把它物化到 t->data → 散射读到新鲜值
   （同 CPY/DUP 端口机制）
4. 附带修复：散射 stage_row D2H 前的默认流竞态（补 ctx->stream drain，
   F15b 同类；idx 路径本来就有、数据路径漏了）

## 验证

- rope-on + KV-on 全默认：**连贯中文** ✓（此前必乱）
- rope-on + nkvo：连贯 ✓（tg 2.6）
- m2 复现配置：干净 ✓；RMS 门移除复测：**仍乱码 → #20 是另一类 bug**，
  门恢复保留（注释更新）。#20=RMS×ADDMUL 合并腐化，独立于 SET_ROWS 类
- TBO 窄集+ROPE：262/275，ROPE 13/13 全过，无新增 FAIL
  （残余：GET_ROWS batched 5、perm1=1 退化维 5，均有 310 起记录的去向）
- 基准（Qwen3-4B F16）：全默认 4.4/0.9；-nkvo 31.6/2.3

## 本批交付（工作树，未 commit）

- SET_ROWS/SET 强制输出口（plan.cpp，根治 shared-layer 一类）
- 散射默认流竞态修复（ggml-cannge.cpp）
- ROPE 门放开（beta 死参数甄别，Qwen3-2507 YaRN 参数不再误杀）
- ROPE 构建死代码修复（dup_pairs reshape 被丢弃 → TBO ROPE 13/13）
- DBG-GATE 三返回点诊断、cpy 分块 fletcher、ggml_probe_rope
- RMS 生产者门保留（#20 仍未解，独立于本类）

## 模式更正（note-40 的升级）

"单元过/模型坏"三次复现不全是同一机制：rope=#20 曾误判同类。实际是
**至少两类**：(a) 散射/副作用读 GE 内部缓冲【已修】；(b) RMS×ADDMUL
合并腐化【未解，门挡】。下轮攻 (b)：用同款"输入输出同图值转储"手法对
m2 的 {ADD,RMS,MUL} 子图逐值验证，定位 ADD 残差流在合并图的哪一环漂移。
