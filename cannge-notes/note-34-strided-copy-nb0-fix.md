# note-34: #21 根因定位修复——strided_copy bounce 路径 nb0≠es 崩塌（2026-10-14）

接 note-33。**全默认冒烟（-ngl all、KV on CANNGE、全 op）已输出连贯中文（f6）**。

## #21 根因：strided_copy 的 es<4 bounce 路径按行 memcpy，无视 nb[0]

V cache 视图链 `VIEW[128,8,256] → PERMUTE → TRANSPOSE` 的 GE 物化输出张量为
**strided**：`out S ne=[256,128,8] nb=[2048,2,256,1048576]`（逻辑 dim0=t 的
步长 2048B ≠ es=2，转置视图典型形态）。物化结果经 staged_ports 的
strided copy-back（dense→strided）写回。

`ggml_cannge_strided_copy` 的 **bounce 分支**（es<4 或 4B 未对齐时启，
注释指 f16 转置视图 2B 步长）对每个 (i1,i2,i3) 行做 `memcpy(dst, src,
ne[0]*es)` —— **假设行内 ne[0] 个元素连续（nb[0]==es）**。当 nb[0]>es
时元素实际按 nb[0] 步长分布：连续 memcpy 把整行堆到行首、行内其余位置
及后续行全部踩乱/留旧值 → GE 输出 "稀疏写、大部分为零"（c0v.log 分块
fletcher 实证：decode 步 +4k..+24k 全 0），CPU attention 拿到坏 V →
随 n_past 增长的渐进 salad（c0/g1/g4/f5 同型）。

对齐分支（aclrtMemcpy2dAsync，spitch/dpitch 取 nb[0]）本来就对——
只有 bounce 三处（out-scatter/in-host-gather/in-mirror-gather）有洞。

## 修复

bounce 三分支统一加 `nb[0]==es` 判断，不等时逐元素 scatter/gather
（正确优先；注释一行指认转置 V 视图形态）。对齐分支不动。

## 验证

- **f6 全默认（无 knob、无 -nkvo）：完美连贯中文** —— 三大破坏者
  (#19 单行表 GET_ROWS 门、#20 RMS 生产者门、#21 strided_copy)全部
  收敛后，KV-on-CANNGE 端到端通过
- n2（全默认+nkvo）此前已干净；f5（修前）轻度 salad → 修复生效

## 冒烟判据达成状态

- ✅ Qwen3-4B F16 910B3 稳定连贯输出（f6，40 token 完整句）
- ✅ 根因链记录 note-30~34（#18 VIEW 门缺口、#19、#20、#21 全定位）
- ⏳ TBO 窄集 FAIL 收敛（批C）：当前 14（perm1=1×9、GET_ROWS batched×5），
  目标 ≤4/≤4 或记录去向——GET_ROWS 单行门会改变 TBO 口径，待重测

## 待办

1. 批C：TBO 重测 + 对拍（f32 门 8 元素口径差异说明、perm1=1 910 家族、
   GET_ROWS batched 记录）
2. #18（-fa off 转置 V）仍 open（VIEW 门），数值正确后已可排期
3. 止血门的最终根因替换（GET_ROWS 单行门待 GE GatherV2 单行修正；
   RMS 生产者门待 alloc/流交互根因）
4. 回归验证一轮（f6 复跑确定性 + -n 更大窗口）
