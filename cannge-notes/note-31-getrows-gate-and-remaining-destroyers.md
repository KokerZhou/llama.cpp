# note-31: GET_ROWS 单行表门修复 + 残余破坏者定位（2026-10-14）

接 note-30。根因 #19 定位+修复，残余 #20(RMS-in-graph) #21(KV 路径)。

## #19 根因：单行表 GET_ROWS 在 GE 只写 1 元素（已修）

- **复现探针** `probes/ggml_probe_gr0.cpp`：get_rows([2560,1] 单行表, idx=[1]{0}) ×2 + ADD
  → CANNGE 输出仅元素 0 正确、其余全 0（bad=2536/2560）；CPU 正确。
- **机制**：GE GatherV2 对单行表推断输出塌陷（等效 [1,1]），ADD 跟随输入形状
  只写 1 个元素。大表（[2560,151936] embedding）不受影响（TBO/冒烟均验证）。
- **模型触发点**：调度器分裂出的 `2xGET_ROWS(node_1432/l_out-34, leaf_473[1]={0}) + ADD → ffn_inp-35`
  子图（DEBUG_BUILD 全节点转储证实；leaf_473=0 常数 idx 是该结构的指纹）。
  家族矩阵之谜由此解开：GET_ROWS 受支持 + ADDMUL 受支持 → 该子图成形 → 乱码；
  任一不在 → 子图不成立 → 干净。与 op 家族数值质量无关。
- **修复**：`ggml-cannge.cpp` GET_ROWS 门拒 `src0->ne[1] == 1`（单行表），
  注释含探针出处。embedding 路径不受影响。
- **验证**：t1/b6（原乱码）修复后输出完美；全默认 f3 仍乱码（见下）。

## 门后全默认单旋钮（f3/g 系列，KV 均在 CANNGE）

| 跑 | 配置 | 结果 |
|---|---|---|
| f3 | 全默认 | ❌ 乱码 |
| g1 | NO_RMS | 部分改善仍乱码（首 token 对后劣化） |
| g2 | NO_MATMUL | ❌ |
| g3 | NO_SOFTMAX | ❌ |
| g4 | NO_RMS+NO_MATMUL | ❌（较轻） |
| g5 | NO_RMS + -nkvo | ✅ **完美** |
| h1 | RMS 开 + -nkvo | ❌ token1 即乱 |

## 残余破坏者判定（控制变量）

- **MUL_MAT / SOFT_MAX / ROPE / ADD/MUL / UNARY**：g5 全开在 CANNGE = 完美 → 全部无辜
- **#20 RMS-in-graph**：h1（仅 RMS 开+KV 关）token1 即乱 → 实锤为 910 上的
  主要破坏者，与 310 note-27 #15 结论一致。孤立 RMS 全形状探针
  （`ggml_probe_rms.cpp`，[2560,1]/[2560,15]/[2560,256]）全 PASS
  → 又是集成层问题（多 op 子图/绑定/别名），非 kernel
  - 注意：g1 显示 RMS-in-graph 的劣化形态是"首 token 对、随后劣化"——
    与 h1 的 token1 即乱不同 → g1 的残余主要是 KV 路径，RMS 的破坏
    在 h1 无 KV 时更纯粹
- **#21 KV-on-CANNGE 路径**：g1/g4（RMS 关+KV 开）仍乱、g5（KV 关）干净
  → KV 路径独立破坏（轻度，随 cache 增长劣化）。c0 探针已证 SET_ROWS/VIEW
  单元正确 → 也是集成问题（D2H 读路径/staging 待查）

## 工具增量

- `GGML_CANNGE_DEBUG_BUILD` 扩展：analyze 全节点转储（name/op/ne/srcs）——
  定位无名节点来源的利器
- 探针：ggml_probe_gr0（#19 复现）、ggml_probe_rms（RMS 阶梯）、
  ggml_probe_mmrms（mm→rms 链；初版权重超 256K 门被拒+参考实现有 bug，
  下批改小 K/N 并重写参考）

## 下批计划

1. #20：小权重 mm→rms 链探针（K,N ≤ 512 过门）复现 RMS-in-graph；
   若单元不复现 → 模型级 DEBUG_IO 值转储对比 RMS 子图输出与 CPU 语义
2. #21：KV 读路径（get_tensor D2H 全量读 vs 视图物化）模型级值取证
3. 两修齐后回归 f3 全默认 → 达成目标冒烟线
4. TBO 对拍（批C）与 #18（-fa off）顺延
