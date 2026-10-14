# note-30: 910B3 首调轮——冒烟打通与乱码二分树（2026-10-14）

接 note-27。本机（910B3 / CANN 9.1.0 / driver 25.5.0）第一轮。HEAD=d3a3ef81f。

## 环境基线（本机首记）

- 910B3 单卡健康；CANN toolkit 9.1.0 用户级 `$HOME/Ascend/ascend-toolkit`
  （innerversion V100R001C11SPC001B243）；无 `/usr/local/Ascend/cann-*`
- llama-cli 构建须 `-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF`（note-27），
  只构建 `ggml-cannge/test-backend-ops` 不触发 UI 下载
- 模型：910 机原本无 Qwen3-4B，从 hf-mirror（github/huggingface 不通，modelscope 通）
  下载 `unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-F16.gguf`
  （8051285344 B = 7.5 GiB，与 note-29 同款同尺寸）→ `~/models/`
- 冒烟脚本 `scripts/03_smoke_q3_4b.sh`：note-29 P1 基线口径 + stdout 截断防 `"> "` 洪泛

## TBO 窄集 910B3 基线（script 02 口径，9 op）

238/252（二进制自报），14 FAIL = perm1=1 家族 9（ADD/MUL f16+f32 的 [10,5,1/4/4x3]
+ MUL [1,1,320,320]）+ GET_ROWS batched 5（4xf32 + 1xf16 be1=7）。
对比 310 note-27（324/333）：perm1=1 310 残 4 → 910 残 9（[10,5,4,1]/[10,5,4,3]
在 310 已被 staging 修好、910 复发，疑 SoC 差异）；GET_ROWS batched +1。
f32 matmul 门 = 8 元素（`ggml-cannge.cpp:1099` #else 分支），与 310P 16 不同属预期。
SET_ROWS 全量 44/44（含模型实际用例 f32→f16 i64）。

## 冒烟时间线与失败链 #18

| 跑 | 配置 | 结果 |
|---|---|---|
| a1 | note-29 模板 `-fa off` | **abort #18**: `ggml-backend.cpp:941 cache_v_l0 (view) cannot run VIEW` |
| a2 | 默认 FA（auto→on） | 全程通（2.8 min，0.6 t/s）但**乱码**（.jams..随蚤...） |
| b1 | NO_RMS | 乱码（首 token 部分改善） |
| b2 | DISABLE_ALL | abort：KV 预分配 SET_ROWS 门拒（预期内） |
| b2nkvo | DISABLE_ALL + -nkvo | **输出完全正常**（4.1 t/s 纯 CPU） |
| b3 | 默认 + -nkvo | 乱码（首 token 即错）→ KV 位置不是主因 |
| b4 | NO_MATMUL | 乱码 |
| b5 | NO_RMS+NO_MATMUL+NO_SOFTMAX | 乱码（首 token 对） |
| c0 | 全部计算 op 关（7 旋钮）KV 留 CANNGE | 轻度 word salad → KV 路径独立轻度破坏 |
| b6 | b5 + -nkvo | 乱码 → 破坏者 ∈ {ROPE,ADD/MUL,UNARY,GET_ROWS,boundary} |
| e1-e4 | 单家族正交（见下） | 见下 |

**#18 根因**：`-fa off` → `v_trans = !flash_attn = true`（llama-model.cpp:2569 系）
→ V cache 转置布局，get_v 的 4D 视图非连续 → VIEW 门 `ggml_is_contiguous`
（ggml-cannge.cpp:1130，c9bfff639 引入）拒绝 → 预分配 abort。310 从未跑过
`-fa off`（note-29 是计划稿）→ 非 SoC 差异，是 d3a3ef81f 的通用缺口。
修 = 支持转置 V 视图（GE Transpose 链）或 VIEW 门放行走 Slice 路径，未开工。

**关键正向事实**：默认 FA 下 910B3 全程无病态编译（note-25 #8 未复现），
KV 图 GE 编译+execute 全通；乱码=数值问题，非编译/调度问题。

## 新工具（本批落地）

- 二分旋钮 +4：`GGML_CANNGE_NO_ROPE / NO_ADDMUL / NO_UNARY / NO_GET_ROWS`
  （ggml-cannge.cpp supports_op，风格同现有 knob）
- DEBUG_IO 修复：cpy 调试块 dst 为设备指针时 memcpy 读设备内存 → segfault；
  按 `ggml_backend_buffer_is_host(dst->buffer)` 选 D2H（ggml-cannge.cpp:299 块）
- KV 探针 `probes/ggml_probe_kv.cpp`：预分配 cache + SET_ROWS(f32→f16,i64)
  + VIEW[128,8,256] 复刻冒烟链路。**结论：SET_ROWS 散射 ✓、VIEW 解析 ✓**
  （两后端均 PASS，max_abs f16 精度内）；permute+transpose 段探针自身有问题
  （CPU 侧同样 FAIL，harness bug，vt 段结论不可信，待修）
- GE 退出挂死：探针 main return 时 `pthread_cond_destroy` futex 死等
  （libge_runner InnerSession 析构）→ 探针用 `_exit` 跳过 atexit

## 证据要点（c0+DEBUG_IO/compile）

- 全部 cpy 行 `SAME`（4KB fletcher 一致），无 DIFF
- cache_k_l0 全量读回：prompt 前 0 → prompt 后 c03ea262 且逐 step 稳定 →
  写发生且无覆盖；各层 fletcher 各不相同 ✓
- CANNGE 子图（6 节点）= SET_ROWS(k) + SET_ROWS(v) + VIEW[128,8,256] +
  PERMUTE(0,2,1,3) + TRANSPOSE → 输出喂 CPU attention；K 走全量扁平 D2H
- V 视图经 "slice path" 构建（base [1024,512] → Slice+Reshape），
  off=0, node_ne=128,8,256,1，仅 3 次编译（n_kv 固定 256，非逐 step 变）
- 真实 KV 布局：cache=[1024,512]（stream 维在 base 之外），
  视图步长 [2B, 256B, 2048B]（ggml_view_4d 签名 = nb1..nb3+offset，勿传 4 个 stride）

## 当前判断

- 910 上至少两类独立破坏者：① KV-on-CANNGE 轻度（c0）；② 计算 op 家族重度
  （b3/b6，首 token 即错）。与 310 #15 "RMS 主破坏者"不完全重合
  （b1 NO_RMS 仍乱码；DISABLE_ALL+nkv0 在 910 干净，310 当时不干净——
  310 结论可能受 ref 损坏污染，note-27 #17 教训）
- staging/边界拷贝嫌疑上升：所有跨边界张量机械检查（布局/校验和）均一致，
  但数值无人对拍过（本机无 .kld 工具链，310 的 KL harness 未移交）

## 关键突破：组合矩阵（e/p/t 系列，全部 -nkvo，三大 op 落 CPU）

| 配置 | CANNGE 上的家族 | 结果 |
|---|---|---|
| e1 | 仅 GET_ROWS | ✅ 完美 |
| e2 | 仅 ROPE | ✅ 完美 |
| e3 | 仅 ADD/MUL | ✅ 完美 |
| e4 | 仅 UNARY | ✅ 完美 |
| p1 | ADDMUL+UNARY | ✅ 完美 |
| p2 | ROPE+ADDMUL | ✅ 完美 |
| p3 | GET_ROWS+ROPE | ✅ 完美 |
| t1 | GET_ROWS+ROPE+ADDMUL | ❌ 乱码 |
| t2 | ROPE+ADDMUL+UNARY | ✅ 完美 |
| t3 | GET_ROWS+ADDMUL+UNARY | ❌ 乱码 |

**单家族/两家族全干净，三家族起乱码；含 GET_ROWS+ADDMUL 的组全灭**。
调度器并非按旋钮分家族——t2 里 ROPE/UNARY 实际全落 CPU（普查无对应子图），
CANNGE 只剩单点元算子。**判别特征 = 嵌入区合并子图**：
`2xGET_ROWS[2560,n] + ADD → ffn_inp-0/35`（node_1433/1434 无名 +
ffn_inp-35 ADD，t1 独有，t2/p3/e1 无）。GET_ROWS 在时只有当 ADDMUL 也
在才会与 ADD 合并成该子图 → 与矩阵完全吻合。

## 数值取证（DEBUG_IO out 值转储，t1 -n4）

- 残差流幅值逐层增长（ffn_inp-34=191.9 → l_out-34=253 → ffn_inp-35=294.9），
  result_norm 输出 (-0.113, 0.251, -0.704) 量纲正常——乱码非一眼数值爆炸
- 该子图输入 = node_32/embd 的 H2D 拷贝（SAME）+ `leaf_473 [1,1,1,1] sum=0`
  （gather 的 idx 叶，值=0）；"embd" 本身作为外部输入进 ADD → 两个 GET_ROWS
  输出在子图内无消费者却不是 GE 输出端口（out 只记 ffn_inp-0）→
  **疑似死节点/非常规拓扑，疑似其设备缓冲与活张量别名冲突**（待证）
- 全部 cpy/in/out fletcher 一致（SAME），无 D2H 差异

## 探针结论（probes/ggml_probe_getrows.cpp）

GET_ROWS×2+ADD 组合：直接绑定/staged 输入/plan-cache 重绑定（3 轮轮换 idx）
全 PASS、逐位一致 → 子图本体无辜，**嫌疑锁定该子图在调度/分配层的
输入输出绑定或缓冲区别名**。

## 待办（下批优先级）

1. 导出真实 decode 图结构（ggml_graph_dump 或 analyze 全节点日志），
   定位 node_1433/1434 的消费者与绑定方式，证实/排除别名假说
2. 若证实：修 analyze 的 output_tensors/别名处理（死端口不绑定或
   绑定独立 staging）
3. 修 #18（-fa off 转置 V 视图）数值正确后排期
4. perm1=1 910 复发 + GET_ROWS batched TBO 对拍
5. GE 退出挂死仅影响探针（_exit 绕过），llama-cli 未见
