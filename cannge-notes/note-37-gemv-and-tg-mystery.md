# note-37: 性能优化第一轮——GEMV 上线 + 计时桩定位 tg 之谜（2026-10-14）

接 note-36。目标：CANNGE decode 0.9 t/s → 逼近 CPU(4.6)/原生(72)。

## 已交付：M=1 GEMV（decode matmul 上设备）

- **门**：MUL_MAT 增加 gemv 分支（src0 2D + src1 仅一列 + 连续）——跳过
  BatchMatMulV3 的 32B 对齐与 256K 权重上限（ggml-cannge.cpp）
- **构建**：`y[n]=sum_k w[k,n]x[k]` 映射 `Cast(f32)×Cast(f32)→Mul(广播)→
  ReduceSum(K轴,keepdims)`；f32 全链对齐 ggml-cpu 的 f32 累加精度
  （f16 乘积+求和会掉尾数），HBM 带宽 ~80MB/matmul ≈ 可忽略（graph-build.cpp）
- **验证**：ggml_probe_gemv 4 形状 max_rel 3e-6；TBO MUL_MAT 25/25
  （M=1 用例从 NS 转 OK）；冒烟 g1/g2 连贯中文
- **效果**：pp 3.1→5.8 t/s（+87%）；**tg 0.9→1.0 基本未动** ← 引出下文

## tg 之谜的排查链（全部有实证）

1. 每子图 GE 执行 0.04ms（ggml_probe_perf 干净进程 200 迭代）——
   **"切图数×固定开销"理论崩塌**（100+ 段×0.04ms 仅 4ms）
2. OMP 自旋争用假说：passive/4 线程消融无效 → 排除
3. 驱逐抖动假说：evicts=0（且 MAX_PLANS 64→256 防御）→ 排除
4. cgraph 指针→plan 旁路缓存：指针不复用全 miss 且插入本身有开销
   （tg 反降至 0.5）→ 已回退
5. **GGML_CANNGE_DEBUG_TIMING 计时桩**（本批新增，常驻）：32753 次调用
   中 bind=45s(stage 31s+sync 5s) / find=19s / exec 仅 1.8s
6. 微基准同型 map find 0.18µs——**plans.find 实测 0.96ms/call 与代码内容
   不符**；进程整体 0% CPU、voluntary 切换 ~10/s、非自愿 0 →
   计算线程大部分时间在睡
7. 干净进程探针同代码 0.04ms/call vs llama-cli 内 2.27ms/call（56×）

**当前判定**：时间消失在 graph_compute 内部的窄窗口里，且与微基准矛盾
+ 进程睡眠为主 → 指向 **ACL 驱动互斥/设备侧序列化**（所有 ACL host 调用
在一把 per-context 锁上排队；进程内数百设备/python 线程放大争用）。
原生 CANN 每 token 仅几次 ACL 调用故无感；CANNGE 百次级调用被放大。

## 下一步（需更强工具）

- Ascend prof / ltrace 锁竞争分析（本机无 perf；driver 内部锁需厂商工具）
- 减 ACL 调用次数：边界拷贝合并、staging 批量、去 per-split 全同步
  （事件化）——调用数降下来，锁争用自然缓解
- 注意力链（FA/ROPE）上设备 = 段数从 100+ → 个位数，根治

## 数字小结（Qwen3-4B F16，本批后）

| 配置 | pp t/s | tg t/s |
|---|---|---|
| CPU | 33.8 | 4.6 |
| CANN graph | 7300 | 71.7 |
| CANNGE（上批） | 3.1 | 0.9 |
| **CANNGE+GEMV（本批）** | **5.8** | **1.0** |

## 未 commit

GEMV + 计时桩 + MAX_PLANS 在工作树（cannge 分支 f811cde6f 之上），
等用户明示后随下批一起推送。探针新增 ggml_probe_{gemv,perf}.cpp。
