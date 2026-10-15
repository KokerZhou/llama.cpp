# note-38: 性能定位修正——KV-on 路径的序列化 D2H+同步是 tg 主犯（2026-10-14）

接 note-37。回应"图编译不可能这么快"的质疑并重做测量，结论修正。

## 质疑的解答：0.044ms 是真实执行延迟，但那是"缓存命中执行"

- 探针补逐次同步后：1 节点图 e2e **0.044ms**、GEMV 0.079ms——依然快
- 关键：**编译只在首次发生**（plan 缓存），计时区间全部是缓存命中执行。
  小 GE 图（1-2 kernel）设备执行确实是几十 µs 级
- 所以"每子图固定开销"理论仍然成立为**否定结论**：100+ 段×0.05ms 仅 ~16ms

## tg 主犯实锤：KV-on 设备路径（消融实验）

GEMV 后同配置对比（Qwen3-4B F16，-n 48）：

| 配置 | pp t/s | tg t/s |
|---|---|---|
| CANNGE+GEMV **KV-on** | 5.8 | 0.9 |
| CANNGE+GEMV **-nkvo** | **32.3** | **2.2** |
| CPU 参考 | 33.8 | 4.6 |

- **KV-on 的代价：pp −26.5 t/s、tg −1.3 t/s**（≈650ms/token 的胶水开销）
- 拷贝量化（DEBUG_IO，12 token，4632 次拷贝）：**全部为小件 H2D
  （≤288KB，无 4MB 级）**；K/V 数据到 CPU FA 走的是 **get_tensor_async
  （全 cache [1024,512] = 1MB/layer 的 D2H）**
- 机制：每次 get/set/cpy 都 `aclrtSynchronizeStream` **全流排空**——
  K 全 cache 读 ×36 层 + SET_ROWS 散射 + V 物化回写，每 token 数十次
  排空把设备流水线切成串行小段；时间消失在同步等待（进程 0% CPU、
  非自愿切换 0 与之一致）。note-37 的"驱动锁"判定修正为
  **"KV 路径的逐次拷贝+全同步序列化"**（更可操作）

## 残余差距（-nkvo 后 tg 2.2 vs CPU 4.6）

~230ms/token 的残余胶水 = 非 KV 的边界拷贝/同步/sched 开销。
消减方向同下。

## 下轮行动（按性价比）

1. **get/set/cpy 同步减次**：跨 split 的多个 D2H/H2D 合并后一次同步；
   或事件化（sched 支持 event 接口，CANNGE 未实现 .event_new）
2. **K 读活跃区化**：物化 [0..n_past) 而非全 cache（sched 物化的是 base
   全量——查 dup_tensor_layout 输入选择）
3. **V 回写**：#21 的逐元素 bounce 在 KV-on 下每层都跑——nb0%4==0 且
   es==2 时可用 4B 成对读（源密侧连续）+ 设备侧 scatter 替代
4. 注意力链上设备（根治，消除整条 KV 跨界）

## 验证状态

- 冒烟连贯（g2）✓；TBO MUL_MAT 25/25 ✓；GEMV 探针 3e-6 ✓
- note-37 的"驱动锁"表述以本 note 修正为准
- 未 commit；新增/修改：GEMV、MAX_PLANS 256、DEBUG_TIMING 桩、
  ggml_probe_{gemv,perf}.cpp
