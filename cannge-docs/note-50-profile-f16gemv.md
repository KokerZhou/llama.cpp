# note-50: 剖面驱动批H——稳态账目、f16 产物默认化、SET_ROWS 快照单同步

## 1. 当前 =all decode 结构（批F 后）

77 片/token：36×[CPU v_cont 5 节点] + 36×[CANNGE 35 节点（kq+softmax+kqv+wo+ffn 全链 + K/Q 图内重排）] + embd + 尾部。swiglu 收编与 K 替换后 CANNGE 侧已很肥；**36 个 CPU v_cont 片段是剩余片数的大头**（其消除被图内 KV 写卡死，note-49）。

## 2. DBG-TIME 稳态分解（-n 60，剔除一次性编译）

一次性 GE 编译 18 签名 ≈ 20s/进程（签名本已无名、跨层共享；这就是短基准 pp/tg 显慢的原因）。稳态每 token ≈ 60ms 后端时间：
- post ~24ms：回写 + SET_ROWS host 散射 + **每片一次流 drain**——drain 等待的就是该片 GPU 真执行时间；
- bind ~28ms：每片 ~15 个小 staging 拷贝的固定开销；
- exec(仅入队) ~10ms、analyze ~7ms、sig ~2.7ms。

关键洞察：post 的 drain 暴露 GPU 真时间；**M==1 gemv 的 f32 产物物化是带宽瓶颈**（wo 42MB、ffn gate/up 各 100MB f32 读写/次）。

## 3. f16 产物默认化（批H 主要收益）

`Mul+ReduceSum` gemv 产物 f16 化（`GGML_CANNGE_GEMV16=0` 退出；batched GQA 分支产物小维持 f32）。探针实测 max_rel 5.5e-4（f16 乘积舍入，GE 内部累加），冒烟+120 token 长程连贯。

- post 2691→1434ms（**减半**，drain 等待时间证实带宽理论）
- tg：=all **6.9**、=v **6.8**、默认 **2.1**（批G 为 6.3/6.0/2.0）；=all 长程（120 tok 杭州景点）7.1 且内容事实正确。

## 4. SET_ROWS 快照单同步

36×2 个阻塞式 D2H 快照（各带全流 sync）→ 异步 D2H 入队 + 复制回写后**一次**共享 drain。post 略降（主要收益被 f16 覆盖前已完成测量：2691→2691 基线内）。

## 5. 剩余账目（每 token ~145ms → 目标 13.9ms）

- CPU 侧 ~85ms：36 v_cont 片段 + lm head（CPU ~15ms）+ 调度器 77 片边界管理。**片数不减无可降**。
- 后端稳态 ~60ms：bind 28（拷贝固定成本）+ post 24（GPU+sync）。
- 下一步排序：① v_cont 消 CPU 段（需前缀注意力手术或可用散射，大改）；② lm head 上 GE（~~cap 之上走 BatchMatMulV3~~ 实测改走分片 Mul+ReduceSum+Concat，批I 已落地）；③ batched GQA gemv 也 f16（小头）；④ 调度器边界开销（events 曾负收益，重测）。

## 6. 批I：lm head 上 GE（分片 gemv）——超预期的 50%

E40021 的 lm head 不走 BatchMatMulV3（M=1 对齐限制依旧），改 **N 分片**（chunk=13107，12 片 Slice+Mul+ReduceSum+Concat 成一个 GE 图，门放开不再拒）。Concat 第三参数是输入个数不是 axis（E19999 26214 vs 151936 一把定位）。**logits  feeding 采样器：产物强制 f32**（探针 3.7e-6；GEMV16 不适用）。

效果远超预估的"省 15ms CPU"：=all **6.9→10.4 t/s**（=v 10.2，默认 2.4）。lm head CPU 段的消除同时省掉了它强制的整同步/边界结构。

里程碑：0.6 → 1.7 → 3.8 → 6.0 → 6.9 → **10.4** tg；距 71.7 剩 6.9x。剩余：36 v_cont CPU 段（消段=散射或前缀手术）、调度边界。

## 7. 批J：batched GQA gemv f16 化 + 事件复测

- kq/v batched gemv 产物同旋钮 f16（chain 探针 3.6e-5 保持精确）：=all **10.4→10.8**。
- 事件化第三次负收益（9.8 vs 10.8）——此后不再重试。
- 批J 后稳态：后端 ~6.3ms/token，token 93ms 的 ~87ms 在调度器+CPU 侧（36 v_cont CPU 段是最大单一来源）。
- 下一刀只能是 v_cont 消段：前缀注意力手术（kq/v 按"缓存前缀+当前行"拆开构图，绕开散射）或 aclnn 直通散射调研。
