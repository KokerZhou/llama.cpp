# note-25: CANNGE 模型集成阶段（Qwen3-4B 冒烟，2026-10-08）

接 note-24。目标：Qwen3 dense/MoE/VL F16 在 CANNGE 上跑通。本文件记录模型级联调发现。

## 修复批次（在 note-24 的 F9-F13 之上）

| 批 | 内容 | 关键证据/机制 |
|---|---|---|
| F14 | cpy_tensor_async 恒用 D2D → CPU 输出拷入设备时 sdma copy error 毒化 GE 会话（stream 创建连锁失败）。按 src/dst buffer 类型选 H2D/D2H/D2D | ggml-cannge.cpp:266 |
| F15a | 设备类型 ACCEL→GPU（对齐原生 ggml-cann），-ngl 生效、权重上 HBM | llama.cpp llama_supports_gpu_offload 只认 GPU/IGPU |
| F15b | set/get_tensor_async 加同步：异步 H2D 期间驱动把全部 staging 页 pin 在宿主，64GB cgroup 加载即 OOM（峰值 45GB；同步后稳态 25.6GB） | smaps Pss_Anon 45.8GB；memory.usage_in_bytes 曲线 |
| F16 | **SET_ROWS 后端散射实现**：门放行（f16/f32 data、I32/I64 idx、contiguous rows、I64 idx 不注册为 GE 输入——GE 不消费它，还绕开 ge_dtype 不支持 I64 的 build 失败）；build 把节点别名到 src2（cache）；execute 后按 idx 逐行 D2D/H2D 散射，**f32 rope 输出→f16 cache 的 dtype 转换在宿主逐行做**（ggml-cpu set_rows_impl 语义：dst[c[i],i2,i3]=data[i,i2,i3]） | TBO SET_ROWS 44/44；llama-kv-cache.cpp:1350/1385/1406 实测 k_idxs 为 I64、f32→f16 |
| F17a | **view 链在切图边界截断**：build_view 的链行走改为 has_holder 语义（沿"有 ES holder"的链走），split 图里调度器把边界张量（自身是 view）作为图输入，链条不能穿过它走到 CPU 侧 root | 冒烟报 "no ES tensor for Vcur-0 op=MUL_MAT" |
| F17b | **同名张量回退**：调度器拷贝保留原名，而 view_src 可能仍指向拷贝前对象 → has_holder 加"按名匹配已注册张量" | 冒烟报 "view_src=Vcur-0 registered=0" 但输入表有同名 "Vcur-0 op=RESHAPE" |

## 关键机制认知（血泪）

1. **ggml mul_mat 输出恒 F32**（ggml.c:3342），GE 输出随输入 dtype → f16 输入必须 Cast（F13）；SET_ROWS 的 dst(cache) f16 / data(rope 输出) f32 同理（F16）
2. **调度器 split 图**：不支持的算子（M=1 的 MUL_MAT 过不了 16 元素门、ROPE、FA）→ CPU 子图；边界张量做后端拷贝（名字带 `CANNGE0#原名#n` 前缀），**view_src 链不随拷贝重定向**（F17a/b 的根源）
3. **KV 写入**：本 fork 用 ggml_set_rows（src 顺序怪异：src0=data, src1=idx, src2=cache，ggml.c:4023），idx 是 **I64**（llama-kv-cache build_input_k_idxs）
4. **TBO nf=1 逐节点切图**：PERMUTE 物化写回自身别名是双射（缓冲内容不变，探针 same=600/600）；staging 的 strided 散射/聚集拷贝必须按元素宽度 2D 拷贝（width=es, height=ne0, pitch=nb[0]），nb[0]>es 的 permuted view 下 pitched 大行宽拷贝非法
5. **GE 病态编译**（note-10 F7 家族）在真机模型图上表现为：加载/首 token 阶段长时间停留（>10min），CPU 忙=编译、闲=挂死（判别法）

## 冒烟时间线（Qwen3-4B F16, -ngl 99, -c 512）

1. sdma 崩溃（F14）→ 修复
2. "no usable GPU"（F15a）→ 修复
3. OOM 137（F15b）→ 修复，稳态 25.6GB
4. -nkvo 下出 token 但**乱码**（.awl蚤蚤SPORT，0.3 t/s）→ 纯 CPU 对照超时不可行（此 fork CPU 路径同样缓慢），0.6B 参考模型已下载
5. 无 -nkvo：KV 视图预分配 assert（SET_ROWS 不支持）→ F16 后通过
6. F17a/b 后 build 错误清零
7. 当前：加载/首 token 阶段超长（疑似 GE 病态编译 or 大 graph 编译，长窗+CPU 采样判别中）
8. **长窗判别结论（1100s）**：单线程 98.8% CPU 空转 18 分钟无进展、GE runtime 线程（ge_davidmdlrun）全闲 → **GE 病态编译**（note-10 F7 家族）命中首个模型图，永不完成。TE 时期 F7 的 EH0008/病态编译在真机图上复现。需要：打印编译中的图签名/节点数（计划编译起点日志）→ 按 ubatch/子图二分定位触发形状

## 当前 diff 规模

4 文件 +449/-54（ggml-cannge.cpp / graph-build.cpp / plan.cpp / plan.h），未 commit（纪律：等批准）

## 待办

- 长窗冒烟结论（编译通过→看输出是否仍乱码；乱码→按算子二分：embedding GET_ROWS / f16 matmul Cast 路径 / RMS_NORM）
- ROPE 接入（RotaryPositionEmbedding 310P 已取证，es_transformer 有原生 op；需 cos/sin 构图或元语实现）
- TBO 全量基线刷新 + note-16 基线表更新
- 性能：首 token 编译时间、0.3 t/s 根因（per-shape 重编译？）
