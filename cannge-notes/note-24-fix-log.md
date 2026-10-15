# note-24: CANNGE 修复与补齐日志（本机 310P3，2026-10-08 起）

纪律：控制变量，一次一批，每批 改码→构建→TBO窄集验证→落盘。不 commit（等用户批准）。
基线：160/45/2366（note-16 §3，28558340a 口径，二进制自报 160/205）。

| 批 | 内容 | 改动 | TBO 结果 | 状态 |
|---|---|---|---|---|
| F9 | 哨兵 OP_NONE 输出 bug（4 例 FAIL 根因，nf>1 整图必败） | plan.cpp analyze 边界输出循环跳过 op==NONE 节点（其 buffer 为 harness 所有，溢出检测用，无需 GE 端口） | 二进制 160/205→161/205；ADD nf=2 perm1=1 FAIL→OK | ✅ |
| F7 | f32 MUL_MAT 门对齐错（GE 内部降 fp16，32B 判据落在 fp16 视图上） | ggml-cannge.cpp MUL_MAT 门：ASCEND_310P 下 f32 align_elems=16（其他 dtype/SoC 不变） | MUL_MAT FAIL 4→2、NS 1667→1669；F7 两例 FAIL→NS | ✅ 结案 |
| F11 | SCALE bias 丢弃（op_params[1] 未接） | graph-build.cpp SCALE 分支：Mul(s) 后 bias!=0 时 Add(scalar)（语义对齐 ggml-cpu ops.cpp mad1） | SCALE 1/3→3/1（剩 1 例 inplace=F8） | ✅ |
| 批① | 参数化 SOFT_MAX（scale+mask，拒 sinks/ALiBi）+ batched MUL_MAT（3D/4D，批维相等） | ggml-cannge.cpp SOFT_MAX 门：src1 mask f16/f32 允许（ne[0] 必须相等，其余维 broadcast），src[2]/max_bias 拒；MUL_MAT 门放开 2-4D 批维相等；graph-build.cpp SOFT_MAX：Mul(scale)+Cast(mask)+Add+SoftmaxV2 | 构建+TBO 进行中 | ⏳ |

## 关键事实备查

- 二进制自报分母 = 非 NS 用例（205→203 因 F7 两例转 NS）；解析器总数恒定 2571
- SOFT_MAX sinks 5 例在阈值附近抖动（ERR ~3e-6 vs 1e-6），flaky 分类 = F12 症状，批①将其确定性转 NS
- F11 后 SOFT_MAX 出现 1 例 OK→FAIL 抖动，同属 F12 症状
- ggml SCALE 语义：dst = src*s + b（b==0 时纯 scale）
- ggml SOFT_MAX 语义：softmax(x*scale + mask)，mask 可 f16，ALiBi 仅 max_bias>0 时生效
- MUL_MAT batched：ggml ne [K,N,b] × [K,M,b] → [N,M,b]；GE shape 反转后 BatchMatMulV3(es1, es0, adj_x2=true) 天然批处理

| 批② | F8 view_src 型 in-place 门（SOFT_MAX/RMS_NORM/SCALE inplace 全止血）+ MUL_MAT 操作数连续性门（permuted 叶子→NS） | ggml-cannge.cpp：view_src 非空且 op 非 VIEW/PERMUTE/TRANSPOSE 则拒；MUL_MAT 门加 is_contiguous 双操作数 | 214/248（二进制）；SCALE/RMS_NORM/SOFT_MAX FAIL 清零 | ✅ |
| 批③ | **F13 根因**：本 ggml 版本 ggml_mul_mat 结果恒为 F32（ggml.c:3342），GE BatchMatMulV3 输出随输入 dtype（f16→f16），写回 ggml 的 f32 缓冲区只写一半→半零/垃圾。f32 输入路径 GE 内部降 fp16 但输出 f32 故正常。探针 3/3"通过"是读法假象（按 f16 读了前半缓冲区） | graph-build.cpp MUL_MAT：node->type != src0->type 时插 Cast | MUL_MAT 5/11→**11/11 FAIL 清零**；窄集 220/248 | ✅ 结案 |
| 批④ | GET_ROWS 同款 dtype 契约（ggml_get_rows 强制 F32 输出，ggml.c:3963）+ 移除临时 DEBUG 桩 | graph-build.cpp GET_ROWS：f16 输入时 Cast 到 node->type | 构建+TBO 进行中 | ⏳ |

| 批⑤ | **staging 完整版**：输入侧（非 dense 输入→行级 2D 拷贝到暂存再绑定）+ 输出侧（非 dense 边界输出→GE 写 dense 暂存→execute 后 strided 散回）。期间修两个实现 bug：(1) 初版 pitched 2D 拷贝假设 nb[1]≥行宽，permuted view 不成立；(2) **overlaps 与 strided 同时命中时 strided 必须优先**（v->data 与输入 b 同址，旧逻辑走 dense memcpy 回写把逻辑序数据线性覆盖物理缓冲，b[1]=本应属于 b[20] 的值——探针实锤 diff=576） | plan.h/plan.cpp/ggml-cannge.cpp 三件套 | **239/248（基线 160/205）**；perm1=1 家族 20→4；探针验证 node0 写回恒等 same=600/600 | ✅ |

| 批⑥ | **F14**: cpy_tensor_async 跨后端拷贝恒用 D2D，CPU 输出→CANNGE 时 src 是宿主地址 → sdma copy error 毒化 GE 会话。按 buffer 类型选 H2D/D2H/D2D | ggml-cannge.cpp cpy_tensor_async | 解码不再 sdma 崩溃 | ✅ |
| 批⑦ | **F15a**: 设备类型 ACCEL→GPU（对齐原生 ggml-cann），-ngl 生效，权重上 HBM；**F15b**: set/get_tensor_async 改同步——异步 H2D 期间驱动 pin 住全部 staging 宿主页，64GB cgroup 下加载即 OOM（峰值 45GB→同步后 25.6GB 稳态） | ggml-cannge.cpp | 加载不 OOM，进入解码 | ✅ |

## 模型冒烟状态（Qwen3-4B F16, -ngl 99 -nkvo -c 512）

- 加载→解码链路通：`[ Prompt: 0.8 t/s | Generation: 0.3 t/s ]`，但**输出为乱码**（".awl蚤蚤SPORT..."）
- 待判别：权重拷贝错 / CANNGE 算子数值错 / KV 跨端路径错。手段：CPU 参考输出（-ngl 0, temp 0）对照
- 已知坑：KV 视图预分配在 CANNGE buffer 时 SET_ROWS（不支持）直接 abort → 需 -nkvo 或实现 SET_ROWS
- SET_ROWS 是 KV 写入硬依赖，且 ROPE 已在批⑧排期（RotaryPositionEmbedding 310P 已取证 AiCore/f16/f32/ND）

## 当前 TBO 状态（batch5g，28558340a+本批改动）

235 OK / 8 FAIL / 2323 NS（二进制 239/248）。残余 8 全归两类 open item：
1. perm1=1 残 4 例：[10,5,1,1]（ne2=ne3=1 且 v.nb=[40,4,20,400]）——staging 已生效但数值仍错，待查（疑 GE 对退化的中间维处理）
2. GET_ROWS batched E13025 4 例：GE 推导输出 = 正确值 × be1×be2（取证见下）

## staging v1 设计（note-14 §5 落地）

- 只覆盖**输入侧**：analyze 对非 dense 输入置 input_staged[i]；execute 时 aclrtMalloc 暂存（plan 生命周期复用，
  尺寸按签名固定），逐行 D2D 拷到暂存，GE 绑暂存
- **输出侧**（非 dense 边界输出）仍走 needs_staging 确定性失败，留后续 phase
- 性能注意：逐行拷贝对 [1,1,320,320] 类形状 = 102k 次小拷贝，正确优先；合并维度/按 stride 排序分块是后续优化
- 失败三分更新：perm1=1 家族从 "staging 缺口" 转为 "拷贝语义 bug"（nb[1]<行宽 时 pitched copy 非法）

## F10 遗留（batched E13025 轴之谜）

- 已取证：GE 形状推断源码 ge 仓 compiler/graph/optimize/symbolic/infer_symbolic_shape/infer/gatherv2.cc：
  out = x[:axis] + indices[batch_dims:] + x[axis+1:]，batch_dims 为 attr（ES 接口默认 0）
- 实测规律：batched（be>1）时 GE 推导输出 = 正确值 × be1×be2（逐例验证 #1 ratio=be2、#2/5 ratio=be1、#3 ratio=be1×be2）
- 按源码语义推导应与 user 一致，实际偏大 → 疑似 ES 构图层（EsGatherV2 对 4D+axis 常量绑定）与 IR 推断层不一致；暂挂为 open item，不阻塞模型路径（embedding 查表是 2D，工作正常）
- vs0=1 两例归 staging 批次

## F13 判别过程备查（重要教训）

- 探针 A2 哨兵实验证明 GE **写了**结果（非 GE 静默不写），位置/值正确——"探针通过、TBO 失败"的悖论来自探针按 f16 只读前 N*M*2 字节
- C2 网格 n=0/8 列为零 = k=16 时该位置真值近零的数据假象，非 kernel bug
- 判别工具：临时 env 调试桩 GGML_CANNGE_DEBUG_MM2（**已计划移除**）；TBO nf=1 逐节点切图（单节点 graph_view）
- 计划缓存语义：编译失败缓存 FAILED plan 且返回错误（ggml-cannge.cpp:393-410），GE 执行成功但写半截是另一回事

## 剩余队列

- F8：rms_norm_inplace 等 view_src 型 in-place（台账修法：白名单+门拒）
- F10：GET_ROWS batched 轴语义 + f16→F32 Cast（8 例）
- F13：f16 MUL_MAT 静默零 2 例（设备判别实验）
- staging 落地（23 例 ADD/MUL perm1=1 + GET_ROWS vs0）
- 后续补齐：ROPE（先取证）、SET_ROWS、DUP/NORM/unary、GLU
