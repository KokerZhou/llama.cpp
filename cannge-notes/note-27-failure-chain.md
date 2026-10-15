# note-27: 冒烟连环失败根因链 + 当前唯一剩余阻塞（2026-10-09）

## 构建注意事项

llama-cli 走 server-impl → llama-ui → 构建期下载 dist.tar.gz（github 不通，每次超时拖慢）。
**配置时加 `-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF`**（tools/ui/CMakeLists.txt，
LLAMA_BUILD_WEBUI 是旧名）。冒烟的 stdout 重定向必须截断"> " EOF 洪泛（CLI stdin EOF 空转，
Agent A 已定位），否则几秒写爆磁盘。

## 冒烟失败根因链（每一层修复后暴露下一层，全部已定位）

| # | 症状 | 根因 | 修复 |
|---|---|---|---|
| 1 | sdma copy error 毒化 GE 会话 | cpy_tensor_async 恒用 D2D，CPU→设备拷贝源是宿主地址 | F14 按 buffer 类型选方向 |
| 2 | "no usable GPU" | 设备类型 ACCEL，-ngl 被忽略 | F15a 改 GPU |
| 3 | 加载期 OOM 137 | 异步 H2D 期间驱动 pin 住全部 staging 页（64GB cgroup） | F15b set/get 同步化 |
| 4 | abort: cache_k_l0 cannot run SET_ROWS | KV 视图预分配在 CANNGE buffer 但门拒 SET_ROWS（dtype/idx 条件） | F16 SET_ROWS 实现（含 I64、宿主 f32→f16 cast），TBO 44/44 |
| 5 | 100% CPU 死循环 | build_view 名字回退自匹配乒乓（同名张量互指） | visited 集合 |
| 6 | "no ES tensor for Vcur-0 op=MUL_MAT" | view 链沿 view_src 走到 CPU 侧链根（切图边界未截断） | F17a has_holder 链行走 |
| 7 | "no ES tensor for Vcur-0 (view)" | 链死端后名字回退匹配的是 node 自身而非 view_src | F17b 回退挂到 view_src 层 |
| 8 | "view wraps a base dim" | 同 buffer 同 span 的 per-head [128,n,8] vs merged [1024,n] 平局选错 | fits() 盒子适配校验 |
| 9 | "no ES tensor"（全候选被拒） | fits 只认逐维盒；merged view 是整 buffer 重解释 | 整 buffer 重解释分支（Reshape 别名） |
| 10 | GE E19999 Reshape 元素数不匹配 | [128,8,256] 缓存槽视图对 2D base 走 Slice+Reshape；正确解是范围盒分解（start/end 偏移沿 base 维分解，盒元素数须等于视图） | 范围盒算法（板对齐校验） |
| 11 | strided_copy sdma error | f16 转置视图 2B 行宽/2B 对齐，SDMA 要求 ≥4B | es<4 或 nb 非 4B 对齐走宿主 bounce |
| 12 | GE 执行期 SMMU Terminate | 密集但宿主驻留的图输入（rope pos 等）直接绑定为设备指针 | analyze 对 host buffer 输入也置 staged |
| 13 | D2D 拷贝 dst=0xfffd…（宿主） | SET_ROWS 散射 dst（node->src[2]）是宿主侧原对象；行拷贝方向只看了 src | ✅ 已修：dst_host 四分支 |
| 14 | TBO f16 暂存类全灭（perm1×12、vs0、DUP-perm×4） | strided_copy bounce 的**设备源输入分支数据流写反**：把未写入的暂存目标 D2H 当数据源散射 | ✅ 已修：D2H 源→宿主镜像→gather 成 dense→H2D |
| 15 | ppl KL 对拍全失真（same-top-p=0、RMS Δp 55.6%、两次逐位一致=确定性） | 分析过程曾被我自写 python .kld 解析器的错位 bug 污染（"argmax 冻结"叙述作废，C++ 工具为唯一可信口径）；**可信结论**（vs 验证过的 ref2 基准）：CANNGE 默认配置 RMS Δp 55.6%/same-top 0%；RMS 落 CPU 恢复到 same-top 79.5% → **RMS-in-graph 是主破坏者**（孤立 RMS [2560,256] 精度完美——图集成问题）；matmul 落 CPU 无变化；全禁（DISABLE_ALL）+任何配置都无变化（后端在场即破坏，与算子无关） | 已提交 d3a3ef81f；修复进行中 |
| 16 | GE BatchMatMulV3 权重 >256K 元素（k=n=640+，f16/f32 同病）输出静默零 | 孤立探针阶梯实证（512 过/640 灭）；已加 L0 门 262144；正路=照 JI 换 BatchMatMulV2 raw-op+显式输出 desc | 已门限，待 V2 路线 |
| 17 | ref-cpu-4b.kld 参考文件损坏（同配置重录 ref2 后对比自证：同配置 100% 一致） | 首份参考不可用导致全部早期 KL 数字存疑；**用 ref2 重测后核心结论不变**（55.625/79.5% 逐位相同） | 教训：对拍前先验证"同配置自比=100%" |

## 当前状态

- TBO：324/333（窄集+SET_ROWS+NORM/DUP/GLU/ROPE+11 个 UNARY；残余 9 个全是已知 open item：
  perm1=1 退化维 4、GET_ROWS batched E13025 4、ROUND f16 半值语义 1）
- 冒烟：加载→初始化→KV 图 GE 编译全过；execute 阶段卡 #13（方案已明，改 set_rows 一处）
- 模式总结：**所有"卡死/乱码"假象的真身=宿主/设备指针混淆的连环案**，每次流中毒后级联报错
  掩盖首犯，必须按时间线抓第一个 ggml 级错误（本 note 的方法论）
