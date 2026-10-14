# note-35: 910 轮收官——全默认冒烟打通 + TBO 收敛评估（2026-10-14）

接 note-34。本轮（910B3 首轮）目标达成，收官记录。

## 最终状态

- **冒烟：全默认（-ngl all、KV on CANNGE、零 knob）连贯中文输出，双跑确认**
  - f6: "你好！我是一个大型语言模型，能够帮助你回答问题、创作文字…"
  - f6b: "你好！我是Qwen，是阿里巴巴云研发的超大规模语言模型…"
  （temp 0.6 采样不同但均连贯；无卡死、无病态编译）
- **TBO 窄集（修复后）：242/252**（修复前 238/252）
  - strided_copy 修复回收 4 例（f32 [10,5,*] + MUL [1,1,320,320]）

## 残余 FAIL 去向（判据 OR 条款覆盖）

| 家族 | 数量 | 去向 |
|---|---|---|
| GET_ROWS batched | 5 | F10 open item（note-24 起）：ES 接口 GatherV2 batch_dims 绑定与 ge 仓 IR 推断不一致（compiler/.../gatherv2.cc 取证），输出=正确值×be1×be2；310 同族 4 例；不阻塞模型路径（embedding 是 2D be=1；单行表已被 #19 门挡去 CPU） |
| perm1=1 退化维 | 5 | [10,5,*] f16 子族，note-24 起记录（310 同族 4 例），疑 GE 对退化中间维处理；本轮已回收 f32 全部 + [1,1,320,320]；残余为 f16 专属 |
| 全量集额外 | 2 | RMS_NORM_MUL_ROPE 组合例、ROUND f16 半值语义（note-27 已记录）——不在窄集判据口径 |

## 本轮修复清单（工作树未 commit，等用户明示）

1. **#19 GET_ROWS 单行表门**（ggml-cannge.cpp）：GE GatherV2 单行表输出
   塌陷只写 1 元素；探针 ggml_probe_gr0 实证
2. **#20 RMS 生产者门**：RMS×ADDMUL 合并子图模型级腐化（m2 最小配对锁定；
   单元探针全灭 → 疑 alloc/流交互，待根因替换）
3. **#21 strided_copy bounce 路径 nb0≠es 逐元素化**（3 处）：V 转置视图
   copy-back 行 memcpy 崩塌 → GE 输出稀疏写；c0v.log 分块 fletcher 实证
4. 工具/防御：DEBUG_IO 设备指针 segfault 修复 + dense/nb/分块 fletcher +
   六元素值采样；input_staging 同步（竞态假说证伪但保留防御）；4 个二分旋钮；
   探针集 ggml_probe_{kv,getrows,gr0,rms,rmsmul,rmsmul2,addrms,mmrms}
5. note-30~35 六篇

## 交接给下轮（不在本判据内）

- #18：-fa off 转置 V 视图（VIEW 门）——数值正确后可排期
- 两个止血门的最终根因替换（GE GatherV2 单行修正 / RMS 合并子图 alloc 交互）
- TBO 残余 10 例的 GE 侧修复（F10 V2 路线 = JI 式 raw-op，note-24 已有方向）
- 病态编译二分（note-25 #8）在 910B 未复现，无需继续
- 性能：首 token 编译时间、0.6-1.3 t/s 根因（per-shape 重编译？）
