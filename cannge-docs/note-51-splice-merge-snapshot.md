# note-51: 批K 中途快照（合并已通、探针精确，模型级一处 reshape 守卫待修）

## 状态：批K 图内新鲜行拼接（前缀手术的 build 级实现）

背景：GE 散射族静默无效（note-49），v_cont CPU 段消不掉。批K 改为**在 build 层把当前行算术拼进注意力**：
- V 侧：`build_cont_chain`——CONT(转置视图) 不再物化，直接复用 fragment 内已建好的 TRANSPOSE holder（之前用 view_src 链重建会丢 PERMUTE——ggml 会折叠视图链！），再用 one-hot 选择把 Vcur 拼到 slot n；
- K 侧：batched GQA gemv 的 sum `{G,R,N}` 上，额外加一个 `{G,R,1}` 小 gemv（Kcur·q）+ one-hot 选择拼到 slot n；
- 触发条件：同 fragment 内找得到写同一 cache base 的 SET_ROWS 且其 data/idx 有 GE holder（跨 fragment 时自动退回原行为，安全）；
- idx 输入 Always 注册（splice 在默认 host-scatter 模式下也要读 idx——批G 的"idx 永不进 GE"假设被废除）。

## 已验证

- 探针 `ggml_probe_kvattn`（完整模型形态：cache + set_rows + K/V 链 + GQA kq + softmax + kqv）：**kq 4.7e-4 / kqv 5.8e-4 双 PASS**（f16 产物量级）。
- decode 片结构：**4 片/token**（embd | 36 层合一 1432 节点 | get_rows | lm head 区）——JittorInfer 式结构达成。

## 待修（下次第一步）

模型级 E19999（prefill 合并 fragment）：`Reshape_52 x=[2,1024] -> [8,128,1]`——prefill 的 Vcur 是 {1024,2}（M=2 两行散射），拼接的行 reshape 只写了单排行形状。**修法：两个拼接点都要求 `ggml_nelements(sr->idx)==1`，多行散射（prefill）跳过拼接**（prefill 本来就不走合并收益路径，保持 host 散射+原 staging 即可）。修完跑 =all 冒烟 + -n 100 基准（4 片结构的收益首次可测）。

## 里程碑（截至批J）

0.6 → 1.7 → 3.8 → 6.0 → 6.9 → 10.4 → **10.8 t/s**（=all，-n 100）；默认 2.4；=v 10.2。距原始 CANN 71.7 剩 6.6x。批K 合并若修通，后端稳态 6.3ms + 片数 77→~6，理论上 15-25 t/s 量级。
