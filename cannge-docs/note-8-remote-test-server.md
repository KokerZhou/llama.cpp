# note-8: 远程测试服务器使用手册（Ascend 310P3 真机）

整理日期：2026-09-10。本文档是远程真机环境的唯一操作入口，连接参数中的凭证一律用占位符，**禁止把真实 token/密码写入本文档或任何落盘文件**。

## 1. 环境概况

| 项 | 值 |
|---|---|
| 硬件 | 2x Ascend310P3（NPU 32768，各约 44GB HBM，容器内可见 /dev/davinci4、davinci5） |
| 系统 | aarch64 Ubuntu 22.04.5 容器（hostname 随机，如 6d59933695d4） |
| CANN | 9.1.0（/usr/local/Ascend/cann-9.1.0），driver 26.0.rc1，ACL 1.17.0 |
| 工具链 | cmake 3.22.1，g++ 11.4.0；无 docker |
| 已验证 | aclInit/GetDeviceCount/GetSocName/GE 全链路端到端跑通（见 report.md，结果包 ~/cannge-remote-results/） |

注意：容器文件系统是**临时的**，重建后所有手动改动（含 pip 依赖、/root 下的代码）丢失。唯一持久化通道是本地结果包与 git。

## 2. 连接方式（关键纪律）

### 2.1 网关特性
- 经跳转网关（Go 实现的一次性凭证代理）接入：`ssh -J jt_<ID>:<TOKEN>@<网关IP>:<端口> root@<目标IP>`
- 凭证**单次使用**：每建立一个到网关的新 TCP 连接并认证一次即作废；连接命令自签发起约 5 分钟有效
- 凭证过期表现：网关接受 TCP 后拒绝转发（`administratively prohibited: platform authorization denied`）或 kex 阶段直接断开
- **严禁并发建立多个连接**——后发的连接会烧掉凭证，把正在用的会话打死

### 2.2 无交互密码（本机无 sshpass/expect）
```bash
mkdir -p /tmp/askp
printf '#!/bin/sh\necho "$SSHPASS"\n' > /tmp/askp/ask.sh && chmod +x /tmp/askp/ask.sh
SSHPASS='<密码>' SSH_ASKPASS=/tmp/askp/ask.sh SSH_ASKPASS_REQUIRE=force DISPLAY= \
  ssh -J jt_<ID>:<TOKEN>@<网关IP>:<端口> -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@<目标IP> '<命令>'
```

### 2.3 长连接（推荐）：ControlMaster 一次建链、本地复用
拿到新凭证后，**一条原子命令**完成建链 + 验证（多试几次循环已在脚本里，但同一时刻只有一个连接在飞）：
```bash
JUMP='jt_<ID>:<TOKEN>@<网关IP>:<端口>'
OPTS='-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=8'
ok=0
for i in 1 2 3; do
  SSHPASS='<密码>' SSH_ASKPASS=/tmp/askp/ask.sh SSH_ASKPASS_REQUIRE=force DISPLAY= \
    setsid nohup ssh -M -S /tmp/askp/cm.sock -o ControlPersist=8h $OPTS -J $JUMP root@<目标IP> -N >/tmp/askp/master.log 2>&1 &
  mpid=$!
  for w in 1 2 3 4 5 6 7 8; do
    sleep 3
    ssh -S /tmp/askp/cm.sock -O check root@<目标IP> 2>/dev/null && { ok=1; break; }
    kill -0 $mpid 2>/dev/null || break
  done
  [ $ok -eq 1 ] && break
  kill $mpid 2>/dev/null; sleep 5
done
echo "master_ok=$ok"
```
之后所有命令走套接字（不占凭证）：
```bash
ssh -S /tmp/askp/cm.sock -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@<目标IP> '<命令>'
```
- `setsid nohup` + `ControlPersist=8h`：主连接脱离 shell 进程组，空闲 8 小时保持
- 断开检测：`ssh -S /tmp/askp/cm.sock -O check root@<目标IP>`；失败即需向用户要新凭证重建，**不要自行重连**

## 3. 环境初始化（容器重建后必做）

```bash
# 1. GE 运行时需要 python 依赖（否则 GEInitialize 失败：InitCannKB import 报错）
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py
pip install "numpy<2" decorator sympy cffi pyyaml absl-py requests protobuf scipy attrs psutil
# 2. 每个构建/运行命令前都要
source /usr/local/Ascend/cann-9.1.0/set_env.sh
```
不 source set_env.sh 的典型症状：aclInit 返回 500000、设备数 0、npu-smi 报 libc_sec.so 缺失。

## 4. 标准工作流

```bash
# 同步代码（只同步工作区，不带 .git/构建目录）
rsync -az -e 'ssh -S /tmp/askp/cm.sock -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' \
  --exclude .git --exclude build-native --exclude build-cannge --exclude build \
  <本地 llama.cpp 路径>/ root@<目标IP>:/root/cannge-ggml/

# 真机编译（SOC_TYPE 用 Ascend310P3）
ssh -S /tmp/askp/cm.sock ... 'source /usr/local/Ascend/cann-9.1.0/set_env.sh; \
  cd /root/cannge-ggml && cmake -B build -DGGML_CANNGE=ON -DGGML_CANN=OFF -DSOC_TYPE=Ascend310P3 \
  && cmake --build build --target ggml-cannge test-backend-ops -j$(nproc)'

# 数值测试窄集（-o 过滤名是大写；GGML_CANNGE_GRAPH=1 是整图路径开关）
ssh -S /tmp/askp/cm.sock ... 'source /usr/local/Ascend/cann-9.1.0/set_env.sh; \
  cd /root/cannge-ggml && GGML_CANNGE_GRAPH=1 ./build/bin/test-backend-ops test -b CANNGE0 \
  -o ADD,MUL,MUL_MAT,SCALE,RMS_NORM,SOFT_MAX,SILU,GET_ROWS,CONCAT'

# 设备状态
ssh -S /tmp/askp/cm.sock ... 'npu-smi info'
```

结果回收：测试日志、remote-vs-local.diff、报告一律拉回本地 `~/cannge-remote-results/`（git 或 rsync 反向），不要只留在容器里。

## 5. 共享环境纪律（不污染服务器）

- 只在 /root 下自建目录（cannge-ggml、cannge-probe 等），不动系统目录与其他用户文件
- 不修改全局配置（/etc、全局 git config、shell rc）；环境变量用命令内 export，不写 rc
- 容器内不 commit 代码副本；真源永远在本机 git 仓库
- 装包只装 python 用户级依赖（容器重建即消失，无残留）；不装系统包、不升级现有包
- 设备是共享的：跑测试前 `npu-smi info` 确认无他人进程；测试用默认 device 0，不独占；失败进程及时清理（aclFinalize/进程退出）
- 凭证是稀缺资源：一个凭证只建一条主连接，用完留给后续复用，不随手新建连接

## 6. 已知坑清单（真机实测攒下）

| 症状 | 原因 | 解法 |
|---|---|---|
| aclInit=500000、设备数 0 | 没 source set_env.sh | source /usr/local/Ascend/cann-9.1.0/set_env.sh |
| GEInitialize 失败（InitCannKB import ...） | 缺 python 依赖 | 见 §3 pip 列表 |
| npu-smi: libc_sec.so 缺失 | 同上 | source set_env.sh 后直接可用 |
| 第二个 ssh 连接被拒/ kex 断开 | 凭证单次使用 | 等用户发新凭证；用 ControlMaster 复用 |
| MUL_MAT 图编译 EZ9999 | 310P 要求 matmul 维度 32B 对齐（CheckDimsAligned310P） | supports_op 拒不对齐形状（下阶段） |
| MatMulV3 编译报 infer_datatype 缺失 | 9.1 的 MatMulV3 无 datatype 推导注册 | 用 BatchMatMulV3（等价，已验证） |
| RMS_NORM 融合核间歇丢行（约半数运行、无报错） | 310P3 小 shape 下的核问题 | 元语实现 x*rsqrt(mean(x^2)+eps)，8/8 稳定 |
| GE 编译首调 10s+ | 正常（整图编译） |  plan 缓存命中后 ~0.2ms |
