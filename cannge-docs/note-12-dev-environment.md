# note-12: CANNGE 开发环境运维手册

> **2026-10-08 更新（收敛）**：非代码资产（笔记、测试归档）的根目录统一为
> `/workspace/shengjiayi/Tsinghua/.kimi-private/`（工作区侧）。home 下的
> `~/.kimi-private` 已废弃并删除——F4 与 910 两轮归档（remote-results-310-f4、
> remote-results-910b）因此丢失，只余 note-9/10/16 里的摘要数字，本文旧路径引用以本条为准。
> 910 的 npussh keyring 当前损坏（`secret-tool EACCES`），用前需重新 `npussh login`
> 或改用 cannlab-ssh 隧道。代码线收敛：远端一律通过 git（fork）对齐，不再用 rsync 整树同步。

建立：2026-09-11。规范对象：CANNGE 后端（ggml/src/ggml-cannge/）开发所需的全部环境资产：
310 长连接真机、910 快通道、本地容器、本机原生构建、外网代理。
凭证纪律的唯一权威来源是 note-8，本文只引用纪律、复制流程，**绝不写入任何真实凭证**；
凡需凭证处一律写"向用户索取"。

## 1. 环境资产总表

| 资产 | 位置/形态 | 用途 | 权威出处 |
|---|---|---|---|
| 310P3 真机长连接 | 远程 aarch64 容器，2x Ascend 310P3，CANN 9.1.0 | 主验证环境（第一目标 SoC） | note-8 |
| 910B3 快通道 | CANNLab 开发环境（npussh 接入），1x Ascend 910B3 | 第二 SoC 编译/运行验证 | note-9、report-9.1 |
| 本地 CANN 容器 | 镜像 `quay.io/ascend/cann:9.1.0-310p-ubuntu22.04-py3.10-devel` | 头文件取证（ES/GE API 精读）、容器内编译 | note-5、note-6 |
| 本机原生构建 | llama.cpp/build-native/（x86_64，Release） | 上游代码回归、非 CANN 路径验证 | llama.cpp/AGENTS.md |
| 外网代理 | `http://127.0.0.1:63145`（用户口径，note 中无记录，见文末待核实清单） | 容器/工具联网 | 用户口径 |
| 连接工具 | `.kimi-private/tools/`（node22 运行时 + npm-global 的 npussh/npuscp/npursync/npusftp） | 910 通道 | 本目录实测 |

## 2. 310 长连接：获取与纪律

### 2.1 获取流程

1. **向用户索取**网关凭证（jt_<ID>:<TOKEN>@<网关IP>:<端口>、目标 IP、SSH 密码）。
2. 按 note-8 §2.3 的一条原子命令建 ControlMaster 主连接（套接字 `/tmp/askp/cm.sock`，
   ControlPersist=8h，脚本内含最多 3 次重试，同一时刻只允许一个连接在飞）。
3. 用 `ssh -S /tmp/askp/cm.sock -O check root@<目标IP>` 验证；之后所有命令走套接字复用。

### 2.2 纪律（全部继承 note-8，此处为规范性重申）

- **禁止**并发建立多个到网关的 TCP 连接：凭证单次使用，后发的连接会烧掉凭证并打死正在
  用的会话。
- **禁止**自行重连。断开检测失败后必须停下，向用户索取新凭证。
- 凭证**禁止**写入任何落盘文件（含本文档、脚本、shell history）；用完的主连接保留给后续
  复用，不随手新建。
- 远程容器文件系统是临时的：重建后必须重做 note-8 §3 的环境初始化
  （get-pip.py + `pip install "numpy<2" decorator sympy cffi pyyaml absl-py requests protobuf scipy attrs psutil`，
  否则 GEInitialize 在 InitCannKB import 处失败）。
- 共享环境纪律：只在 /root 下自建目录（远程代码目录 `/root/cannge-ggml`）、不动系统目录与
  全局配置、容器内不 commit、跑测试前 `npu-smi info` 确认无他人进程、失败进程及时清理。
- 标准命令模板（构建/运行前必须 `source /usr/local/Ascend/cann-9.1.0/set_env.sh`）：

```bash
# 代码同步（排除 .git 与构建目录）
rsync -az -e 'ssh -S /tmp/askp/cm.sock -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' \
  --exclude .git --exclude build-native --exclude build-cannge --exclude build \
  <本地 llama.cpp>/ root@<目标IP>:/root/cannge-ggml/

# 真机编译（note-8 §4；SOC_TYPE 用 Ascend310P3）
ssh -S /tmp/askp/cm.sock ... 'source /usr/local/Ascend/cann-9.1.0/set_env.sh; \
  cd /root/cannge-ggml && cmake -B build -DGGML_CANNGE=ON -DGGML_CANN=OFF -DSOC_TYPE=Ascend310P3 \
  && cmake --build build --target ggml-cannge test-backend-ops -j$(nproc)'

# 结果一律拉回本地 ~/.kimi-private/remote-results*/ 归档，禁止只留容器内
```

## 3. 910 快通道：npussh / npursync

### 3.1 工具位置与登录

工具安装在 `.kimi-private/tools/`：node22 运行时（`tools/node22/bin/`）+
全局 npm 包（`tools/npm-global/bin/npussh|npuscp|npursync|npusftp`，包名 npussh）。
使用时把二者加入 PATH（注意是**工作区**下的 .kimi-private，不是 $HOME）：

```bash
export PATH="/workspace/shengjiayi/Tsinghua/.kimi-private/tools/node22/bin:/workspace/shengjiayi/Tsinghua/.kimi-private/tools/npm-global/bin:$PATH"
npussh login      # 系统浏览器 OAuth 登录（GitCode），凭据存系统密钥环，不写项目目录
npussh list       # 查看账号下环境（910B3 环境名为 DevEnvC_Abwb9，见 report-9.1）
```

依赖：Node.js 20+（node22 满足）、OpenSSH 客户端、unzip、rsync（本地与远端都要）。

### 3.2 标准用法

```bash
# 远程命令（像普通 ssh）
npussh <环境名或ID> '<命令>'

# 增量同步代码到远端（910 轮实证目录 /home/developer/cannge-ggml）
npursync -av --delete <本地 llama.cpp>/ <环境名或ID>:/home/developer/cannge-ggml/

# 拉回结果
npursync -av <环境名或ID>:/home/developer/<结果目录>/ ./local-results/
```

910 环境每个构建/运行命令前必须 source 用户级 toolkit 环境：
`source /home/developer/Ascend/ascend-toolkit/set_env.sh`。已确认（2026-09-11）：
`ascend-toolkit` 是指向 `cann-X.X.X`（现 cann-9.1.0）的符号链接，两个路径等价，
source 任一即可。

### 3.3 纪律

- 910 环境的 CANN toolkit 是**用户级安装**（~/Ascend 下），禁止动系统目录与全局配置。
- 与 310 不同，910 通道（CANNLab）不烧单次凭证，但同样**禁止**在仓库或笔记中写入任何
  token；npussh 的凭据由系统密钥环托管。
- 远端代码以本机 git 仓库为唯一真源；远端只允许临时改动，验证结束必须 diff 落回或还原
  （910 轮纪律：验证后逐文件 diff 远程 ggml-cannge/ 与本地一致，见 report-9.1）。
- 9.0 轮的 ES 兼容 shim（try_compile 探测等）在远程 `/home/developer/cannge-ggml` 留有副本、
  **未落回本地**（note-9 §5）；下次恢复时先 diff 落回再 amend。

## 4. 本地 CANN 容器

- 镜像：`quay.io/ascend/cann:9.1.0-310p-ubuntu22.04-py3.10-devel`（本机已 pull）。
  容器内 toolkit 路径 `/usr/local/Ascend/cann-9.1.0/`（note-5/note-6 的全部头文件取证来源）。
- 用途：**头文件精读**（ge/ge_api.h、es/es_*.h、exe_graph/runtime/tensor.h）、ops-info 配置
  取证（`/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/config/`，16 个 SoC 目录，
  note-10 F3）、以及在无真机时的**容器内编译**（语法/链接级验证）。
- 容器内构建命令（与远程同构，工具链路径换为容器内路径）：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cmake -B build -DGGML_CANNGE=ON -DGGML_CANN=OFF [-DSOC_TYPE=Ascend310P3]
cmake --build build --target ggml-cannge test-backend-ops -j$(nproc)
```

- 本机 x86_64 直接运行该镜像，**无需 `--platform`**（镜像内是 x86_64-linux 工具链、交叉编译
  目标 Ascend310P，toolkit 路径 `/usr/local/Ascend/cann-9.1.0/x86_64-linux/`）。实测命令
  （2026-09-10 双构建轮验证）：

```bash
cd /workspace/shengjiayi/Tsinghua/llama.cpp && docker run --rm -v "$PWD":/src -w /src \
  quay.io/ascend/cann:9.1.0-310p-ubuntu22.04-py3.10-devel bash -c '<命令>'
```

  容器内构建：`cmake -B build-cannge -DGGML_CANNGE=ON -DGGML_CANN=OFF -DSOC_TYPE=Ascend310P
  -DCMAKE_BUILD_TYPE=Release && cmake --build build-cannge --target ggml-cannge -j$(nproc)`。
- 容器联网需走外网代理 `http://127.0.0.1:63145`（用户口径）。

## 5. 双构建纪律

1. 任何 ggml-cannge 代码变更后，**必须**完成双侧构建，禁止只构建一边就宣告完成：
   - **容器构建**：ggml-cannge 目标编译链接通过（验证 CANN/ES 头与库链接）；
   - **本机构建**：`llama.cpp/build-native/` 原生构建通过（验证不依赖 CANN 的上游路径未被
     破坏，含 ggml 其余后端与 test-backend-ops）。
2. 本机 x86_64 环境无 CANN toolkit 时，build-native 配置中 `GGML_CANNGE=OFF`
   （当前 CMakeCache 实测如此）；容器构建覆盖 CANNGE 编译验证。两侧互补，缺一不可。
3. 构建产物属主注意：容器内以 root 构建会在挂载的工作区留下 root 属主文件，
   清理/重建前需注意 `chown` 还原（细节见文末待核实清单）。

## 6. 凭证总则

- 一切凭证（310 网关 token/密码、910 OAuth、任何账号口令）**必须**向用户索取，
  **禁止**写入笔记、脚本、git 历史或任何落盘文件；note-8 的"禁止把真实 token/密码写入
  本文档或任何落盘文件"为本条的上位依据。
- 服务器地址、路径、工具用法可以写入文档；凭证与网关命令模板中的占位符一律保留
  `<...>` 形式。

## 7. 待核实清单

（2026-09-11 全部清零：代理路径、set_env.sh 符号链接、docker run 参数、root 属主场景均
已确认，见 §3.2/§4/§5。）
