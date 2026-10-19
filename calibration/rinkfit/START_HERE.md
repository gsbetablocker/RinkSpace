# 冈田芽衣土耳其站 SP 第一镜头：Codex 拟合启动包

**版本：初始架构 0.1.0 / 设计合同 v1.1。** 本包包含真实的整段审核输入、可运行的配准前端、物理模型原语、已编译验证的 Ceres 两阶段约束框架和本地 Codex 的完整接手指令。

**交给 Codex：** 解压后，让 Codex 在现有 RinkSpace 项目中读取 `CODEX_START.md` 并继续执行。该文件已经说明接入位置、启动顺序、现有代码能力、剩余实现工作、运行监控、完整状态恢复和最终验收条件。

这份包交付的是能运行的初始架构。`start` 可以直接开始真实整段 S1 初始化；最终全镜头物理求解需要 Codex 完成事件 `ProblemBuilder`、原始统计/语义接口及独立物理重放。当前没有附带已拟合成功的最终相机，也没有把合成后端示例当成本片求解器。

## 1. 已经包含什么

| 层 | 实际交付 |
|---|---|
| S0 输入 | 1248 帧、182017 条原始 UV、171 个材料标签；保留五个原始数组的值、dtype、顺序与 SHA-256 |
| S1 配准 | 原生像素坐标；规范化 DLT；有限线/圆弧/模型折线/Bézier；正权归一化 soft-min；多尺度；连续有限材料重放；实际残差秩；多种子和不可变候选 |
| S2 图 | 独立真实实体角点边；多锚双向传播；每个分量的显式规范；稀疏联合图精化；真实循环诊断；完整 1248 帧记录 |
| S3 原语 | 显式 `H_atlas Q ~ P M` 的共同投影矩阵初值；单平面退化和未知 Q 明确返回未升级 |
| 物理内核 | 共享固定双轴云台；随云台旋转的光心；低维 zoom/Brown/入口瞳；原生三次样条；RS 根和隐式导数；有限曲线及连续围挡表面 |
| 新状态初值 | 已将全部 1248 帧旧相机转换为新命名固定头/三次控制状态，附转换误差与明确未解决项 |
| 求解框架 | Ceres LM + SuiteSparse；外层 PHR 增广拉格朗日；Phase I 最小化 b、Phase II 固定 b=1；完整状态、原生重放合同、数值停止与续跑 |
| 接手材料 | 设计正文、机器合同、伪代码、API 文档、事件适配表、源数据映射、运行命令和验证报告 |

压缩包没有重复放入原视频、批量 PNG、历史求解缓存、旧运行目录或平台二进制。已审核的 UV 数据足以直接启动 S0/S1；最终叠加审片时再绑定本机原视频。

## 2. 一次性交给 Codex 的文字

> 请以此启动包的 `CODEX_START.md` 为本次任务入口，先读当前 RinkSpace 的 AGENTS.md，再按 `spec/design_contract.json` 和 `spec/SPEC.html` 接入已有项目。包中是真实的 1248 帧、182017 条已审核观测。直接完成列明的事件适配与必要验证，并启动整段单镜头的共享相机/冰场物理拟合；运行中持续报告原生误差、有效义务、物理约束、数值状态和完整 checkpoint。使用子代理并行处理独立模块。不要重新要求我审核点，不要缩成前 N 帧，不要把 S1、合成 demo 或程序退出当作拟合成功。完整保存所有原始义务和模型分支；在已有授权范围内持续推进。

## 3. 最小运行命令

优先使用现有项目的 Python 环境。若其 NumPy/SciPy 版本不满足 `requirements.txt`，在项目现有环境管理方式下补齐，不另复制一套大环境。

以下 PowerShell 命令在本包目录执行。运行数据目录应位于项目现有 Data 根目录：

```powershell
python -m pip install -r requirements.txt
python launch.py inspect
python launch.py prepare --output D:\CodexData\RinkSpace_Data\runs\okada_s1_r11
python launch.py start --output D:\CodexData\RinkSpace_Data\runs\okada_s1_r11 --workers 4
```

`start` 会先复用兼容的 S0 准备结果，再调度全部 3606 个局部平面任务。4 个 worker 是可调整起点；每个 worker 禁止默认再开一层 BLAS 线程池，环境变量显式设置可覆盖这一初值。S1 是 CPU 参考实现，完成耗时需要按本机吞吐测量；本包没有宣称全段实时。

后续命令：

```powershell
python launch.py status --run-dir D:\CodexData\RinkSpace_Data\runs\okada_s1_r11
python launch.py graph --run-dir D:\CodexData\RinkSpace_Data\runs\okada_s1_r11
python launch.py upgrade --run-dir D:\CodexData\RinkSpace_Data\runs\okada_s1_r11
python launch.py bootstrap --run-dir D:\CodexData\RinkSpace_Data\runs\okada_s1_r11
python launch.py export-backend --run-dir D:\CodexData\RinkSpace_Data\runs\okada_s1_r11
```

`graph` 可在 S1 尚未完成时运行，真实角点边能独立启动有支持的 projective 分量。`upgrade` 只将已有的、同帧满秩平面候选转成条件共同 P；没有所需证据的帧继续保留未升级状态。Codex 应利用并行时间完成事件适配，不必串行等待全部 S1 参考任务跑完才编写后端。

暂停 S1：`python launch.py stop --run-dir <目录>`。当前已提交的小批任务结束并保存后暂停。恢复时先 `clear-stop`，再 `register`；已提交完整 `result.json` 的任务不会重算。更改输入、模型版本或目标设置应建立新 run，不能混入旧 checkpoint。

Linux/WSL 可将输出改为 Data 对应的挂载目录；下面是已实际验证的 Ceres 构建方式：

```bash
cmake -S . -B /tmp/rinkfit_build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build /tmp/rinkfit_build -j 4
ctest --test-dir /tmp/rinkfit_build --output-on-failure
```

已验证依赖为 GCC 13.3、CMake 3.28.3、Ceres 2.2.0、Eigen 3.4、SuiteSparse 7.6。实际运行平台为 Linux；未声称已经验证本机 Windows 工具链。Windows 可复用项目已有 Ceres 安装并指定 `Ceres_DIR`，或由 Codex 使用既有 WSL 环境；构建输出也放在 Data 区域。

## 4. 当前结果与边界

完整输入准备和导出已通过。27 项 Python 数学/语义测试、3 项实际 C++/Ceres 测试通过。真实 `frame 1 / board_y_max` 的极短 CLI 诊断能生成完整候选及日志，局部最大误差约 17.80 px，尚未达到 2 px。真实角点图精化将其图因子的原生见证 RMS 从约 0.7985 px 降到 0.3531 px；这个数值是图一致性诊断，不能换算成全镜头物理精度。

当前 153 个材料、176237 条观测可进入条件平面前端；另外 18 个材料、5780 条观测需要曲面围挡/原始分支适配。它们全部保留在原始输入和后端导出中，最终仍必须逐条解释。

完整 grouped W、真实 profile 查询元数据和经证实的 run 顺序需要 Codex 从现有项目的原始 packet 读取。旧标量 sigma 已带入，不能代替完整 W。圆角物理内核目前支持 p=2；旧场景 p=1.9051596 会明确报告待接入，绝不会静默改为 2。`data/bootstrap_state.json` 已包含可加载的新相机状态；其共享低维 Brown 转换在默认半径 2 的保守认证中有 123 帧不足，亦明确写入待解决项。这些是初值转换限制，不是观测错误判定。所有具体余项见 `docs/event_integration.md` 和 `capabilities.json`。

## 5. 文件阅读顺序

1. `CODEX_START.md`：本次执行指令和完成定义。
2. `docs/event_integration.md`：真实事件适配的精确接口和工作顺序。
3. `spec/SPEC.html`、`spec/design_contract.json`：v1.1 完整设计和公式。
4. `docs/*_api.md`、`docs/solver_backend.md`：已实现代码的构造方法。
5. `docs/validation.md`、`package_manifest.json`：实测范围和包内文件校验。

本文件只说明本次交接，不新增项目永久规则，也不改动现有项目的 AGENTS.md。
