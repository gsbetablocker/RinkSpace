# 重构实现状态

状态核对日期：2026-10-26。本页只记录现有能力和缺口；精确调用见 [INTERFACES](INTERFACES.md)，模块设计见 [ARCHITECTURE](ARCHITECTURE.md)，算法目标见 [SOLVER](SOLVER.md)。历史阶段名称不决定后续执行顺序。

| 范围 | 已实现能力与限制 |
|---|---|
| native ABI 2 | CPU 共同几何、固定节点、连续围挡、静态／完整时变相机和预测行隐式 RS；可选 CUDA 共用当前几何／投影完成原 UV 残差、相机／station 导数、相关白化、连通消元和连续 LM；安全停止及同一 handle 继续已实现，磁盘完整续算未交付 |
| 完整状态导入 | `model import-state` 保留冻结镜头的完整相机与实体；`import-d78` 仅生成未拟合结构初值；缺镜头状态不补造 |
| prepare | 当前合同、观测、相关组、源 PTS 和相机定义域检查；自包含 scene、camera/controls 及共同参数快照；当前仍校验所读数组，增量缓存未实现 |
| review | 同一 native、同源 PTS、全模型和可选同源对照；读取共同候选参数；不优化 |
| 固定交点 | `junction_relation="ice_board_bottom"` 已接通合法 z=0 跨表面交点；普通交点仍受同表面限制，不由数学交点新增 UV |
| 广告 D1／D3 | 来源适配、模板预览、展开面／连续围挡／原曝光展示；`--frame` 为未核实图片预览，`--source-video` 配合 scope 可核实原曝光 |
| 广告内部状态 | 共享尺寸／内部组、独立安装、字形及解析方形／矩形，经同一 native 求值并提供几何导数；不等于完整相机因子 Jacobian |
| 显式局部参数图 | prepare/SceneParameterModel 可消费 local_charts，保存并重载 v2 局部布局、FactorPlan 和已绑定 station；调用及列映射见 INTERFACES，此处不表示几何参数已参与 GPU 优化；当前 GPU 相机／station 因子见下文 |
| 相机快照 | `snapshot_exposure` 与 `snapshot_window` 保留覆盖读出域的原控制点并用同一 native 求值；对应 R1／R2 记录见节目入口 |
| 原片连续取点 | `observe read` 接回既有训练权重、RAFT／ConvGRU、线／圈材料读数与实际反馈消费；围挡按正面／顶面视角区分，顶面外侧不得进入广告。已有同源 UV 保留，几何语义来自当前 native。**完整能力未恢复**：新材料候选尚未进入正式读数和下一帧引导，证据集中于同一 10 帧；S01 中／末段独立启动有候选而正式冰面读数为 0。该工作按用户要求暂停，详见节目 source_reader_audit |
| G1 GPU 残差 | 不可变 native resident problem 已实际执行；通过 NativeSceneGPU 支持当前冰面、连续围挡、固定点和已定位广告＋完整时变相机的原 UV 残差；不做白化、Jacobian 或参数更新 |
| G2 GPU 线性化 | 同一 G1 handle 内计算局部 Jacobian、完整 W、按组秩归一的 Huber IRLS 与连通局部块；正常每次回传 32 字节摘要。真实窗口活动为 pan／tilt／log(f) 三个常量修正及原 station，几何等冻结；不消元、不更新参数 |
| G3 GPU 一次候选 | 同一 handle 原生控制阻尼、cuSOLVER 连通局部消元、cuDSS SPD 求解、回代、有限 station 可行步和原目标接受／拒绝；G3 的一次真实候选已接受，拒绝不改 accepted state；后续真实连续尝试的失败／停止与工程验证分别记于下文 |
| 连续 GPU 求解 | NativeSceneGPU／ResidentResidualProblem.solve 接入全部当前可定位路径；C++ 连续迭代、数值停止、安全停止和同一 handle 继续；当前活动量为显式相机控制方向及 station，模型参数保持冻结。取点数据集适配与磁盘完整 checkpoint/resume 尚未交付 |

`core/native.py` 是薄绑定。当前实体图／回放仍使用 `compat.d78_entities` 的静态采样辅助，这是尚未退出的兼容依赖，不会重跑历史数据生产。广告导入不授予拟合资格；来源核实、几何前向成功和画面验收分别判断。

## 场景与实证记录

当前安卡拉场景、模板版本、结构确认和未拟合状态由[场景索引](../events/ankara_2026/scene/README.md)及其模型引用维护。完整镜头范围、官方源时钟、R1 单曝光与 R2 短窗口对照由[节目入口](../events/ankara_2026/programmes/okada_mei_sp/README.md)维护。旧三对广告审计基准、当前库存和历史六镜头结果分别保留身份，不互相替代。

R1／R2 验证的是相机读取与前向等价；未完成新场景拟合或完整 S01 准确性验收。现有共同场地、广告尺寸／安装仍有未拟合偏差，工程测试通过不能改变该状态。

## 构建入口

在项目根的已有环境执行；源码未安装时仅为当前进程设置 `PYTHONPATH`：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
python scripts/build_core.py --out D:\CodexData\RinkSpace_Data\build\core
```

构建输出 `build.log/build.json`，后者的 `library` 给出实际库路径，默认 CPU 构建标记为 `cuda_built=false`；显式 `--cuda` 使用已有 CUDA 工具链构建 G1／G2，标记为 `cuda_built=true`。显式 `--cudss <本项目已有库根>` 再启用原生 LM，记为 `lm_candidate_implemented=true`、`nonlinear_solver_implemented=true`；`persistent_resume_implemented=false`。Windows/MSVC 路径已有使用记录；脚本不安装工具链或启动求解。原生检查见 [native/README](../native/README.md)，其余命令参数集中在 INTERFACES。

字形重建的 SciPy、实体图辅助的 Matplotlib 属于 `advanced` 可选依赖，不是一般 native 几何前向的必要依赖。公开数值资产与本机原素材的边界见 [GITHUB_REVIEW](GITHUB_REVIEW.md)。

R3 已用 R2 同一完整相机与原生 PTS 307200、320000、332800（24、25、26 秒）比较冻结 d78 与当前共同几何中的 R3/R4 靶圈，共六条既有材料轮廓。原片 RGB 与 R2 同曝光逐字节一致；同一 native 核仅替换几何，并核对共列、配对同行、四行等距、共同半径及上下外圈等边距。25 秒当前 R3 轮廓相对旧轮廓向左上方移动，仍有可见未拟合偏差；不据此宣称像素改善。对照图与全部 624 条当前路径背景图位于 `D:/CodexData/RinkSpace_Data/runs/r3_20261025/`，绑定及数值记录见 `events/ankara_2026/programmes/okada_mei_sp/r3.json`。未改相机、几何参数或观测，未启动拟合；本次是三个曝光的结构变化展示，不是整镜头验收。

G1 已用 R2 同一 24.00–26.00 秒窗口的 101 个原生曝光、R3/R4 六条轮廓、5,920 条原 UV，在 GPU 内生成几何并求解完整 RS、计算残差。当前 CPU 对原 DLL 投影差为 0，CPU/GPU 最大差约 1.14×10⁻¹² px；冻结 station 残差中位数 53.23 px，仍未拟合。原片 101 帧对照、冷热耗时、内存、完整相关组引用和不支持清单见节目 [g1.json](../events/ankara_2026/programmes/okada_mei_sp/g1.json) 与其 Data 产物；81 项相关测试通过，不改变原有人工确认。

G2 沿用 G1 同一 101 帧、5,920 条观测，保留 197 个完整相关组和 73 个独立观测，形成 270 个局部连通分量；白化维数 1,300，最大局部块有 46 个 station。CPU／GPU 块、原 W 独立乘法及旧 CPU DLL 的小样本数值导数检查通过。三个活动方向的条件诊断及 192 个秩亏局部块均记录；这不是可直接无阻尼求逆的系统。GPU 热运行中位数 0.8441 ms、CPU 11.8695 ms，G1＋G2 自有 GPU 缓冲 4,094,992 字节；范围与测试在节目 [g2.json](../events/ankara_2026/programmes/okada_mei_sp/g2.json)。同源小扰动只作显示，几何、原 UV、PTS、相关组和原相机状态未改，没有优化或改善声明。

G3 已对同一窗口执行且只执行一个真实 LM 候选。270 个连通局部块先加阻尼，cuSOLVER 分解后由 cuDSS 求 3×3 保留系统；16 个边界 station 固定，内部边界活动集求解两次，符号分析一次。实际总目标 49,565.4356 → 19,348.2023，增益比 1.39838，因此数值接受；原像素残差中位数 53.23 → 33.86 px，仍有明显偏差；75 帧中位残差下降，25.50–26.00 秒的后 26 帧上升，数值接受不表示逐帧质量通过。当前 native 几何和原 UV／PTS／W 未改，新相机与 station 单独保存在节目 `g3/`。首次 native 闭环 45.1522 ms，分项及 9.93 MiB 自有 GPU 缓冲见 [g3.json](../events/ankara_2026/programmes/okada_mei_sp/g3.json)。报告序列化曾因范围字段位置错误失败，随后只复用保存候选完成 101 帧对照；首次准备及进程峰值元数据未保存，不补造。121 项回归、6 个子检查通过；没有开展第二次真实候选或改变人工验收。

当前 GPU 共接入 624 条可定位 native 路径（64 冰面、2 围挡、558 广告），Ad15／Ad16 保持模型中原有的未定位状态。`tests/test_gpu_geometry.py` 将全部路径与改动前 native 前向比较，并在内存工程输入中以 24 个相机控制坐标完成连续求解；同时验证非零合法广告内部参数、完整相关 W、秩亏重试、有限 station 边界、停止／继续与输入数组保持不变。103 项相关测试及 native smoke 通过；这些是数值实现证据，不是实际节目拟合验收。模型参数和取点数据集未因此改写，跨进程 checkpoint/resume 未实现。

此前 G4 沿用六条靶圈的真实连续尝试，先发生局部分解失败，随后按用户要求停止，没有最终收敛状态。失败及停止日志保留在 Data 的 `runs/g4_20261025/`，不能将这次尝试或后来全几何工程测试写成真实窗口已完成。当前交付入口为 [NativeSceneGPU](../src/rinkspace/session/gpu_geometry.py) 和 [ResidentResidualProblem.solve](../src/rinkspace/core/gpu.py)。
