# RinkSpace

RinkSpace 研究如何从花样滑冰转播中恢复冰场、连续围挡、广告与相机的共同几何模型，再将选手运动连接到统一场地和原始视频时间轴。

RinkSpace 处于研究验证阶段。`rebuild` 是当前开发分支和公开默认分支，[`main`](https://github.com/gsbetablocker/RinkSpace/tree/main) 保存 2026-10-24 的旧版。安卡拉的冈田芽依和金书贤短节目是实际开发样本，独立相机物理标定、人体空间位置和跨赛事能力仍待验证。

安卡拉两套短节目已在同一冰场几何上完成拟合和整片回放，并于 2026-11-13 登记为[暂定最终版本](events/ankara_2026/README.md#暂定最终版本)。整片视觉验收仍待完成。原生核心支持共同几何、时变相机、输入准备和连续 GPU LM，接口及能力边界见[实现状态](docs/REBUILD.md)。

两套节目的源时间、镜头划分和回放入口见[赛事索引](events/ankara_2026/README.md)。

冈田芽依的[人体与鞋刀候选](events/ankara_2026/programmes/okada_mei_sp/README.md#人体与足鞋结果)已覆盖完整处理区间。可从[播放器说明](docs/RESULTS_3D_PLAYER_ZH.md)查找最新全片候选和 S01 对照。人体空间位置和真实接冰仍待独立验收。

## 快速试用：合成数据

使用 Python 3.11 或更新环境，在仓库根目录运行：

~~~console
python -m pip install -e ".[dev]"
python -m rinkspace doctor
python -m rinkspace demo --out ../rinkspace-demo
~~~

打开输出目录中的 `index.html`，查看合成轨迹、速度／曲率和带来源索引的拼接图；`summary.json` 保存数值摘要。示例圆周半径为 8 m、速度为 2.8 m/s，用于检查基础工具，不证明真实节目恢复或人体垂直投影准确。

基础测试无需比赛素材、GPU 或原生后端：

~~~console
python -m pytest tests/test_geometry.py tests/test_motion.py tests/test_mosaic.py tests/test_demo.py -q
~~~

完整测试还涉及高级数值、原生后端和本机历史回放，依赖见[环境说明](environments/README.md)。

## 阅读入口

| 内容 | 入口 |
|---|---|
| 本地三维结果及回放 | [播放器说明](docs/RESULTS_3D_PLAYER_ZH.md) |
| 代码结构、算法与接口 | [架构](docs/ARCHITECTURE.md)、[求解方法](docs/SOLVER.md)、[接口合同](docs/INTERFACES.md) |
| 已实现能力与缺口 | [实现状态](docs/REBUILD.md)、[原生核心](native/README.md) |
| 当前赛事模型与节目结果 | [安卡拉赛事](events/ankara_2026/README.md) · [冈田芽依短节目](events/ankara_2026/programmes/okada_mei_sp/README.md) · [金书贤短节目](events/ankara_2026/programmes/jin_shuxian_sp/README.md) |
| 数据来源、公开范围与失败快照 | [离线审阅](docs/GITHUB_REVIEW.md)、[冻结节目快照](events/ankara_2026/programmes/okada_mei_sp/review/README.md) |
| 准确性、源时间与尺度 | [验证说明](docs/VALIDATION.md) |
| 历史后端 | [rinkfit](calibration/rinkfit/README.md)、[solver_runtime](calibration/solver_runtime/README.md) |
| 独立 PTZ 图像检验 | [工具说明](tools/ptz_verifier/README_zh.md)；历史试跑结论为证据不足，未认证相机物理参数 |
| 开发工具、存储与工作要求 | [维护说明](MAINTENANCE.md#work-entry) |

下载完整仓库后，可用 Python 标准库检查冻结审阅归档：

~~~console
python -B scripts/read_offline_review.py --check
~~~

该命令检查归档目录、成员与大小，不执行求解。公开树保留代码、模型、参数和必要数值观测，不分发 ISU 比赛视频、截图或叠加画面。原像素核对需要按[官方来源与源时间](docs/GITHUB_REVIEW.md#官方视频与源时间)取得对应素材。冻结九镜头失败快照和历史六镜头试验按各自的源区间与版本解释。

## 许可

项目采用[分项许可声明](LICENSE.md)，目前未额外授予开源或商用许可；第三方内容适用各自原许可。来源与缺口见[许可核查](docs/LICENSING_REVIEW.md)。部分神经权重使用 Git LFS。
