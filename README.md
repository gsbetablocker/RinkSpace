# RinkSpace

RinkSpace 研究如何从花样滑冰转播中恢复冰场、连续围挡、广告与相机的共同几何模型，再将选手运动连接到统一场地和原始视频时间轴。

项目处于研究验证阶段。当前开发分支为 `rebuild`；本页描述该分支的工作区能力，公开默认分支和冻结审阅包各有自己的版本范围。安卡拉赛事与冈田芽衣节目是开发样本，完整节目定位、相机物理参数、人体空间位置及跨赛事能力尚未完成验证。

当前原生核心支持 CPU／GPU 共用几何、完整时变相机、共享场景与曲线位置的联合连续 LM，可安全停止并在同一进程继续。真实样本已接入 S01–S09 九镜头、三个物理机位，完成真实联合拟合与完整候选回放；当前候选未数值收敛，完整画面、源特征覆盖与相机物理参数仍待验证。已有人工网格反馈只对应各自快照，范围与当前结果见[节目入口](events/ankara_2026/programmes/okada_mei_sp/README.md)。完整磁盘断点恢复和全节目自动取点尚未交付，能力与缺口见[实现状态](docs/REBUILD.md)。

**冈田芽衣短节目的重构处理起点为原片 22.120 秒，PTS 283136，原生时基 1/12800。** 当前处理范围与历史物理镜头范围的差别见[范围文件](events/ankara_2026/programmes/okada_mei_sp/rebuild_scope.json)和[源时间说明](events/ankara_2026/programmes/okada_mei_sp/README.md#analysis-start)解释。起点前的原片与观测保留，不进入当前拟合。

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
| 代码结构、算法与接口 | [架构](docs/ARCHITECTURE.md)、[求解方法](docs/SOLVER.md)、[接口合同](docs/INTERFACES.md) |
| 已实现能力与缺口 | [实现状态](docs/REBUILD.md)、[原生核心](native/README.md) |
| 当前赛事模型与节目输入 | [安卡拉场景](events/ankara_2026/scene/README.md)、[冈田芽衣短节目](events/ankara_2026/programmes/okada_mei_sp/README.md) |
| 数据来源、公开范围与失败快照 | [离线审阅](docs/GITHUB_REVIEW.md)、[冻结节目快照](events/ankara_2026/programmes/okada_mei_sp/review/README.md) |
| 准确性、源时间与尺度 | [验证说明](docs/VALIDATION.md) |
| 历史后端 | [rinkfit](calibration/rinkfit/README.md)、[solver_runtime](calibration/solver_runtime/README.md) |
| 独立 PTZ 图像检验 | [工具说明](tools/ptz_verifier/README_zh.md)；历史试跑结论为证据不足，未认证相机物理参数 |
| 开发工具、存储与工作要求 | [维护说明](MAINTENANCE.md#work-entry) |

下载完整仓库后，可用 Python 标准库检查冻结审阅归档：

~~~console
python -B scripts/read_offline_review.py --check
~~~

该命令检查归档目录、成员与大小，不执行求解。公开树保留代码、模型、参数和必要数值观测，不分发 ISU 比赛视频、截图或叠加画面。原像素核对需要按[官方来源与源时间](docs/GITHUB_REVIEW.md#官方视频与源时间)取得对应素材。冻结九镜头失败快照与历史六镜头停止状态按各自输入和版本解释，与当前九镜头候选分别登记。

## 许可

项目采用[分项许可声明](LICENSE.md)，目前未额外授予开源或商用许可；第三方内容适用各自原许可。来源与缺口见[许可核查](docs/LICENSING_REVIEW.md)。部分神经权重使用 Git LFS。
