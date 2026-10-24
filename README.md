# RinkSpace

RinkSpace 研究如何从花样滑冰转播中恢复冰场、连续围挡、广告与相机的共同几何模型，再将选手运动连接到统一场地和原始视频时间轴。

项目处于研究验证阶段。安卡拉赛事与冈田芽衣节目是开发样本；现有九镜头恢复快照保留为失败状态，完整节目定位、人体空间位置及跨赛事能力仍未完成验收。数值收敛、平滑结果和测试通过均不能代替真实镜头验证。

2026-10-24 的重构已接入共同 native 几何、完整时变相机前向、观测合同和原生 PTS 回放。当前场地包含已确认的 M1–M5 几何关系，Ad1/Ad14、Ad2/Ad15、Ad3/Ad16 共享各自模板与尺寸，实例位置独立；Ad15、Ad16 仍未定位。具体尺寸和广告安装仍为待拟合初值。新 GPU LM、`solve` 和 `resume` 尚未实现，不能把结构图或 D3 回放称为新拟合。见[当前实现状态](docs/REBUILD.md)。

此前第1–6镜头的恢复已停止并转入[系统审核](docs/GITHUB_REVIEW.md#joint6-20261021)。历史 GPU 求解确实执行过，其源资格与候选照常保留；它与当前尚未完成的统一求解引擎分别登记。

## 快速试用：合成数据

使用自己的 Python 3.11 或更新环境，在仓库根目录运行：

~~~console
python -m pip install -e ".[dev]"
python -m rinkspace doctor
python -m rinkspace demo --out ../rinkspace-demo
~~~

打开输出目录中的 `index.html`，查看合成轨迹、速度/曲率和带来源索引的拼接图；`summary.json` 保存数值摘要。示例的已知圆周半径为 8 m、速度为 2.8 m/s。这只是已知几何和合成接触点的基础工具试跑，不证明真实节目恢复或人体垂直投影准确。

基础测试无需比赛素材、GPU 或原生求解后端：

~~~console
python -m pytest tests/test_geometry.py tests/test_motion.py tests/test_mosaic.py tests/test_demo.py -q
~~~

完整测试还包括高级数值、原生后端及本机历史回放，不属于基础安装范围。按能力选择依赖见[环境说明](environments/README.md)。

## 阅读与数据审阅

| 想了解什么 | 入口 |
|---|---|
| 代码分工、赛事与节目关系 | [架构与接口](docs/ARCHITECTURE.md)、[算法](docs/ALGORITHMS.md) |
| 如何安装所需能力 | [环境说明](environments/README.md) |
| 公开数据和失败快照怎样读取 | [离线审阅](docs/GITHUB_REVIEW.md)、[节目快照](events/ankara_2026/programmes/okada_mei_sp/review/README.md) |
| 准确性、源时间和尺度怎样判定 | [验证说明](docs/VALIDATION.md) |
| 模块 API 与后端 | [rinkfit](calibration/rinkfit/README.md)、[solver_runtime](calibration/solver_runtime/README.md) |
| 重构后的原生接口、场地与广告模型 | [实现状态](docs/REBUILD.md)、[接口合同](docs/INTERFACES.md)、[当前场景](events/ankara_2026/scene/field/model.json) |

下载完整仓库后，使用 Python 标准库检查审阅归档：

~~~console
python -B scripts/read_offline_review.py --check
~~~

该命令只检查归档目录、成员与大小，不执行求解。公开仓库保留代码、模型、参数和必要数值观测；不分发 ISU 比赛视频、截图或叠加画面。需要像素核对时，按[官方来源及时间戳](docs/GITHUB_REVIEW.md#官方视频与源时间)自行取得原素材，使用原生 PTS 重建线索。

## 许可

本项目采用[分项许可声明](LICENSE.md)，目前未额外授予开源或商用许可；第三方内容适用各自原许可。权利范围、模型来源与缺口见[许可核查](docs/LICENSING_REVIEW.md)。部分神经权重使用 Git LFS。

开发命令、本机工作台和公开检查见[维护说明](MAINTENANCE.md)。
