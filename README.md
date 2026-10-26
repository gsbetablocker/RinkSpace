# RinkSpace

RinkSpace 研究如何从花样滑冰转播中恢复冰场、连续围挡、广告与相机的共同几何模型，再将选手运动连接到统一场地和原始视频时间轴。

项目处于研究验证阶段。安卡拉赛事与冈田芽衣节目是开发样本；现有九镜头恢复快照保留为失败状态，完整节目定位、人体空间位置及跨赛事能力仍未完成验收。数值收敛、平滑结果和测试通过均不能代替真实镜头验证。

截至 2026-10-27，重构已接入共同 native 几何、完整时变相机、观测合同与原生 PTS 回放；GPU 已支持相机／共享场景／曲线位置的联合连续 LM、安全停止及同进程继续。S01 已接入七字段原始点和完整广告版式，仍有真实冰面偏移及相机可辨识性问题；跨进程完整断点恢复、S02–S09 完整接入和自动取点尚未交付。可调用范围见[实现状态](docs/REBUILD.md)，当前结构、广告库存、未定位项及待审状态见[场景索引](events/ankara_2026/scene/README.md)。结构确认和 D3 回放不等于拟合通过，尺寸与安装仍需相应证据验证。

此前第1–6镜头的恢复已停止并转入[系统审核](docs/GITHUB_REVIEW.md#joint6-20261021)。历史 GPU 求解确实执行过，其源资格与候选照常保留；它与当前尚未完成的统一求解引擎分别登记。

**rebuild 的冈田芽衣短节目统一起点：原片 22.120 秒，PTS 283136，原生时基 1/12800。** 拟合、网格交付和节目分析从 demo 的这个计时起点开始，扩展至全节目九镜头时仍沿用；此前画面和观测保留归档，不进入拟合。权威范围见[节目范围文件](events/ankara_2026/programmes/okada_mei_sp/rebuild_scope.json)及[节目入口](events/ankara_2026/programmes/okada_mei_sp/README.md#analysis-start)。

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
| 长期规则、技术要求与具体任务分别在哪里 | [工作入口与要求归属](MAINTENANCE.md#work-entry) |
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
