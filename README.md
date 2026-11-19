# RinkSpace

RinkSpace 从花样滑冰转播重建冰场、围挡、广告与时变相机的共同几何，并将选手的三维运动与原片曝光时间对齐。

`rebuild` 是公开默认分支和当前开发版。[安卡拉赛事](events/ankara_2026/README.md)涵盖四套共用冰场模型的短节目，提供场地、相机、源区间和本地回放登记。[冈田芽衣人体与鞋刀结果](events/ankara_2026/programmes/okada_mei_sp/README.md#人体与足鞋结果)覆盖完整处理区间。整片视觉验收、独立相机与尺度标定、人体空间位置和接冰验证仍待完成。

## 快速试用

使用 Python 3.11 或更新版本，在仓库根目录运行以下命令。示例使用合成轨迹，无需比赛素材、GPU 或原生后端。

~~~console
python -m pip install -e ".[dev]"
python -m rinkspace doctor
python -m rinkspace demo --out ../rinkspace-demo
python -m pytest tests/test_geometry.py tests/test_motion.py tests/test_mosaic.py tests/test_demo.py -q
~~~

打开输出目录中的 `index.html` 查看轨迹、速度、曲率和带来源索引的拼接图。数值摘要在 `summary.json`。

## 文档

| 内容 | 入口 |
|---|---|
| 当前赛事模型和节目数据 | [安卡拉 2026](events/ankara_2026/README.md) |
| 本地三维回放与原片回投 | [播放器](docs/RESULTS_3D_PLAYER_ZH.md) |
| 实现与开发 | [实现状态](docs/REBUILD.md)、[架构](docs/ARCHITECTURE.md)、[环境](environments/README.md)、[原生核心](native/README.md) |
| 数值方法与调用接口 | [求解](docs/SOLVER.md)、[接口](docs/INTERFACES.md)、[建模与接入](docs/rink_model/BUILD_SPEC.md) |
| 数据与验证 | [公开数据与离线审阅](docs/GITHUB_REVIEW.md)、[验证说明](docs/VALIDATION.md)、[独立 PTZ 检验工具](tools/ptz_verifier/README_zh.md) |
| 本机存储与维护 | [维护说明](MAINTENANCE.md) |

## 许可

项目自有内容及第三方资产的适用条款见[分项许可声明](LICENSE.md)和[许可核查](docs/LICENSING_REVIEW.md)。比赛视频、截图和叠加画面保存在本机。官方原片地址见[数据说明](docs/GITHUB_REVIEW.md#官方视频与源时间)。部分模型使用 Git LFS。
