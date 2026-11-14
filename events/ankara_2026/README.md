# 安卡拉 2026：赛事入口

本目录保存本赛事的来源、数值及适用范围。模型注册或结构确认不等于准确性验收，也不能把这些数值直接用于其他冰场。

| 内容 | 入口 |
|---|---|
| 当前场景、广告实例、冰面取点定义与审核状态 | [scene/README.md](scene/README.md) |
| 节目原片、源时钟、镜头范围、结果与历史索引 | [金书贤短节目](programmes/jin_shuxian_sp/README.md)、[冈田芽依短节目](programmes/okada_mei_sp/README.md) |
| 2026-10-06 模型与人工事实登记 | [manifest.json](manifest.json) |
| 保全原件及外部依赖 | [首轮清单](asset_preservation_manifest.json)、[2026-10-13 增量清单](asset_preservation_manifest_20261013.json) |
| 用户确认的场地、广告与机位事实 | `model_constraints/`；实际适用性按记录的赛事、安装时期及版本读取 |
| 历史注册、工作模型和初始化 | `models/registered/`、`models/working/`、`models/initializations/`；保全副本不自动成为当前输入 |
| 冻结来源、人工事实及旧方法证据 | `observations/source_packets/`、节目 `source_datasets/`；[2026-10-13 来源保全](tools/strict_source_authorities/gray_review_20261013/README.md) |

## 暂定最终版本

2026-11-13 登记。两套完整回放使用同一份冰场几何，保留各自的节目相机和原生源时钟。独立标定及整片视觉验收仍待完成。

| 对象 | 原片范围 | 登记 |
|---|---|---|
| 金书贤短节目完整拟合 | S01–S24，32.300–200.720 秒，8,422 个原生帧 | [`stability_review_1113`](programmes/jin_shuxian_sp/provisional_final_1113.json) |
| 冈田芽依短节目完整拟合 | S01–S09，22.120–188.680 秒，8,329 个曝光 | [`joint_review_1113`](programmes/okada_mei_sp/fit_v11/provisional_final_1113.json) |
| 安卡拉冰场共同几何 | 两套节目共用 | [`shared_joint_review_1113`](scene/provisional_final_1113.json) |

## 历史与来源

历史源清单中的候选数、HELD 数及当时状态以各自 JSON 为准。 [联合拟合与稳定性复核源码](tools/joint_review_1113/README.md) · [当前实现](../../docs/REBUILD.md)

Data 文件使用已登记的精确地址和 [AssetStore 映射](../../MAINTENANCE.md)，不从本页拼接一个假定统一的媒体目录。保全清单与旧 receipt 供追查，不是正常求解前置条件。场地准入与验证分别见 [BUILD_SPEC](../../docs/rink_model/BUILD_SPEC.md) 和 [VALIDATION](../../docs/VALIDATION.md)。
