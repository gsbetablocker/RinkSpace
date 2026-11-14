# 安卡拉共同冰场与相机复核源码

[赛事登记](../../README.md#暂定最终版本)固定两套节目的相机与共同冰场。此处保存对应的项目源码，[来源清单](source_manifest.json)记录迁入前的 Data 路径及 SHA-256。已迁入的入口通过 `AssetStore` 定位本机 Data；实际运行读写 `Data/runs/ankara_joint_1113`。准备好的数据包含原始源 PTS、UV、角色和权重。

| 环节 | 入口 |
|---|---|
| 共同场景、原片来源复核和资格准备 | [旧基线绑定](inspect_geometry.py) · [观测准备](prepare_guarded.py) |
| 已准备数据的联合拟合、候选选择及配准 | [求解](solve_guarded.py) · [候选复核](select_review.py) · [登记](register_review.py) |
| 金书贤相机稳定性检查与修订 | [光学和运动检查](stability_1113/camera_audit.py) · [局部修订](stability_1113/refit_local.py) · [安装修订](stability_1113/refit_mount.py) · [候选选择](stability_1113/select_final.py) |

用已配置 Data 根的环境执行，依赖见[环境说明](../../../../environments/README.md)和[求解运行要求](../../../../docs/SOLVER.md#run-contract)。已准备数据的联合拟合和稳定性修订直接消费保存的数值输入。重新取点或重制历史检查画面时，还需要 Data 中 q 系列源脚本、观测、原片和匹配的原生库。
