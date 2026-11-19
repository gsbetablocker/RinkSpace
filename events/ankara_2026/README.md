# 安卡拉 2026：赛事入口

本赛事登记四套短节目的共同冰场、相机和源时间。取点定义、场景参数及名义模板见[场景入口](scene/README.md)。

<a id="暂定最终版本"></a>
<a id="当前四节目完整取点修复"></a>
## 四节目共同几何与完整回放

[当前指定记录](scene/current_fit.json)指向冻结的联合登记 `s14_boards_1118`；其共同冰场数值是 `lane4_side_1118`。以下为原片首末帧（含两端），均为 50 fps：

| 本次暂定最终登记 | 镜头 | 原片首末帧 | 曝光数 |
|---|---|---|---:|
| [冈田芽衣](programmes/okada_mei_sp/s14_boards_1118.json) | S01–S09 | 00:22.120–03:08.680 | 8,329 |
| [金书贤](programmes/jin_shuxian_sp/s14_boards_1118.json) | S01–S24 | 00:32.300–03:20.720 | 8,422 |
| [Chen Xi Jin](programmes/chen_xi_jin_sp/s14_boards_1118.json) | S01–S20 | 00:27.040–03:18.120 | 8,555 |
| [Nayeon Ko](programmes/nayeon_ko_sp/s14_boards_1118.json) | S02–S10 | 00:20.800–03:08.420 | 8,382 |

本版工程复核覆盖 62 个镜头、33,688 个源曝光。各节目登记中的 `full_review.video` 是本机 PotPlayer 回放地址，回放从对应原片首帧起算。用户整片视觉验收待完成。[版本说明](scene/s14_boards_1118.md) · [冈田芽衣人体与鞋刀](programmes/okada_mei_sp/README.md#人体与足鞋结果)

## 来源与历史

- [赛事清单](manifest.json)、[机位与几何约束](model_constraints/)和[源观测包](observations/source_packets/)记录来源及人工确认范围。
- 前次暂定版见 [2026-11-17](scene/provisional_final_1117.json)与 [2026-11-13](scene/provisional_final_1113.json)。密集观测与稀疏失败对照见 [joint_dense_1118](scene/joint_dense_1118.json)和 [joint_free_1117](scene/joint_free_1117.json)。
- 完整报告及运行证据可按模型登记和[维护说明](../../MAINTENANCE.md)追查。本赛事结果按[验证说明](../../docs/VALIDATION.md)解释。
