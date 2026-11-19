# 金书贤安卡拉站短节目

<a id="当前完整取点修复版"></a>
当前 [`s14_boards_1118`](s14_boards_1118.json) 使用[赛事共同冰场](../../README.md#暂定最终版本)和本节目[相机参数](camera_s14_boards_1118.npz)。S14 使用远侧印刷边及近侧围挡重新约束相机。处理范围为 S01–S24、原片 00:32.300–03:20.720，共 8,422 个原生曝光。登记的 `full_review.video` 是本机 PotPlayer 回放地址，用户整片视觉验收待完成。

## 源与范围

| 内容 | 范围及入口 |
|---|---|
| 官方单节目 | 1920×1080、50 fps、原生时基 `1/12800`，首末 PTS 413440、2569216。[节目范围](rebuild_scope.json) |
| 官方全场 | 对应 37:22.567–40:11.100，10,113 个 60 fps 呈现帧。8,422 个单节目帧中 8,419 个有双向匹配，余下三帧保留邻近候选。[源选择及帧对应](primary_source_selection.json) |
| 处理边界 | 32.300 秒作为开场动作前的保护帧，200.720 秒为首次显示 2:48 的末帧。单节目源无音轨，音乐起点、亚秒停表时间与独立曝光节奏仍待核实 |
| 镜头和机位 | 23 个切点已按原片复核。2026-11-05 确认 24 镜头的机位分配，其中 S05＝主 0、S09＝副 1、S16＝主 0、S17＝副 1。[镜头及确认范围](rebuild_scope.json) |

## 历史索引

此前暂定版和密集观测版见[2026-11-17 登记](provisional_final_1117.json)及[密集观测登记](joint_dense_1118.json)。

较早的 `joint_review_1113` 保留[相机参数及复核证据](joint_review_1113.json)，其首版整片回放已退役。早期[冻结几何](geometry_frozen_1103.json)记录冈田芽依 `camera_solver_refined_1102` 的参考基线。[相机试验归档](camera_trial_archive_1113.json)登记 Data 地址和 SHA-256，[范围文件](rebuild_scope.json)保留取点、机位确认和历史节目页的来源。[联合拟合与稳定性复核源码](../../tools/joint_review_1113/README.md)

冻结几何的原 JSON 位于 Data 的 `runs/jin_sp_20261103/frozen_geometry/geometry_frozen_1103.json`，公开版本的本机临时路径已清理。两份文件的身份见[公开派生记录](../../../../configs/privacy_derivatives.json)。
