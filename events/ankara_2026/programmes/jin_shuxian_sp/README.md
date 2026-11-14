# 金书贤安卡拉站短节目

## 当前完整拟合

2026-11-13 登记的 `stability_review_1113` 使用[安卡拉共同冰场](../../scene/provisional_final_1113.json)，覆盖 S01–S24、原片 00:32.300–03:20.720，共 8,422 个原生帧。整片回放已导出。整片视觉验收、独立相机物理标定和米制验证仍待完成。

[版本登记及本地回放地址](provisional_final_1113.json) · [相机参数](stability_review_1113_camera.npz) · [稳定性复核与源观测](stability_review_1113.json)

S04 原片 45.300 秒后数帧仍有小幅抖动，按用户决定暂缓修复。

## 源与范围

| 内容 | 范围及入口 |
|---|---|
| 官方单节目 | 1920×1080、50 fps、原生时基 `1/12800`，首末 PTS 413440、2569216。[节目范围](rebuild_scope.json) |
| 官方全场 | 对应 37:22.567–40:11.100，10,113 个 60 fps 呈现帧。8,422 个单节目帧中 8,419 个有双向匹配，余下三帧保留邻近候选。[源选择及帧对应](primary_source_selection.json) |
| 处理边界 | 32.300 秒作为开场动作前的保护帧，200.720 秒为首次显示 2:48 的末帧。单节目源无音轨，音乐起点、亚秒停表时间与独立曝光节奏仍待核实 |
| 镜头和机位 | 23 个切点已按原片复核。2026-11-05 确认 24 镜头的机位分配，其中 S05＝主 0、S09＝副 1、S16＝主 0、S17＝副 1。[镜头及确认范围](rebuild_scope.json) |

## 历史索引

较早的 `joint_review_1113` 保留[相机参数及复核证据](joint_review_1113.json)，其首版整片回放已退役。早期[冻结几何](geometry_frozen_1103.json)记录冈田芽依 `camera_solver_refined_1102` 的参考基线。[相机试验归档](camera_trial_archive_1113.json)登记 Data 地址和 SHA-256，[范围文件](rebuild_scope.json)保留取点、机位确认和历史节目页的来源。[联合拟合与稳定性复核源码](../../tools/joint_review_1113/README.md)

冻结几何的原 JSON 位于 Data 的 `runs/jin_sp_20261103/frozen_geometry/geometry_frozen_1103.json`，公开版本的本机临时路径已清理。两份文件的身份见[公开派生记录](../../../../configs/privacy_derivatives.json)。
