# 冈田芽依短节目

## 当前完整拟合：暂定最终状态

2026-11-13 登记的 `joint_review_1113` 使用[安卡拉共同冰场](../../scene/provisional_final_1113.json)，覆盖 S01–S09、原片 00:22.120–03:08.680，共 8,329 个原生曝光。整片回放已导出，整片视觉验收、独立相机物理标定与米制验证仍待完成。

[版本登记及本地回放地址](fit_v11/provisional_final_1113.json) · [相机参数](fit_v11/joint_review_1113_camera.npz) · [源观测、误差及原生库绑定](fit_v11/joint_review_1113.json)

## 源与范围

| 内容 | 权威记录 |
|---|---|
| 单节目原片、SHA、下载表征与全场源对应 | [源选择](primary_source_selection.json)，1920×1080、50 fps、原生时基 `1/12800` |
| 处理范围和物理切镜 | [节目范围](rebuild_scope.json)，S01–S09，首末 PTS 283136、2415104 |
| 共同冰场、广告与取点定义 | [场景索引](../../scene/README.md) |
| 三台物理机位的型号与安装事实 | [人工确认](../../model_constraints/camera_hardware/user_camera_mount_and_height_20261006_v4.json) · [安装配置](../../../../configs/camera_hardware/okada_sp_installations_z190_v1.json) |

<a id="analysis-start"></a>
### 统一节目起点

拟合、网格交付和节目分析从原片 **22.120 秒，PTS 283136** 开始，与 [demo 计时零点](outro_stats.json)一致。19.960 秒准备画面用于开场姿势与定位辅助。S01 处理段为 22.120–44.960 秒、1,143 个曝光，物理镜头从 20.020 秒开始。完整 S05 为 74.140–101.640 秒、1,376 个曝光，其中 76.000–95.000 秒、951 个曝光是历史旋转检查窗口。节目末帧为 188.680 秒，排他终点为 188.700 秒。

## 人体与足鞋结果

全节目候选均覆盖上述 8,329 个曝光。人体空间位置、真实接冰和条件米制仍待独立验收。本地播放器、比较入口和历史依赖见[使用说明](../../../../docs/RESULTS_3D_PLAYER_ZH.md)。

[人体拟合和复核源码](tools/body_review_1113/README.md)

| 结果 | 状态与入口 |
|---|---|
| `okada_first_spin_1113` | 当前首旋修订，以登记的 `joint_review_1113` 相机和共同冰场为准，改动窗口为原片 81.140–93.320 秒。局部回投和自由鞋方向仍有偏差。[运动控制](fit_v11/body_first_spin_1113_controls.npz) · [复核与本地回放](fit_v11/body_first_spin_1113.json) |
| `okada_registered_cut_continuity_1113` | 首旋修订的前一候选，以相同相机检查八处切镜。部分切口的鞋刀方向与深度仍不稳定。[运动控制](fit_v11/body_cut_continuity_1113_controls.npz) · [复核与本地回放](fit_v11/body_cut_continuity_1113.json) |
| S01 `skate_s01_1103` | 保留用户目视较满意的 S01 对照，原片 22.120–44.960 秒。鞋刀图像误差及接冰偏差仍在。[结果绑定](skate_s01_1103.json) · [S01 播放器](../../../../launchers/START_3D_COMPARISON.cmd) |

## 有界结果与人工确认

| 对象 | 确认范围 |
|---|---|
| `joint_review_1113` | 2026-11-13 指定为暂定最终拟合，整片视觉验收仍待完成。[登记](fit_v11/provisional_final_1113.json) |
| 历史九镜头 `joint_mount_flex` | 2026-11-01 用户认为网格贴合基本满意，局部仍有偏差。反馈绑定[当时的完整状态](fit_v11/camera_review_mount_flex.json) |
| 六镜头 Lane 基线 | 用户认为该冻结版本的网格贴合基本满意。该状态为 927,425 点、4,463 个曝光，冻结状态 SHA-256 为 `54234d25cbf3e8d03dc3d52fdef5d367aa87ce59889cccd904ad1303ac175473`。[参数](native_lane_source_parameters.npz) |
| S01／S04 等轴焦距基线 | 用户认可这两镜头的网格贴合，冻结状态 SHA-256 为 `c7bf576dc92a635fb254633c8fb1da1ce02f309cc42e2dba2d11fd72bf7c00c4`。对应的居中主点实验在同一 Data 运行的 `camera_models/center_pp_square/result.json` |
| 单曝光材料对照 | [Ad5 矩形](ad5_rectangle_review.json)限 S05 原片 99.520 秒，用户看不出新旧变化。[Ad2 O 边界](ad2_o_review.json)限原片 25.000 秒，保留原 UV，未纳入拟合 |

各项人工意见按其对应的画面、源曝光和保存状态解释。

<a id="s01s04-当前同机位联合输入"></a>
<a id="legacy-maintenance"></a>
<a id="来源与历史导航"></a>
## 历史索引

<a id="2026-11-02-历史版本"></a>
2026-11-02 的 `camera_solver_refined_1102` 保留为历史暂定最终拟合。该版本在代价进展停滞时结束，局部运动偏差及光学、读出限制见[历史绑定](fit_v11/camera_solver_refined_1102.json)和[历史复核](fit_v11/camera_solver_refined_1102_review.json)。

<a id="whole-programme-body"></a>
<a id="whole-programme-actions"></a>
<a id="structure-camera-1108"></a>
<a id="lsp-s09-source-repair-1109"></a>
<a id="skate-evidence-1109"></a>
<a id="人体整合与根轨迹对照2026-11-04未验收"></a>
<a id="s05-与动作中间结果导航"></a>
### 历史人体候选导航

| 历史对照 | 结果绑定 |
|---|---|
| 1106／1107 轨迹与动作 | [1106 轨迹](programme_trajectory_1106.json) · [1107 轨迹与尺度／深度](programme_trajectory_1107.json) · [动作修订](programme_action_finish_1107.json) |
| 1108／1109 结构、鞋刀与全片复核 | [结构及相机](programme_structure_camera_1108.json) · [鞋刀证据](programme_skate_evidence_1109.json) · [九镜头复核](programme_full_review_1109.json) · [独立双鞋失败记录](independent_shoes_review_1109.json) |
| 1112 整鞋与 1113 开场方向实验 | [1112 绑定](skate_video_1112/record.json) · [1113 开场及方向](skate_video_1112/audit_followup_1113.json) · [实验方法](../../../../experiments/skate_video_v1/README.md) |
| 1113 整鞋复核 | [运动控制](skate_video_1112/whole_skate_pro_1113_controls.npz) · [来源、状态与偏差](skate_video_1112/whole_skate_pro_review_1113.json) |
| S05 源轮廓和条件支撑 | [源结构](spin_source_structure_1106_audit.json) · [软支撑对照](spin_support_correction_1106_audit.json) |
| S01、开头与足鞋 | [S01 全身](body_fullbody_1102.json) · [开头时序](opening_timing_1103.json) · [开头动态失败](opening_motion_1103.json) · [足鞋穿戴](boot_integration_1102.json) · [参数字段与身份](body_parameters_1103_fields.json) |

<a id="原生工程对照"></a>
[原生工程与来源源码](source_datasets/feature_refinement_1101/source_code.json) · [原片取点检查](source_reader_audit.json) · [历史六镜头试验](fit_joint6/status.json) · [历史九镜头失败快照](review/README.md)

逐轮诊断、撤回记录、失败原件与人工意见可按各自的结果绑定追查。早期节目页与报告原文保存在本机已有文档归档 `Data/maintenance/agents_audit_20261025/documentation_sources.zip`。
