# 冈田芽衣短节目

本页登记源时间、当前拟合状态、结果绑定与人工确认范围。赛事共同模型见[场景索引](../../scene/README.md)，能力与接口见 [REBUILD](../../../../docs/REBUILD.md) 和 [INTERFACES](../../../../docs/INTERFACES.md)，准确性见 [VALIDATION](../../../../docs/VALIDATION.md)。文中的 `Data/` 指配置的外部数据根。

## 当前完整拟合：暂定最终状态

**当前状态为 `camera_solver_refined_1102`，用户于 2026-11-02 将其指定为完整九镜头拟合的暂定最终状态。** 精确输入、数值状态、原生库与回放由[当前结果绑定](fit_v11/camera_solver_refined_1102.json)统一维护，精简参数为 [camera_solver_refined_1102.npz](fit_v11/camera_solver_refined_1102.npz)。

| 项目 | 当前范围与状态 |
|---|---|
| 镜头与机位 | S01–S09，三个物理机位，共享赛事场景 |
| 源时间 | 原片 22.120–188.680 秒，8329 个原生曝光，50 fps，time_base=1/12800 |
| 输入与目标 | 2,213,335 行原始像素观测，无 W／Huber；输入 SHA256 `ecb7012af725c94026ba85c9249e731e560c20d2f2189efd25cb28113512f5fb` |
| 模型 | 冰面关系 ±1% 容差；相机、运动与场景联合，三机位光学曲线独立，共享有效行时序 |
| 完整冻结状态 | SHA256 `b9f1cb81c8f11163152e7ca4fea16494cad91951be304dfe7c2182c6db4fabec`，本机位置见当前结果绑定 |
| 拟合 | 因代价进展停滞停止，未达到一阶数值收敛；原始像素 RMS 约 2.314956 px |
| 回放 | 完整九镜头视频已导出，8329 帧、原生 PTS 和 1× 播放速度已核验 |
| 确认范围 | 暂定最终状态的指定已登记；独立相机物理标定、米制及完整节目准确性仍未验证 |

本地视频：`Data/runs/native_programme_20261031/feature_refinement_1101/focus_repair_1pct/review_camera_solver_refined_1102/camera_solver_refined_1102_complete.mp4`，对应原片 00:22.120–03:08.680，供本地播放器检查。视频、完整运行输入和编译库不随公开仓库分发。

当前状态保留入瞳变焦项，完整前向工作限幅为 −10…+30 cm；端机位当前畸变项保留。这些是该状态的工程选择，不是 Z190 厂家实测参数。S08 开头少数曝光仍有约 4–6 px 局部运动偏差，S05 末曝光与 S08 首曝光的读出窗口仍有亚帧加速度尖峰。[机器可读复核](fit_v11/camera_solver_refined_1102_review.json)保存对应证据与解释范围；暂定最终登记不扩大这些结论。

## 人体与足鞋结果

当前主要检查入口如下，所有人体候选均未通过独立三维准确性或真实接冰验收：

| 结果 | 原片范围 | 当前状态与绑定 | 检查入口 |
|---|---|---|---|
| S01 连续候选 `skate_s01_1103` | 22.120–44.960 秒，1143 个曝光 | 用户目视较满意；仍有穿冰／离冰、部分鞋刀图像误差和开头姿态回退；[结果](skate_s01_1103.json)、[参数](skate_s01_1103.npz) | [S01 比较](../../../../launchers/START_3D_COMPARISON.cmd)，默认从 22.120 秒开始 |
| S05 完整源结构与条件支撑位置候选 `sequence_source_structure` | 76.000–95.000 秒，951 个曝光 | 完整源轮廓和条件支撑 XY 已进入同一连续状态；预算停止，刀尾、支撑位置及转出身份仍有失败；[结果](spin_source_structure_1106_audit.json)、[参数](spin_sequence_source_structure_1106.npz)、[配置](viewer3d_sequence_source_structure_1106.json) | [S05 源结构检查](../../../../launchers/START_3D_SEQUENCE_BLADE.cmd)，无参数启动按配置从 87.640 秒开始 |
| S05 新源像素与有限软支持候选 `support_grounded` | 76.000–95.000 秒，951 个曝光 | 支撑native回投、完整结构反向与源梯度进入同一状态；三轮16/10/13接受步；漂移仅小幅减轻，91.20秒俯仰和局部图像退化仍在，未通过整体视觉验收；[结果](spin_support_correction_1106_audit.json)、[参数](spin_support_grounded_1106.npz)、[配置](viewer3d_source_fusion_support_1106.json) | [本轮新旧对照](../../../../launchers/START_3D_SOURCE_FUSION_SUPPORT.cmd)，保留所有旧候选 |

人体、骨架和鞋刀由同一状态求值，场景和相机冻结于 `camera_solver_refined_1102`；这些对照使用 59.96169706678219 米／场景单位的条件尺度，未完成独立米制标定。普通历史 S01 播放器的另一套尺度按[操作说明](../../../../docs/RESULTS_3D_PLAYER_ZH.md#画面含义与尺度)单独解释。

S05 当前源结构只有 36 个曝光、40 条鞋刀轮廓及 32 条条件支撑位置；其余曝光为连续模型估计。五条转出轮廓未确认左右身份，87.640 秒刀尾偏高，92.800 秒移动见证点仍偏离源范围。条件位置依赖同帧相机和冰面，移动最低点不等于固定材料点或真实受力点；同源图像残差下降不能作为独立定位验收。

本轮新候选不再将整条源下缘投到Z=0的凸包当成已知接触位置；32条支持候选改用native图像带匹配，旧XY仅诊断。42条完整可见结构反向因素和34条源像素剖面分摊既有证据预算，保留有限软冰面、踝枢轴附着及原始UV/半径。人体旧代理仍在源改进方向上产生明显回拉，精确接触材料和内外刃身份未确定；没有新增独立源曝光或将模型最低点当成接冰真值。结果与失败原件在 `Data/runs/body_s01_20261029/support_correction_1106/RESULT_ZH.md`。

FS-Jump3D验证支线已实际完成一个300帧、12视角试次；相机7单独输入ExMR，其他视角用于评价，没有训练或姿态/世界配准。其输入图像误差较小仍伴随明显视线深度误差，说明单视角图像贴合不足以证明世界定位；不能据此断言S05的唯一错误来源。[试次与结果身份](../../../../configs/continuous_inputs/fs_jump3d_skater_a_axel_1_1106.json)连接 `Data/runs/fs_jump3d_1106/RESULT_ZH.md`，原视频、完整模型输出和验证证据留Data。

<a id="人体整合与根轨迹对照2026-11-04未验收"></a>

### S05 与动作中间结果导航

| 对照组 | 范围与限制 | 结果绑定 |
|---|---|---|
| 跳跃／步法／旋转 A/B 与根修正 | 44.980–52.000、25.000–35.000、76.000–95.000 秒；根轨迹异常及部分图像退化仍在 | [三段结果](body_integration_1104.json)；[片段选择](../../../../launchers/START_3D_INTEGRATION.cmd) |
| 源支撑与网络初值对照 | 源支撑核心 88.000–92.000 秒、上下文 87.500–92.500 秒；网络对照为 76.000–95.000 秒；世界漂移及结构 unknown 保留 | [源支撑](spin_support_1104_audit.json)、[GVHMR／ExMR](spin_models_1104_audit.json) |
| 抓握与连续入旋 | 76.000–95.000 秒；抓握／深度、重力与源图回投仍未验收 | [抓握](spin_grasp_1104_audit.json)、[入旋](spin_entry_1104_audit.json) |
| 绝对连续 A/B、整段续算与局部关联修复 | 整段状态均覆盖 76.000–95.000 秒，局部修复只改 81.200–82.000 秒；中间版本分别保存 | [A/B](spin_continuous_1105_audit.json)、[整段续算](spin_continuation_1105_audit.json)、[局部修复](spin_281_repair_1105_audit.json) |
| 靴轮廓、刀轴与 runner 支撑形态 | 局部轮廓／无向轴不是刀端材料或接冰真值；姿态和回投失败保留 | [鞋刀结果](body_blade_1105_audit.json)、[靴轮廓证据](boot_axis_evidence_1105.json)、[条件形态](runner_support_mode_1105.json) |
| 较早完整鞋刀源描线 | 22 个源曝光，用于原图结构复核；不代表全段真实接触 | [源描线](spin_source_skate_structures_1105.json)；当前源版本为[1106 描线](spin_source_skate_structures_1106.json) |

旧 `skate_s01_1103` 和 `skate_s01_integrated_1103` 不应解释为已使用后续有限冰面软项重新拟合。未知接触与腾空不按支持状态处理，区域内零残差及模型参考高度均不能替代真实接冰验收。

### 历史人体候选导航

| 候选 | 范围与解释 | 绑定 |
|---|---|---|
| 较早鞋刀接冰 `skate1103` | 身体可动窗口 28.120–43.800 秒，共享鞋刀回放整个处理 S01；已封存，图像与接冰仍有失败 | [结果](skate_contact_1103.json)、[参数](skate_contact_1103.npz) |
| 开头时序与右臂 `hand1103` | 修复 22.120–25.000 秒；用户指出基本到 22.900 秒只有手部动作，此后才下探；仍有腕点偏差和刀刃高度残留 | [身体时序](opening_timing_1103.json)、[右臂](opening_hand_1103.json) |
| 开头动态失败 `motion1103` | 22.120–25.000 秒；下探过早、开头约一秒平移 1.11 米，用户已否定 | [原结果](opening_motion_1103.json) |
| 开头裁切失败 `opening1102` | 22.120–25.000 秒；固定姿态和近静止条件压制真实动作，后续版本已继续修正 | [原结果](opening_repair_1102.json) |
| 足鞋穿戴候选 | 新增身体优化为 35.200–36.800、40.000–41.500 秒共 157 帧；足鞋配准作用于整个处理 S01；内腔改善伴随外鞋图像误差变差 | [结果](boot_integration_1102.json)、[穿戴检查](../../../../launchers/START_BOOT_INSPECTION.cmd) |
| 全身五短段候选 | 24.500–27.000、32.200–33.800、35.200–36.800、37.400–39.000、40.000–41.500 秒，共 445 帧；其余 S01 帧继承 J1，存在明显穿冰 | [结果](body_fullbody_1102.json)、[播放器配置](viewer3d_fullbody_1102.json) |
| 较早整段整合 | 有限冰面软项之前的历史 S01 对照，未验收 | [结果](skate_s01_integrated_1103.json) |

参数字段、SHA 与分发边界见[字段索引](body_parameters_1103_fields.json)和[许可核查](../../../../docs/LICENSING_REVIEW.md#body-parameters)；复核源码见[来源索引](body_verification_sources_1103.json)。播放器的默认版本、源时间、条件尺度与操作见[使用说明](../../../../docs/RESULTS_3D_PLAYER_ZH.md)。完整结果、模型、源图、逐轮日志和可视化证据保留在 `Data/runs/body_s01_20261029/`，由各结果记录定位。

## 源与范围

| 内容 | 权威记录 |
|---|---|
| ISU 官方单节目原片、SHA、表征与历史源对应 | [primary_source_selection.json](primary_source_selection.json)；1920×1080、50 fps、time_base=1/12800 |
| S01–S09 与物理／处理区间 | [rebuild_scope.json](rebuild_scope.json) |
| 名义结构、当前场景与广告版式 | [名义结构](../../scene/field/model.json)、[当前运行场景](../../scene/field/ad4_physical_supported.json)、[版式定义](../../scene/print_layout_components.json) |
| 三个物理机位的型号及安装事实 | [用户事实](../../model_constraints/camera_hardware/user_camera_mount_and_height_20261006_v4.json)、[安装配置](../../../../configs/camera_hardware/okada_sp_installations_z190_v1.json)；型号不等于读出或畸变已标定 |

<a id="analysis-start"></a>
### 统一节目起点

当前重构拟合、网格交付与节目分析从 **原片 22.120 秒，PTS 283136** 开始，沿用原生时基 1/12800；该值与 [demo 计时窗口](outro_stats.json)一致。起点前画面、观测及结果保留；19.960 秒准备画面用于姿势和定位辅助，节目计时与处理起点不变。

权威区间为 scope 的 `processing_scope/processing`；`physical_scope/physical` 保留实际切镜边界。S01 处理 22.120–44.960 秒，共 1143 个曝光；历史物理 S01 从 20.020 秒开始，共 1248 帧。节目处理尾界为 188.680 秒。处理终点、物理切镜与节目计时零点分别解释。

## 输入与模型

七字段点输入为 `pts, observation_ids, stations, source_rows, kind, uv, path_indices`。源 UV、整数 PTS、观测身份与模型预测分别保存；来源修订和数值优化的收益分别解释。当前结果绑定连接实际输入、场景、相机配置、完整状态和原生库；完整优化器磁盘断点恢复仍未实现。

S07–S09 使用既有完整源记录作来源对照，并接入经过身份与人物遮罩检查的长线、圆环读数。来源候选、活动点数和九镜头接入不自动表示每条可见特征已完整覆盖或每个读点都已验收。

同一物理安装共享 rig，不同镜头使用独立运动块，切镜之间不强加运动连续。当前独立光学曲线和共享读出按当前结果绑定解释；旧基线的固定倍率及共享径向关系属于各自历史配置。

源素材准备见 [prepare_source_bootstrap.py](prepare_source_bootstrap.py) 和[无损切片准备](../../../../calibration/program_tracking/body_tracking/prepare_source_clip.py)。前者按 `--scope` 核对源身份与 PTS，输出至 `--out`；后者使用 `--review` 和 `--out` 保留 S01 包字节及解码像素核对。准备输出不代表整段画面验收。

## 有界结果与人工确认

| 对象 | 已确认范围与限制 |
|---|---|
| 当前完整九镜头暂定最终状态 | 用户于2026-11-02明确指定 `camera_solver_refined_1102` 为完整拟合的暂定最终状态；参数、DLL、视频及已知局部限制绑定[当前状态记录](fit_v11/camera_solver_refined_1102.json) |
| 历史九镜头 joint_mount_flex | 用户于 2026-11-01 表示“我对网格的贴合基本满意，虽然还有局部贴合不完全的地方”；绑定完整状态 SHA256 `b7ab8e53cbd4a1146f5456a3698d6bba43de84a961dec9d373b94be3d2ca2570` 和 [原状态记录](fit_v11/camera_review_mount_flex.json)。不表示相机物理标定、全部源对应或全节目准确性通过 |
| 六镜头 Lane 基线 | 用户表示“我对网格的贴合基本满意”；仅绑定 `Data/runs/native_programme_20261028/six_shot/lane_source_refinement/review/frozen_state.npz`，SHA256 `54234d25cbf3e8d03dc3d52fdef5d367aa87ce59889cccd904ad1303ac175473`，927425 点／4463 曝光。精简参数为 [native_lane_source_parameters.npz](native_lane_source_parameters.npz)。不扩展到后续相机简化、冰面容差或九镜头候选，也不表示相机物理参数通过 |
| S01／S04 等轴焦距基线 | 用户确认对应两镜头视频的网格贴合满意。确认状态为 `Data/runs/native_joint_s01_s04_20261027/camera_models/square_repair/review/candidate_pixels.npz`，SHA256 `c7bf576dc92a635fb254633c8fb1da1ce02f309cc42e2dba2d11fd72bf7c00c4`；输入为 569960 点／1936 曝光。该确认不扩展至后续主点模型、四镜头候选、相机物理参数或已验证米制 |
| 主点核查 | `camera_models/center_pp_square/result.json` 保存上述基线确认及居中常量主点实验；数值精修仍为非收敛停滞，相机安装／入瞳分解和广角支持存在不确定性 |
| [Ad5 矩形对照](ad5_rectangle_review.json) | S05，PTS 1273856／99.52 秒，用户意见为“看不出新旧变化”；限于该曝光的新旧矩形对照，不扩展到其他 PTS、Ad18 或源定位精度 |
| [Ad2 O 边界](ad2_o_review.json) | PTS 320000／25 秒，两个既有记录的身份与读数；原 UV 未改，`active_for_fit=false`，不代表模板或相机拟合通过 |
| [来源读取检查](source_reader_audit.json) | 历史恢复检查主要覆盖同一组 10 帧；带人工引导和已有遮挡掩膜，不证明全 S01 或全节目自动取点已恢复。后续候选接线能力见 REBUILD，完整真实节目闭环仍未验证 |
| [已撤销的 O0.1 对照](o01/result.json) | 保留原卡与失败，仅供历史追查；已有人工标注和确认范围保留 |

上述 `camera_models/...` 相对 `Data/runs/native_joint_s01_s04_20261027/`。人工认可绑定具体输入、状态与画面；优化器 accepted、数值有效、覆盖 complete 和用户验收是不同字段。

<a id="原生工程对照"></a>
2026-11-01／02 源图整理与资格打包、最终参数导出的五份执行源码已保存在 [feature_refinement_1101/source_code.json](source_datasets/feature_refinement_1101/source_code.json) 登记的目录；与 Data 执行原件字节一致，需要各自现有输入，不在正常求解启动时重放。

## 来源与历史导航

| 对象 | 既有入口与范围 |
|---|---|
| 历史九镜头 joint_mount_flex | [状态与人工反馈绑定](fit_v11/camera_review_mount_flex.json)、[精简参数](fit_v11/focus_mount_flex_1pct.npz)；人工认可仅适用于原快照 |
| 相机中间对照 | [固定杆臂对照](fit_v11/fixed_lever_y_comparison.json)、[相机修正](fit_v11/camera_physical_1102.json)、[局部摇镜修正](fit_v11/camera_physical_pan_1102.json)、[相机收整](fit_v11/camera_consolidated_1102.json)；各自保留独立输入和参数 |
| 较早 guarded_lines 候选 | [原结果绑定](fit_joint9/native_joint9_guarded_lines.json)、[参数](fit_joint9/native_joint9_guarded_lines.npz)；其输入和停止状态不用于描述当前版本 |
| S01 冰面与广告身份 | [冰面首批](ice_identity_review_20261026.json)、[冰面后续](ice_identity_review_20261027.json)、[Ad4 边界](ad4_boundary_identity_review_20261026.json)、[Ad6 断笔](ad6_disconnected_contour_review_20261026.json)、[Ad6 タ](ad6_ta_identity_review_20261027.json) |
| S04 广告来源候选 | [广告包索引](source_datasets/ad_points_s04_20261026/result.json)，不等于当前完整联合输入 |
| 早期窗口级原生检查 | [R1](r1/exposure.json)、[R2](r2/result.json)、[R3](r3.json)、[G1](g1.json)、[G2](g2.json)、[G3](g3.json)；不代表完整镜头验收 |

<a id="s01s04-当前同机位联合输入"></a>
<a id="legacy-maintenance"></a>
## 历史索引

| 历史对象 | 入口与范围 |
|---|---|
| d78 原版、恢复状态与后续视觉拒绝 | [s01_original/index.json](s01_original/index.json) |
| 旧 solve2 输入绑定 | [solve2/input.json](solve2/input.json) |
| 2026-10-21 六镜头试验与停止状态 | [fit_joint6/status.json](fit_joint6/status.json)、[scope.json](fit_joint6/scope.json) |
| 2026-10-19 九镜头失败快照 | [review](review/README.md)，读取方法见[离线审阅](../../../../docs/GITHUB_REVIEW.md) |

原 UV／PTS、逐轮修订、人工意见、失败与相机报告保留在对应 Data 运行目录，由结果记录定位。旧入口与报告原文保存在既有 `Data/maintenance/agents_audit_20261025/documentation_sources.zip`，历次正文原文按日期成员保存；本机归档不随公开仓库分发。历史状态按各自输入与源时间解释。
