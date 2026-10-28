# 冈田芽衣短节目

本页登记源时间、当前联合输入、结果范围与既有证据入口。赛事共同模型见[场景索引](../../scene/README.md)，接口能力见 [REBUILD](../../../../docs/REBUILD.md) 与 [INTERFACES](../../../../docs/INTERFACES.md)，准确性见 [VALIDATION](../../../../docs/VALIDATION.md)。文中的 `Data/` 指配置的外部数据根，本机为 `D:/CodexData/RinkSpace_Data/`。

## 源与范围

| 内容 | 权威记录 |
|---|---|
| ISU 官方单节目原片、SHA、表征与历史源对应 | [primary_source_selection.json](primary_source_selection.json)；1920×1080、50 fps、time_base=1/12800 |
| S01–S09 与物理／处理区间 | [rebuild_scope.json](rebuild_scope.json) |
| 当前结构、广告资产与参数关系 | [field/model.json](../../scene/field/model.json)、[版式定义](../../scene/print_layout_components.json) |
| 三个物理机位的型号及安装事实 | [用户事实](../../model_constraints/camera_hardware/user_camera_mount_and_height_20261006_v4.json)、[安装配置](../../../../configs/camera_hardware/okada_sp_installations_z190_v1.json)；型号不等于读出或畸变已标定 |

<a id="analysis-start"></a>
### 统一节目起点

当前重构拟合、网格交付与节目分析从 **原片 22.120 秒，PTS 283136** 开始，沿用原生时基 1/12800；该值与 [demo 计时窗口](outro_stats.json)一致。起点前画面、观测及结果保留，退出当前拟合。

权威区间为 scope 的 `processing_scope/processing`；`physical_scope/physical` 保留实际切镜边界。S01 处理 22.120–44.960 秒，共 1143 个曝光；历史物理 S01 从 20.020 秒开始，共 1248 帧。节目处理尾界为 188.680 秒。处理终点、物理切镜与节目计时零点分别解释。

## 当前四镜头、两机位联合输入

当前联合实验目录为 `Data/runs/native_joint_s01_s02_s04_s06_20261028/`，清单为 `joint_manifest.json`。本次输入覆盖以下四镜头，其余登记镜头未包含在这次联合实验中。

| 镜头 | 原片时间（秒，含末曝光） | 首／末 PTS | 独立曝光 | 点数 | 共享机位 |
|---|---|---|---:|---:|---|
| S01 | 22.120–44.960 | 283136–575488 | 1143 | 329112 | 主机位 |
| S02 | 44.980–56.300 | 575744–720640 | 567 | 94411 | 长边副机位 |
| S04 | 58.280–74.120 | 745984–948736 | 793 | 240846 | 主机位 |
| S06 | 101.660–111.360 | 1301248–1425408 | 486 | 97623 | 长边副机位 |
| 合计 | 四段原生时间区间 | 按各镜头登记 | 2989 | 761992 | 两组 rig |

表中点数对应 `point_observations.npz` 的 SHA256 `3c54bdd13672e3eb9728dcb7035f2bb059d0be1778984f4ff1ab8a3e9c1b1e6f`；后续源修订以同目录 manifest 和完整性记录的实际身份为准。七字段为 `pts, observation_ids, stations, source_rows, kind, uv, path_indices`，没有权重、sigma、相关统计或误差范围。来源修正绑定已有实体；保留点原 UV、整数 PTS、ID 等源字段不被模型投影覆盖。

两组相机采用 `fx/fy=1`、主点 `(959.5,539.5)`、主点变焦斜率为零的基线。这是工程近似，不是 Sony 标定。其余相机方向、四段独立运动与共享场景联合求解；切镜之间不施加运动连续。Ad26 平行方向和两端几何对称约束接入后，当前活动方向为运动 935、相机 52、场景 140，共 1127 列（Ad26 约束后、两端约束前为 1130 列）；注册列数不代表独立可辨识维数。

曝光 halo 使用每段端部三次曲线的连续延伸，初始状态和每个更新方向采用一致表示，保留帧中心区间内的运动空间。该输入的兼容数值种子为 `source_batch17_reviewed_seed.npz`；它包含相机、运动、场景与 station 数值，不是完整优化器磁盘检查点。

本轮按用户明确的三分准入逻辑处理大误差点：明确原实体则保留，明确另一实体则改绑，明确取错或没有明确结论则移出拟合；统一要求见 [BUILD_SPEC](../../../../docs/rink_model/BUILD_SPEC.md#observation-eligibility)。batch14–15 改绑 39 点、排除 138 点；随后 batch16 将已审仍未决及缺少明确实体证据的大误差点和邻帧点共 2047 点移出拟合，包含 S06 的 112 px F2 内圈误读点。所有原始记录与决定分别存于 `binding_review/source_batch14_c0_b1/`、`source_batch15_over50/` 和 `source_batch16_admission/`。`joint_source_batch15/` 在第 15 步停止，停止 cost 为 4,758,470.44；保持该相机、运动、场景不变清理观测后，`batch16_evaluation/` 的 cost 为 4,362,148.83，S06 最大误差 49.68 px。该变化是数据修订收益，不是优化收益。当时高于 50 px 的 5 个 S02 门区点曾在 `review_s02_crease_remaining/review.json` 中被错误放行；batch17 根据原片闭合直边、圆弧端点和反光的重新区分撤销了该放行，历史错误结论保留供追查。下段 batch13 状态和视频为先前冻结交付，不能作为当前输入的兼容续算状态。

上一轮固定数值结果为 `review_source_batch16/candidate_pixels.npz`，输入仍为上述 batch16；`joint_source_batch16/` 在第 13 步按用户要求保存停止，cost 从 4,362,148.83 降至 4,328,912.05。精简日志为 `joint_source_batch16/fit_progress.log`，四镜头完整视频写入 `review_source_batch16/`，冻结身份见该目录的 `frozen_review.json`，实际导出完成及逐帧验证以 `reviewed_videos.json` 为准。此次仅继续求解当前数据，未再改动读点；停止时 S01/S02/S04/S06 最大误差分别为 47.48/96.28/35.00/63.27 px，仍存在待审问题，不能称为完整拟合或物理参数验收通过。用户要求导出本轮视频后暂停全部任务工作。

暂停后，先按用户要求单独接入 [Ad26 平行方向约束](../../scene/README.md#共同参数与版式)，对应数值状态为 `ad26_parallel_constraint/candidate_pixels.npz`。只将 Ad26 绕徽标中心摆正并固定安装角，未运行优化；相机、运动、其余场景数值、station 和全部七字段读点不变。旧模型及其完整资产保存在 `ad26_parallel_constraint/prior_scene/`，旧视频保持原样。变更及验证见 `ad26_parallel_constraint/verification.json`，native 只评估结果见同目录 `evaluation/`；3 个原片时刻的投影见 `after/`。该候选 cost 为 4,358,237.05，因施加方向约束而变化，不是拟合收益。

随后按用户要求接入两端球门线、半椭圆和梯形的镜像及中心重合关系，对应数值初值为 `goal_end_constraints/candidate_pixels.npz`；Ad26 约束继续保留。旧状态左球门线为 −0.487297306 L、右球门线为 +0.431690637 L；新共享半距初值为 0.438022601 L，取有观测右端作参考，并满足已有 Lane 左端绘制支持不越过左球门线的约束。共享半轴取右端原值。此次迁移只改这组几何，源读点、相机、运动、station 及其他实体数值不变。全量 GPU 初值评估 cost 为 33,131,362.16，S02 最大误差为 417.71 px；这是强制新几何关系后尚未重拟合的初值。前后场景、数值、native 构建和严格镜像检查见 `goal_end_constraints/verification.json`、`source_manifest.json`、`prior_scene/` 和 `evaluation/`。

用户随后要求用新 native 试跑一轮。首次启动在曲线 station 初始化后因旧边界检查仍读取已停用的左球门线独立槽而失败，保留于 `goal_end_constraints/joint_fit/`。已同时修正可行步投影、耦合边界法线和驻点判定，使三处均使用 `左球门线=-d`，旧几何、共享距离的加法与正值参数化回归均包含真实下降步。修复后的 `goal_end_constraints/joint_fit_fixed/` 完成 20 个试探步，其中 12 步接受，约 261 秒；曲线位置初始化将 cost 从 33,131,362.16 降至 6,569,799.03，联合优化再降至 4,328,930.23。最终最大误差 96.33 px，超过 20 px 共 1751 点，无无效投影；S01/S02/S04/S06 最大误差分别为 47.49/96.33/35.05/63.45 px。后段降幅明显缩小，按本轮试跑目的在安全边界停止，`numerically_converged=false`；没有改动七字段输入或新增权重。

本次修订前的冻结状态为 `goal_end_constraints/review_joint_fit/candidate_pixels.npz`，只与 batch16 输入匹配。用户要求输出这一版四镜头完整视频；同目录的 `frozen_review.json` 绑定快照、源数据、native 库和几何约束检查，完整导出与逐帧 PTS 核验以 `reviewed_videos.json` 为准。新对称、中心重合、圆弧范围和 Ad26 平行约束在最终参数中保持成立；视频等待用户检查，未宣称画面验收或全局收敛。

当前输入已经应用 `binding_review/source_batch17_over40/` 的源图复核：覆盖全部 197 个 >40 px 根点及已审同源观测，共 2743 条逐点决定；改绑 294 点、修正 UV 2 点、排除错误或身份未决点 1989 条，并在 11 个源曝光补入 23 条实拍弧线／灰线观测。S02 后段 143 条原拟排除的灰线点通过 S01 用户标注、S04 原片和 S02 连续关系找回实体身份，保留原 UV。原片没有可靠材料的区域未强行补点；S02 仍有 10 个曝光缺冰面点，但均保留围挡／广告观测；S06 F4 的 493 条反光、遮挡或身份未决读点已退出，约束缺口明确保留。覆盖见 `prospective_source_coverage.json`，其中计数现已对应正式输入；数量不证明参数可辨识性。

`current_resume_snapshot` 现指向 `source_batch17_reviewed_seed.npz`；相机 rig、运动 controls、场景值和 native 几何／库逐项不变，S01／S04 七字段逐项不变，只对改绑、改 UV 和新增观测初始化对应 station。当前输入 SHA256 为 `bdad7f4a34191a7d21253caebd4f34b50d6768a1cd6e913f37781a4ad455020d`，共 761992 点、2989 个源曝光。`binding_review/source_batch17_over40/evaluation/` 仅执行固定参数前向检查、优化步数为零：cost 为 3,916,169.79，>20 px 共 1208 点，无无效投影；S01／S02／S04／S06 最大误差为 47.49／37.95／35.05／39.92 px。只剩用户已确认是实际拟合误差的 S01 F3 点超过 40 px；新增 23 点最大误差为 8.50 px。该变化来自数据修订和改动点 station 初始化，不是相机／几何优化收益，不构成全镜头验收。修订前完整输入、旧坐标与证据、逐点回执、兼容种子和最终完整性结果分别保存在 `point_observations_before_review.npz`、`applied/`、上述种子和 `final_check.json`。按用户要求，完成本次检查后全部停止。

输入完整性见 `binding_review/current_input_integrity.json`。历史 `joint_source_visual_reviewed/` 在第 36 步停止（cost 5,065,462.82），`joint_interior_mount/` 在第 32 步停止（cost 5,032,527.26）；后者的冻结状态和四镜头视频保留在 `review_source_batch13/`。这些旧状态和数值质量按各自产物的输入 SHA 解释。两组机械初值与失败候选保留，杆臂、轴间距和入瞳变焦参数在当前联合拟合中仍开放。当前库 `goal_end_constraints/build_goal_boundary/rinkspace_core.dll` 延续正式 `dev/zoom_branch_build/` 的优化算法，增加本次两端几何选项、GPU 入口和相应物理边界映射；当前构建源与 SHA 见 `goal_end_constraints/boundary_fix_build.json`，首次几何构建记录保留在 `source_manifest.json`，未采用工作区的实验优化算法。已有候选仍有误差尾部、整体网格偏移、材料身份与相机运动待验证，四镜头整体尚未通过完整原片验收。

## 有界结果与人工确认

| 对象 | 已确认范围与限制 |
|---|---|
| S01／S04 等轴焦距基线 | 用户确认对应两镜头视频的网格贴合满意。确认状态为 `Data/runs/native_joint_s01_s04_20261027/camera_models/square_repair/review/candidate_pixels.npz`，SHA256 `c7bf576dc92a635fb254633c8fb1da1ce02f309cc42e2dba2d11fd72bf7c00c4`；输入为 569960 点／1936 曝光。该确认不扩展至后续主点模型、四镜头候选、相机物理参数或已验证米制 |
| 主点核查 | `camera_models/center_pp_square/result.json` 保存上述基线确认及居中常量主点实验；数值精修仍为非收敛停滞，相机安装／入瞳分解和广角支持存在不确定性 |
| [Ad5 矩形对照](ad5_rectangle_review.json) | S05，PTS 1273856／99.52 秒，用户意见为“看不出新旧变化”；限于该曝光的新旧矩形对照，不扩展到其他 PTS、Ad18 或源定位精度 |
| [Ad2 O 边界](ad2_o_review.json) | PTS 320000／25 秒，两个既有记录的身份与读数；原 UV 未改，`active_for_fit=false`，不代表模板或相机拟合通过 |
| [来源读取检查](source_reader_audit.json) | 历史恢复检查主要覆盖同一组 10 帧；带人工引导和已有遮挡掩膜，不证明全 S01 或全节目自动取点已恢复。后续候选接线能力见 REBUILD，完整真实节目闭环仍未验证 |
| [已撤销的 O0.1 对照](o01/result.json) | 保留原卡与失败，仅供历史追查；已有人工标注和确认范围保留 |

上述 `camera_models/...` 相对 `Data/runs/native_joint_s01_s04_20261027/`。人工认可绑定具体输入、状态与画面；优化器 accepted、数值有效、覆盖 complete 和用户验收是不同字段。

## 来源修订与证据位置

当前四镜头的逐点修订保存在联合目录 `binding_review/`，该输入相对原始导入累计 1859 点改绑、4868 点退出拟合、2 点修正 UV、23 点新增；处置依据包括原片材料身份确认，以及用户明确要求将身份未决点退出拟合。S01 七字段保持原样；S04 在原版本上另改绑 22 个短边观测、排除 2 个反光误读。S04 的 22 个改绑包含主会话依据连续源线走势和用户判断作出的身份推断，独立视觉复核的弱对比疑问一并保留在逐点记录中，尚不构成完整段落验收。所有修改前的原 UV、PTS 与 ID，以及修订前输入、排除点和图像证据均保留；当前 UV 的两处源图更正单列记录，其余保留点坐标不变。

| 证据组 | 既有入口 |
|---|---|
| S01 冰面与广告身份 | [冰面首批](ice_identity_review_20261026.json)、[冰面后续](ice_identity_review_20261027.json)、[Ad4 边界](ad4_boundary_identity_review_20261026.json)、[Ad6 断笔](ad6_disconnected_contour_review_20261026.json)、[Ad6 タ](ad6_ta_identity_review_20261027.json) |
| S01 模型与失败候选 | `Data/runs/native_joint_s01_20261026/` 中的 `print_shape_audit/`、`print_component_layout/`、`print_artwork_layout/`、`camera_identifiability/`、`ice_residual_audit/` |
| S01／S04 同机位实验 | `Data/runs/native_joint_s01_s04_20261027/joint_manifest.json`；同目录保留来源替换、身份复核、原输入、数值状态及 `camera_models/` |
| S02／S04／S06 来源修订 | 当前四镜头目录中的 `binding_review/`、`review_batch5/` 及其逐点来源记录；S02 和 S04 原始包仍在对应原运行目录 |
| 相机表示与数值检查 | 当前四镜头目录中的 `physics_review/`、`dev/`；这些检查不构成物理参数或图像验收 |
| 广告来源包 | [S04 广告包索引](source_datasets/ad_points_s04_20261026/result.json)；属于来源候选包，不等于当前完整联合输入 |

源身份修订、模型版式变化和优化收益分别解释。旧版大自由度广告、错误绑定、失败相机初值和停止状态保留在这些既有目录中，当前模型不恢复逐字／逐行自由变形。

## 原生工程对照

| 记录 | 验证范围 |
|---|---|
| [R1](r1/exposure.json) | PTS 320000 单曝光完整读出域回放，未拟合 |
| [R2 摘要](r2/result.json)、[窗口](r2/window.json) | 24–26 秒、101 个曝光的前向等价，未验收完整 S01 |
| [R3](r3.json) | 六条既有轮廓的几何投影差，不是实测源残差或改善证明 |
| [G1](g1.json) | 同窗口 5920 行的 CPU／GPU 原始残差等价，未更新状态 |
| [G2](g2.json) | 同窗口局部导数、相关 W 与块累计对照；当时只开放三个相机常量方向 |
| [G3](g3.json)、[保存状态](g3/state.json) | 同窗口一次 LM 候选接受；末段仍有误差上升，未完成真实拟合验收 |
| 连续原生 LM | 当前能力见 [native](../../../../native/README.md#当前模型的-gpu-求解入口)；上述历史窗口的因子限制不代表当前引擎限制，完整磁盘恢复仍未实现 |

<a id="s01s04-当前同机位联合输入"></a>
<a id="legacy-maintenance"></a>
## 历史索引

| 历史对象 | 入口与范围 |
|---|---|
| d78 原版、恢复状态与后续视觉拒绝 | [s01_original/index.json](s01_original/index.json) |
| 旧 solve2 输入绑定 | [solve2/input.json](solve2/input.json) |
| 六镜头试验与停止状态 | [fit_joint6/status.json](fit_joint6/status.json)、[scope.json](fit_joint6/scope.json) |
| 2026-10-19 九镜头失败快照 | [review](review/README.md)，读取方法见[离线审阅](../../../../docs/GITHUB_REVIEW.md) |

历史状态按原输入、库与源时间解释，不代替当前 manifest，也不自动成为当前拟合输入。
