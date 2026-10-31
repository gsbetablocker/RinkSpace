# 冈田芽衣短节目

本页登记源时间、当前联合输入、候选状态与人工确认范围。赛事共同模型见[场景索引](../../scene/README.md)，能力与接口见 [REBUILD](../../../../docs/REBUILD.md) 和 [INTERFACES](../../../../docs/INTERFACES.md)，准确性见 [VALIDATION](../../../../docs/VALIDATION.md)。文中的 `Data/` 指配置的外部数据根。

## 当前九镜头候选

当前候选为 `native_joint9_guarded_lines`，精确身份和本机产物位置见[结果绑定](fit_joint9/native_joint9_guarded_lines.json)，公开数值参数见 [native_joint9_guarded_lines.npz](fit_joint9/native_joint9_guarded_lines.npz)。

| 项目 | 当前范围或状态 |
|---|---|
| 镜头与机位 | S01–S09，三个物理机位，共享同一赛事场景 |
| 原片范围 | 22.120–188.680 秒，8329 个原生曝光，50 fps |
| 输入 | 1,360,063 条源观测；原始像素平方误差目标，不含统计 W 或 Huber |
| 参数化 | 66 个独立相机列，263 个活动场景列；冰面关系采用 ±1% 容差版本 |
| 拟合 | 已执行真实联合求解；因代价进展停滞停止，未数值收敛 |
| 回放 | 完整候选视频已导出，冻结状态与逐帧源 PTS 已核验 |
| 验收 | 整段视觉验收与完整节目准确性验证尚未完成 |

本地候选视频：`Data/runs/native_programme_20261031/review_guarded_lines/native_joint9_guarded_lines_complete.mp4`，对应原片 00:22.120–03:08.680。视频与大型输入不随公开仓库分发。

当前候选继承相机简化和[冰面容差场景](../../scene/field/ice_tolerance_current.json)。该容差版本已用于六镜头真实拟合及九镜头联合候选；六镜头结果保存在 `Data/runs/native_ice_tolerance_20261030/reviewed_source_fit/tolerance_station/joint_fit/result.json`。零偏差准备状态、已接受的数值状态和完整优化器检查点分别解释；完整磁盘断点恢复仍未实现。

已有少量静帧检查不能代表完整原片验收。源特征覆盖、材料身份、相机机械／光学分解、工作边界与场地米制仍有待验证项；具体限制见结果绑定。下列人工确认各自绑定独立快照。

## 源与范围

| 内容 | 权威记录 |
|---|---|
| ISU 官方单节目原片、SHA、表征与历史源对应 | [primary_source_selection.json](primary_source_selection.json)；1920×1080、50 fps、time_base=1/12800 |
| S01–S09 与物理／处理区间 | [rebuild_scope.json](rebuild_scope.json) |
| 名义结构、候选场景与广告版式 | [名义结构](../../scene/field/model.json)、[冰面容差结构](../../scene/field/ice_tolerance_current.json)、[版式定义](../../scene/print_layout_components.json) |
| 三个物理机位的型号及安装事实 | [用户事实](../../model_constraints/camera_hardware/user_camera_mount_and_height_20261006_v4.json)、[安装配置](../../../../configs/camera_hardware/okada_sp_installations_z190_v1.json)；型号不等于读出或畸变已标定 |

<a id="analysis-start"></a>
### 统一节目起点

当前重构拟合、网格交付与节目分析从 **原片 22.120 秒，PTS 283136** 开始，沿用原生时基 1/12800；该值与 [demo 计时窗口](outro_stats.json)一致。起点前画面、观测及结果保留，退出当前拟合。

权威区间为 scope 的 `processing_scope/processing`；`physical_scope/physical` 保留实际切镜边界。S01 处理 22.120–44.960 秒，共 1143 个曝光；历史物理 S01 从 20.020 秒开始，共 1248 帧。节目处理尾界为 188.680 秒。处理终点、物理切镜与节目计时零点分别解释。

## 输入与模型

七字段点输入为 `pts, observation_ids, stations, source_rows, kind, uv, path_indices`。源 UV、整数 PTS、观测身份与模型预测分别保存；来源修订和数值优化的收益分别解释。当前冻结输入、场景、native 库和数值状态由结果绑定及其 Data 内的 `frozen_problem/joint_manifest.json` 关联。

S07–S09 使用既有完整源记录作来源对照，并接入经过身份与人物遮罩检查的长线、圆环读数。来源候选、活动点数和九镜头接入不自动表示每条可见特征已完整覆盖或每个读点都已验收。

同一物理安装共享 rig，不同镜头使用独立运动块，切镜之间不强加运动连续。固定倍率、共享径向曲线及安装参数约束是当前工作模型，现场设置、安装尺寸和物理参数的独立核验仍未完成。

源素材准备入口已保存在源码树：[prepare_source_bootstrap.py](prepare_source_bootstrap.py)保留 S07–S09 固定范围、源 SHA 与原生 PTS 检查，`--scope` 指定范围文件、`--out` 指定外部 Data 输出；只生成代表帧，不构成整段画面验收。S01 私有审阅的[无损切片准备](../../../../calibration/program_tracking/body_tracking/prepare_source_clip.py)使用 `--review` 和 `--out`，保留 283136–575488 PTS 的包字节与解码像素核对。两者的执行原件与历史回执继续保留在各 Data 运行目录，未随此次整理生成新媒体。

## 有界结果与人工确认

| 对象 | 已确认范围与限制 |
|---|---|
| 六镜头 Lane 基线 | 用户表示“我对网格的贴合基本满意”；仅绑定 `Data/runs/native_programme_20261028/six_shot/lane_source_refinement/review/frozen_state.npz`，SHA256 `54234d25cbf3e8d03dc3d52fdef5d367aa87ce59889cccd904ad1303ac175473`，927425 点／4463 曝光。精简参数为 [native_lane_source_parameters.npz](native_lane_source_parameters.npz)。不扩展到后续相机简化、冰面容差或九镜头候选，也不表示相机物理参数通过 |
| S01／S04 等轴焦距基线 | 用户确认对应两镜头视频的网格贴合满意。确认状态为 `Data/runs/native_joint_s01_s04_20261027/camera_models/square_repair/review/candidate_pixels.npz`，SHA256 `c7bf576dc92a635fb254633c8fb1da1ce02f309cc42e2dba2d11fd72bf7c00c4`；输入为 569960 点／1936 曝光。该确认不扩展至后续主点模型、四镜头候选、相机物理参数或已验证米制 |
| 主点核查 | `camera_models/center_pp_square/result.json` 保存上述基线确认及居中常量主点实验；数值精修仍为非收敛停滞，相机安装／入瞳分解和广角支持存在不确定性 |
| [Ad5 矩形对照](ad5_rectangle_review.json) | S05，PTS 1273856／99.52 秒，用户意见为“看不出新旧变化”；限于该曝光的新旧矩形对照，不扩展到其他 PTS、Ad18 或源定位精度 |
| [Ad2 O 边界](ad2_o_review.json) | PTS 320000／25 秒，两个既有记录的身份与读数；原 UV 未改，`active_for_fit=false`，不代表模板或相机拟合通过 |
| [来源读取检查](source_reader_audit.json) | 历史恢复检查主要覆盖同一组 10 帧；带人工引导和已有遮挡掩膜，不证明全 S01 或全节目自动取点已恢复。后续候选接线能力见 REBUILD，完整真实节目闭环仍未验证 |
| [已撤销的 O0.1 对照](o01/result.json) | 保留原卡与失败，仅供历史追查；已有人工标注和确认范围保留 |

上述 `camera_models/...` 相对 `Data/runs/native_joint_s01_s04_20261027/`。人工认可绑定具体输入、状态与画面；优化器 accepted、数值有效、覆盖 complete 和用户验收是不同字段。

<a id="原生工程对照"></a>
## 来源与历史导航

| 对象 | 既有入口与范围 |
|---|---|
| S01 冰面与广告身份 | [冰面首批](ice_identity_review_20261026.json)、[冰面后续](ice_identity_review_20261027.json)、[Ad4 边界](ad4_boundary_identity_review_20261026.json)、[Ad6 断笔](ad6_disconnected_contour_review_20261026.json)、[Ad6 タ](ad6_ta_identity_review_20261027.json) |
| S04 广告来源候选 | [广告包索引](source_datasets/ad_points_s04_20261026/result.json)，不等于当前完整联合输入 |
| 历史四镜头输入与修订 | `Data/runs/native_joint_s01_s02_s04_s06_20261028/` 中的 manifest、`binding_review/`、冻结状态及完整性记录 |
| 六镜头 Lane 基线与相机简化 | `Data/runs/native_programme_20261028/six_shot/lane_source_refinement/`；基线与后续简化候选各有独立状态和回放 |
| 工程性能与失败记录 | `Data/runs/gpu_solver_20261029/gpu_solver_optimization.json` 及其指向的既有运行目录 |
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

逐轮读点修订、原 UV／PTS、人工意见、失败与停止证据保留在各运行原目录。旧入口原文保存在既有 `Data/maintenance/agents_audit_20261025/documentation_sources.zip` 的 `20261031/` 成员中；该本机归档不随公开仓库分发。历史状态按各自输入与源时间解释。
