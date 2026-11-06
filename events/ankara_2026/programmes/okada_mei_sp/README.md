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

人体结果目前覆盖 S01 和若干动作片段，尚未完成全节目追踪或独立三维验收。[主比较入口](../../../../launchers/START_3D_COMPARISON.cmd)默认加载 `skate_s01_1103`，从原片 **22.120 秒**播放至 **44.960 秒**，共 1143 个原生曝光。用户对该版本目视较满意；接冰、开头动作保持及部分鞋刀图像误差仍未通过验证。

后续人体候选共同冻结 `camera_solver_refined_1102` 的场景和相机，使用 59.96169706678219 米／场景单位的条件尺度，该换算不构成独立米制标定。人体、骨架和鞋刀由同一状态求值；物理左右脚、屏外下肢、真实深度与接触按各自证据范围解释。

<a id="人体整合与根轨迹对照2026-11-04未验收"></a>

| 结果 | 原片范围 | 当前结论与绑定 | 检查入口 |
|---|---|---|---|
| 全 S01 连续候选 `skate_s01_1103` | 22.120–44.960 秒 | 预算停止；仍有穿冰／离冰、鞋刀图像尾部误差变差及开头姿态回退；[结果](skate_s01_1103.json)、[参数](skate_s01_1103.npz) | 主比较入口 |
| 三段动作 A/B 与根修正 | 跳跃 44.980–52.000 秒；步法 25.000–35.000 秒；旋转 76.000–95.000 秒 | 用户指出根轨迹异常；根修正伴随部分图像退化，全部未验收；[结果绑定](body_integration_1104.json) | [动作选择](../../../../launchers/START_3D_INTEGRATION.cmd) |
| 源支撑旋转候选 | 核心 88.000–92.000 秒，上下文 87.500–92.500 秒 | 局部训练图像改善，留出区域与世界刀刃轨迹仍失败；[结果绑定](spin_support_1104_audit.json)、[源区域](support_spin_1104.json) | [源支撑对照](../../../../launchers/START_3D_SUPPORT.cmd) |
| GVHMR／ExMR 相机坐标对照 | 76.000–95.000 秒，951 个显示曝光 | 全部仍有根漂移；ExMR 足细化后有 12 个鞋口截面 unknown，未得到合格最终拟合；[结果绑定](spin_models_1104_audit.json) | [模型对照](../../../../launchers/START_3D_SPIN_MODELS.cmd) |
| 提刀抓握与支撑脚参照 | 76.000–95.000 秒 | ExMR BodyNet＋HandMR、关闭 FootMR；原抓握／深度、支撑与回投失败保留；[抓握记录](spin_grasp_1104_audit.json)、[后续记录](spin_entry_1104_audit.json) | [抓握](../../../../launchers/START_3D_GRASP.cmd)、[支撑脚](../../../../launchers/START_3D_SPIN_SUPPORT_FOOT.cmd) |
| 连续入旋与质心重力候选 | 76.000–95.000 秒，951 曝光连续可调 | 入旋骨盆最大水平速度约 23.33→2.67 m/s；质心重力残差及肢体／鞋回投误差仍在，未验收；[精确阶段与失败](spin_entry_1104_audit.json) | [入旋对照](../../../../launchers/START_3D_SPIN_ENTRY.cmd) |
| S05 绝对连续状态 A/B | 76.000–95.000 秒，951 曝光，共用一条绝对曲线 | A 源约束接受23步；B 加有限腾空物理接受3步。局部腕/鞋反转改善，复核鞋误差退化，未验收；[结果与限制](spin_continuous_1105_audit.json) | [连续状态对照](../../../../launchers/START_3D_SPIN_CONTINUOUS.cmd)，默认 A |
| S05 优化后整段续算 | 76.000–95.000 秒，951 曝光，显式 v2 连续状态 | 最终46步；人体／支撑鞋回投改善，旋转漂移及释放段仍未验收；[记录](spin_continuation_1105_audit.json) | [新候选／原 A/B](../../../../launchers/START_3D_SPIN_REFINED.cmd)，默认新候选 |
| S05 第281/282帧局部修复 | 原片81.60/81.62秒；只调81.2–82.0秒内连续控制 | 14条膝踝观测关联修复，29接受步；低处腿不再短暂换号；[记录](spin_281_repair_1105_audit.json) | [当前对照入口](../../../../launchers/START_3D_SPIN_REFINED.cmd)，默认局部修复，保留上一整段候选 |

三段与旋转后续试验使用有限权重的冰面软项；未知接触与腾空不按支持状态处理。旧 `skate_s01_1103` 和 `skate_s01_integrated_1103` 不应解释为已经使用该新软项重新拟合。区域内零残差、模型参考高度和单截面数值均不能替代真实接冰验收；骨盆投影也不是经过验证的质心或实际位移。

上述首轮绝对连续候选直接优化根与局部旋转，未叠加旧高频曲线。相同台账下，88–92 秒人体图像中位误差由 gravity6 的12.62降至 A4.59像素，非提刀右腕40毫秒离群由105.34°降至6.92°。B 的短腾空质心位置残差下降，但复核支撑鞋区域误差变差，所以播放器默认 A；两个候选均因预算停止，仍有鞋刀偏差、转速波动及质量近似不确定性。最终参数为 [A](spin_continuous_source_1105.npz)／[B](spin_continuous_flight_1105.npz)；实际台账、同帧回投与完整 COM／惯量／H 留在 `Data/runs/body_s01_20261029/spin_continuous_1105/`。

完整 Joint 性能优化已实测同两接受步约 178.51→50.76 秒；它改变执行路径，未改变图像／接触目标或冰面阈值。[完整数值记录](body_performance_1104_audit.json)保留工作量、数值等价、内存、CPU 核秒和 GPU 范围。质心由当前唯一身体与鞋刀状态的分段体积质量近似计算，腾空内部约束水平匀速与竖直重力；骨盆不代替质心，阶段边界不确定性保留。

当前连续人体求解已另行接入 GPU 足鞋批处理、按需导数、稀疏装配及直接 GN 内层解法。同起点两个接受步，新完整求解35.14秒，旧 Monitor 区间205.70秒；首次 J 单独19.48→17.33秒。[本轮性能与数值记录](body_performance_1105_audit.json)区分实际计时边界和缺项；已有 A/B 与 S01 参数未覆盖，此加速不是新的轨迹准确性结论。

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


## S05 整段续算 2026-11-05

[当前候选与限制](spin_continuation_1105_audit.json)绑定 76–95 秒全部 951 个原生曝光、同一条 293 控制点连续人体状态；相机、S01 和共享足鞋形状均保持原 SHA。初始化仍为 ExMR BodyNet＋HandMR，FootMR 关闭。六轮整段续算累计 131 个接受步，最终一轮 46 步、1048.44 秒，在两小时预算内因 `ftol` 返回；最优性仍为 906.28，未达到数值收敛或准确性验收。

源图复核修订了左右关联和重复检测的准入，保存原 UV、sigma、unknown 及历史台账。11 个近冰支撑区域作为条件性冰面位置参照，保持原有限区域和权重；真实鞋刀回投与该参照分别报告。原生投影、鞋刀附着和有限软冰面仍作用于同一状态。入旋水平连续性缺口已补上，已确认的 81.36–81.46 秒启用有限质心重力与完整角动量项。旋转表示新增显式 `absolute_motion_v2`／归档 SU(2) 分支；全部曝光节点的升级初态等价，旧 v1 参数保持原解释和文件字节。

同一修订台账下，原 A 对比最终候选：

| 诊断 | 原 A | 新候选 |
|---|---:|---:|
| 全段人体回投中位数／p95 | 3.36／22.04 px | 2.94／19.92 px |
| 支撑鞋超出源区域 p95 | 14.54 px | 0.22 px |
| 刀端超出源区域 p95 | 7.93 px | 3.90 px |
| 入旋最大 20 ms 骨盆水平位移 | 10.27 cm | 7.09 cm |
| 88–92 s 支撑几何见证点 XY 跨度 | 0.739×0.335 m | 0.660×0.377 m |

最后一项并未全面改善；该见证点还会沿 rocker 迁移，不能解释为真实材料点轨迹。实际见证点源区域 p95 仍约 10.24 px，释放段 93.2–93.5 秒鞋刀朝向及小腿配置仍有明显偏差。有限腾空仅六帧、0.1 秒，弹道位置最大偏差约 2.93 cm，短时加速度仍偏离重力。以上均保留为未通过／unknown，不以像素代价下降代替接冰或物理验收。

双击 [START_3D_SPIN_REFINED.cmd](../../../../launchers/START_3D_SPIN_REFINED.cmd) 查看完整 76–95 秒，新候选为默认，原 A/B 保留在同一播放器中。三套 951 帧缓存已离线生成，支持同帧原片、自由三维、脚下／俯视与 COM／刀材料点／移动见证点分别显示。12 组关键帧原片／A／B／新候选原生回投，以及失败续算保存在 `Data/runs/body_s01_20261029/spin_continuation_1105/`；精确路径和 SHA 由上述审核记录定位。播放器模块导入、缓存和状态检查与实际打开分别记录；本轮未打开 GUI，未生成视频，未提交或推送。


### 第281/282帧的短暂换腿修复

用户指出的两帧对应原片81.60/81.62秒、PTS1044480/1044736。源图相邻帧确认81.58–81.64秒检测膝踝侧别短暂反转；原片没有换腿。14条有来源支持的观测改绑，遮挡髋/上肢及重复踝点保留unknown；不生成新UV或sigma。

在原有同一条绝对连续曲线上，仅开放16个局部控制点、1104变量，接受29步、560.47秒。81.54–81.68秒低处模型脚的短暂切换从2次变为0次，根朝向最大20ms转角44.43°→13.02°。81.2–82.0秒外的P/Q/local_R、原生关节及鞋刀状态逐项不变，派生回投仅有约1.14e-12px浮点舍入差。原A/B、上一整段C6、S01及相机参数SHA保持不变。

现有START_3D_SPIN_REFINED.cmd已备份旧字节并指向新配置，默认定位第280帧附近；上一整段C6可在同一播放器切换，两套951帧缓存及10组同源回投已准备。完整绑定、原始台账、未知项和图示由[局部修复记录](spin_281_repair_1105_audit.json)定位。本次不扩大到真实接冰或整段姿态准确性通过。

## 2026-11-05 靴轮廓／轴线局部候选

- 参数：`spin_boot_axis_entry_1105.npz`（81.2–82.0秒8步）；`spin_boot_axis_support_1105.npz`（再87.6–90.0秒8步，内部足鞋结构诊断退化保留）。
- 源数值证据：`boot_axis_evidence_1105.json`；均为有限靴轮廓／无向靴轴，钢子段没有升级为刀端材料或真实接冰。
- 播放器：`viewer3d_boot_axis_1105.json`，保留原默认local_281，比较新候选用variant `boot_axis_entry`。
- 实际结果／未解决项：`D:/CodexData/RinkSpace_Data/runs/body_s01_20261029/blade_orientation_1105/BLADE_FIX_REPORT_ZH.md`。未通过全段准确性或接冰验收。

## 2026-11-05 刀刃支撑形态修正与剩余失败

[本轮绑定与未验收记录](body_blade_1105_audit.json)汇总三候选、原生951 PTS、数据SHA和失败。旋转段增加有限的runner形态软项，仅使用既有根与支撑腿参数，固定相机、鞋模板、单位和时间；冰面仍为既有有限软项，没有持续加罚。

`spin_runner_support_1105.npz`在86.8–92.46秒接受8步、91.22秒后因步数预算停止。284个曝光内模型刀纵倾中位数−25.46°→−15.14°，前端最低226→196帧，原始最低部件为刀齿178→3帧；参考面高度中位数反而2.63→4.53cm，91.68秒源回投仍不符。最低部件与参考高度都不是实际接冰真值，候选未通过准确性或人工验收。

[专用播放器配置](viewer3d_runner_support_1105.json)保留原默认local_281；[启动器](../../../../launchers/START_3D_RUNNER_SUPPORT.cmd)显式选择新候选。新增图像数值见[靴轮廓证据](boot_axis_evidence_1105.json)，条件支撑形态见[源门控参数](runner_support_mode_1105.json)。未知材料/接触不补成真值。77、85、93秒未安全可观测的方向问题仍保留。

本轮Data中遗漏的12个项目审计／重现脚本已按[原件索引](../../../../calibration/program_tracking/body_tracking/source_helpers/blade_1105/source_index.json)保存源码；含固定历史路径和顶层写入，只供显式审核/重现，不自动运行。

## 2026-11-06 源图鞋刀完整结构核对

[完整源图描线](spin_source_skate_structures_1105.json)覆盖既有22个支撑脚曝光，每帧分别保存鞋体、鞋底、跟块、刀架、开孔、刀齿区域、完整钢刃下缘及刀尾；实线为直接描线，虚线保留局部分界或邻帧推断，尚未经过用户目视验收。[现有播放器](../../../../launchers/START_3D_SEQUENCE_BLADE.cmd)默认定位87.64秒，显示源图描线，可用“上一／下一源标注”逐帧核对；模型实际刀刃、蓝橙参考轴和旧短线段分别开关。未标注曝光不会继承邻帧描线。

`Data/runs/body_s01_20261029/body_sequence_1105/full_structure_22/review/index.html`保存22帧原片／完整描线／线稿，以及逐帧冰面坐标图和CSV。坐标仅由同帧相机、源像素与现有冰面求交，不使用模型鞋位置；整段弯刀的视线落冰足迹不等于整段真实接触。像素范围映射不包含相机、场地和尺度标定误差。本次未重跑人体拟合，原A、整段候选及失败结果均保留。

## 2026-11-06 完整源结构与支撑位置进入S05联合拟合

[本轮实际结果与失败绑定](spin_source_structure_1106_audit.json)记录76–95秒951个原生曝光。沿全段继续审阅后，[源结构](spin_source_skate_structures_1106.json)共36个曝光、40条鞋刀轮廓，其中35条有脚别关联、5条保留未分解剖身份。实际目标包含278条九类结构观测、32条条件支撑XY及转出未分左右轮廓的集合匹配；原22帧完整钢刃另加入反向覆盖，和正向项平分原权重。源范围内不吸引到中点；冰面仍为既有有限软项，不加毫米交付门槛或持续罚权。

保留30步单向诊断候选，再从其同一连续状态接受35步，按合计40分钟求解预算退出。[实际参数](spin_sequence_source_structure_1106.npz)与[播放器配置](viewer3d_sequence_source_structure_1106.json)已保存；[现有启动器](../../../../launchers/START_3D_SEQUENCE_BLADE.cmd)默认定位87.64秒，保留原A、旧14步与旧39步对照。四套951帧缓存、全951姿态／相机原生投影闭合已核对，未打开GUI或重跑网络。

同源拟合诊断中，钢刃下缘p95从31.81降至13.45px，刀尾p95从29.15降至14.61px；88–92秒支撑钢刃移动见证点XY跨度从0.654×0.382变为0.494×0.353m。仍未通过准确性验收：87.64秒刀尾偏高，92.8秒见证点仍约0.527条件米位于源回投范围外，转出五条源轮廓的左右身份未确认。移动最低点不是固定材料点或真实受力点，原始参考面高度另保留诊断。

完整双脚上下刃轨迹、原生UV、固定材料点、移动几何见证点、源观测／中间估计标记与CSV位于 `Data/runs/body_s01_20261029/body_sequence_1105/trajectory_source_1106/`；逐帧源图与冰面图见同级 `source_review_1106/index.html`，九组原片／旧／新回投在 `source_final_cards_1106/`。所有数值下降均为同源拟合诊断，不作独立留出或用户目视通过结论。
