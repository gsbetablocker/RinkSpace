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

## 开头裁切人体修复（2026-11-02）

已修复 **22.12–25.00 秒**开头窗口的横躺和大幅漂移。旧的裁切处理保留了“屏外下肢为推断”的标志，但当前全身参考继承了后面24.60秒的全身姿势及已有根修正，开头又没有低平移速度条件；只惩罚加速度允许近恒速漂移。现在恢复可靠上身参考，加入完整根朝向和下肢修正，并使用前镜头准备姿势及冰场位置。

原片 **19.960 秒**准备画面复用当前 `secondary_long_side` 安装和光学，只拟合该曝光的pan/tilt/log(f)。MEIWA蓝、绿条的轮廓误差中位数分别为0.51、0.57px；未参与该帧相机拟合的10个冰线目测点，投影距离中位数7.99px。实际native的B2与F2–R2有限线交点为scene `(0.1178174667,-0.0594002111,0)`，在当前统一尺度下约 `(7.0645,-3.5617,0)` 米。交叉腿准备姿势和交点邻域作为条件初值／软约束，未锁定精确接触点；两种物理左右脚映射候选均保留。

| 同一117帧：22.12–24.44秒 | 修复前 | 修复后 |
|---|---:|---:|
| 骨盆XY净位移 | 6.338 m | 0.181 m |
| 骨盆XY最大两点跨度 | 6.338 m | 0.224 m |
| 躯干离世界向上角P95／最大 | 82.34°／87.09° | 18.53°／19.23° |
| 同一头肩观测误差中位数／P95 | 46.57／95.55 px | 6.16／25.31 px |
| 同一手臂观测误差中位数／P95 | 391.06／731.21 px | 7.42／50.79 px |
| 每帧最低刀刃Z范围 | −3.34…+76.28 cm | −2.81…+9.11 cm |

距离为当前场景的条件米制。剩余失败：22.980秒最深穿冰2.81cm；24.320秒两侧刀刃均高于平面，最低仍为9.11cm；屏外下肢与物理左右映射仍属于推断。根朝向保留转身，没有把身体锁成静止。优化按代价停止，最终梯度最优性11.20，未宣称一阶收敛。

145个曝光进入修复窗口，25.000秒起的P/Q/局部旋转/关节/冰鞋与父版逐数组相同；原观测、源PTS、体型、骨长、共享足鞋参数、当前相机与场景均保留。最终参数完整回放误差为0。[参数与证据](opening_repair_1102.json)、[紧凑参数](opening_repair_1102.npz)连接实际结果；`Data/runs/body_s01_20261029/opening_repair_1102/results.txt`提供自动报告，准备镜头叠图和原图对照保存在同目录。

双击 [START_3D_COMPARISON.cmd](../../../../START_3D_COMPARISON.cmd)从22.120秒进入，切换“修复前：全身与冰鞋结果”和“修复后：开头站姿、位置与转身”。[播放器说明](../../../../docs/RESULTS_3D_PLAYER_ZH.md)包含复现命令。旧结果及两次中间候选保留，可随时切回。

## 足鞋关系补充（2026-11-02）

已在上述冻结场景中接入 **22 面开放鞋口、完整人体穿戴检查、全 S01 共享足鞋配准、近似内腔／足床／跟杯，以及小腿间隙和接触驱动的三个局部鞋口／鞋舌模式**。人体、下鞋、刀具及曝光内的鞋面共同使用唯一状态；仅四个鞋口顶点参与形变，下鞋和刀具保持刚性。原始鞋部观测保留，外鞋点、刀刃和人体检测踝代理按不同语义进入拟合。

这轮新增人体优化为原片 **35.20–36.80、40.00–41.50 秒，共 157 帧**；同一份足鞋配准作用于 **22.12–44.96 秒全部 1,143 帧**。其他身体帧继承下文全身候选。共享鞋长从 231.1 调到 274.0 mm，宽从 88.1 调到 109.9 mm；人体体型和骨长不变，没有逐帧滑鞋参数。

| 全 S01 自动检查 | 原配准 P95 | 新配准 P95 |
|---|---:|---:|
| 足表面采样的最大内腔坐标越界 | 43.53 mm | 3.26 mm |
| 足跟 patch 到近似跟杯锚点距离 | 60.60 mm | 9.67 mm |
| 前掌 patch 到近似支撑锚点距离 | 76.65 mm | 6.48 mm |

内部表格使用同一近似内腔定义，不是鞋内实测。当前失败也已量化：全 S01 外鞋轮廓中位误差 6.47→8.81 px、鞋头／鞋跟 3.74→16.47 px；前鞋舌模式有 325/2,286 个脚曝光达到 7 cm 位移上界，穿冰仍存在。这份候选解决了明显的足鞋装配错位，同时暴露了共享装配与现有姿态／图像观测的冲突。已有完整关节状态只覆盖 S01，没有新增全节目推理，也未把刀刃端点当作未观测的刀齿。

双击 [START_BOOT_INSPECTION.cmd](../../../../START_BOOT_INSPECTION.cmd)，在同一播放器比较“旧封闭鞋壳”“相同顶点的开放鞋口”“新穿合与局部形变”。[参数与证据绑定](boot_integration_1102.json)、[紧凑参数](boot_integration_1102.npz)和[运行说明](../../../../docs/RESULTS_3D_PLAYER_ZH.md)连接实际结果；完整报告在 `Data/runs/body_s01_20261029/boot_integration_1102/results.txt`。

## 上一轮人体短段候选（2026-11-02）

已在当前 `camera_solver_refined_1102` 冻结网格与相机上完成 S01 五段、445 帧的两组优化，并将原 J1、新观测＋旧根腿自由度、新观测＋全身／根朝向接入同一三维播放器。实际区间为原片 24.50–27.00、32.20–33.80、35.20–36.80、37.40–39.00、40.00–41.50 秒；S01 其余 698 帧保留 J1，完整节目的人体持续跟踪尚未完成。

最终参数与来源见 [body_fullbody_1102.json](body_fullbody_1102.json) 和 [紧凑控制参数](body_fullbody_1102.npz)，[播放器绑定](viewer3d_fullbody_1102.json)连接同一最新场景。当前 [START_3D_COMPARISON.cmd](../../../../START_3D_COMPARISON.cmd)从22.120秒进入，可定位36.020秒检查上一轮短段；[复现命令与证据位置](../../../../docs/RESULTS_3D_PLAYER_ZH.md)提供完整运行入口。

观测关联为可撤销覆盖层：9 点修正、29 点因源图身份／位置不清暂排除，原像素、缓存检测、源 PTS、审阅和前一轮失败均保留。唯一全身状态驱动完整表面、骨架与鞋刀，含根位置／朝向及21个局部旋转；体型、骨长和鞋刀几何固定。改善仅按同一相机、同一采用点分母报告，所有窗口均为开发材料；代价终止不等于梯度收敛或三维准确性通过，接冰和物理刀身份仍未知。

**当前不能验收为准确的三维人体动作与轨迹。** 对五段全部445帧的自动检查发现，408帧至少一侧保存的刀刃样点低于冻结冰面超过1厘米，最深约17.9厘米（原片32.500秒）；这里1厘米只是描述性计数，不是验收容差。旧自由度与全身两份拟合的骨盆三维位置差异P95约20.8厘米，主要沿相机视线方向；这是解之间的差异，不是实测误差或置信区间。播放器通过和开发点的二维误差下降均不能替代独立三维证据，不要求用户凭肉眼判断不可见的深度。

## 源与范围

| 内容 | 权威记录 |
|---|---|
| ISU 官方单节目原片、SHA、表征与历史源对应 | [primary_source_selection.json](primary_source_selection.json)；1920×1080、50 fps、time_base=1/12800 |
| S01–S09 与物理／处理区间 | [rebuild_scope.json](rebuild_scope.json) |
| 名义结构、当前场景与广告版式 | [名义结构](../../scene/field/model.json)、[当前运行场景](../../scene/field/ad4_physical_supported.json)、[版式定义](../../scene/print_layout_components.json) |
| 三个物理机位的型号及安装事实 | [用户事实](../../model_constraints/camera_hardware/user_camera_mount_and_height_20261006_v4.json)、[安装配置](../../../../configs/camera_hardware/okada_sp_installations_z190_v1.json)；型号不等于读出或畸变已标定 |

<a id="analysis-start"></a>
### 统一节目起点

当前重构拟合、网格交付与节目分析从 **原片 22.120 秒，PTS 283136** 开始，沿用原生时基 1/12800；该值与 [demo 计时窗口](outro_stats.json)一致。起点前画面、观测及结果保留；本轮按用户要求使用19.960秒准备画面作姿势和定位辅助，节目计时与处理起点不变。

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

原 UV／PTS、逐轮修订、人工意见、失败与相机报告保留在对应 Data 运行目录，由结果记录定位。旧入口与报告原文保存在既有 `Data/maintenance/agents_audit_20261025/documentation_sources.zip`，本次原文位于 `20261102/` 成员；本机归档不随公开仓库分发。历史状态按各自输入与源时间解释。
