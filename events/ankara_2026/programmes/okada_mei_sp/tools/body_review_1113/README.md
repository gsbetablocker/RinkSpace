# 冈田芽依切镜与首旋拟合源码

这组脚本对应[切镜连续性](../../fit_v11/body_cut_continuity_1113.json)、[首旋修订](../../fit_v11/body_first_spin_1113.json)和[此前整鞋候选](../../skate_video_1112/whole_skate_pro_review_1113.json)。[来源清单](source_manifest.json)记录源码的原 Data 路径和 SHA-256。完整运行状态、原片、三维表面与拟合输入由节目绑定定位。

本节目人体比较和相关候选使用了 Meta 的 SAM 3D Body／DINOv3，并采用 SMPL-X 的人体表示。 [第三方来源与署名](../../../../../../docs/THIRD_PARTY_ASSETS.md)

切镜试验使用[局部求解](cut_continuity_1113/fit_cuts.py)、[表面验证](cut_continuity_1113/refine_cut_guard.py)与[全片检查](cut_continuity_1113/check_cuts.py)。首旋试验使用[模型与输入适配](first_spin_1113/spin_model.py)、[首旋求解](first_spin_1113/fit_spin_polish.py)、[两侧保留](first_spin_1113/preserve_spin_flank.py)和[全片检查](first_spin_1113/check_spin_all.py)。

本机运行使用已有的人体环境，`RINKSPACE_AUDIT_RUN` 指向含 `prepared_refined` 的 `skate_audit_cd1` 运行目录。脚本读取其同层 `audit_af48_1113` 下的源绑定和阶段参数，可通过 `RINKSPACE_OKADA_BODY_REVIEW` 指定同结构的复核目录。运行前选定与登记相符的输入；部分准备和登记脚本在执行时写入对应运行文件。使用[连续人体后端](../../../../../../calibration/program_tracking/body_tracking/trajectory_backend.py)复用完整程序的求解逻辑。
