# 安卡拉四节目场景复核

[2026-11-19 的赛事指定](current_fit.json)选择 [`s14_boards_1118`](s14_boards_1118.json)，使用 [`lane4_side_1118` 场地数值](field/lane4_side_1118.npz)及[配套结构](field/crease_offset_1118.json)。各节目范围、相机与本地回放见[赛事入口](../README.md#暂定最终版本)。

金书贤 S14 的相机使用同帧印刷边、围挡上沿和接冰线约束。原片 122.500 秒的 12 个远侧广告观测，像素绝对误差 P90 由 10.94 降至 1.88 px。123.700 秒的 161 个近侧广告观测，P90 由 12.46 降至 3.99 px。这些统计描述相应观测的回投。金书贤和 Nayeon Ko 共七个镜头更新了取点。

联合登记中的 `specific_source_evidence`、`four_programme_checks` 与各节目 `full_review` 保存复核范围、证据及回放身份。整片用户视觉验收、独立物理相机标定及实地尺度验证待完成。[验证说明](../../../docs/VALIDATION.md)

## 复核源码

[四节目拟合和复核工具](../tools/joint_refine_1117/)读取所选 native 库和 Data 中的预处理源观测。[S14 运动复核](../tools/joint_refine_1117/evaluate_s14_motion.py)使用该版与前版的连续状态。[C0 漆线读数原始脚本](../tools/joint_refine_1117/s14_circle_source/)依赖 `RS_CIRCLE_AUDIT` 指定的原运行目录，其中包含源像素、读数与检查表。脚本用于相符输入的技术复核，部分会写入 Data 结果。
