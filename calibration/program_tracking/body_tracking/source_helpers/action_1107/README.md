# 1107 全节目人体与求解复核源码

这里保留实际运行的关键脚本。源文件地址、原始与发布哈希、差异基线见 [source_index.json](source_index.json)。入口 [trajectory_backend.py](../../trajectory_backend.py) 负责正常求解。

- [complete_actions_v2.py](complete_actions_v2.py) 是使用真实足腔导数的全节目阶段。首轮、网格专项与最后的 S09 阶段以对该文件的文本差异保存。还原使用 Git 的零上下文补丁模式（git apply --unidiff-zero），输入和预期输出的 SHA 见来源索引。
- [refit_axel_axis.py](refit_axel_axis.py) 与 [verify_axel_axis.py](verify_axel_axis.py) 记录 2A 的旋转分支选择、落冰后右刀保护和全节目复算。
- [compare_actual_jacobians.py](compare_actual_jacobians.py) 对照相同检查点的实际加权 CSR、残差和方向乘积。[verify_real_fused_foot.py](verify_real_fused_foot.py) 对照相同足鞋的分开与融合线性化。

- [source_review_frames.py](source_review_frames.py)、[assemble_source_ledger.py](assemble_source_ledger.py) 与 [prepare_source_revision.py](prepare_source_revision.py) 保留源曝光核对和早期台账装配过程。[axis_review.py](axis_review.py) 与 [summarize_action_result.py](summarize_action_result.py) 保留轴线与最终分组结果的计算。
- [run_trajectory_measured.py](run_trajectory_measured.py) 和 [programme_perf_measurements.py](programme_perf_measurements.py) 保留同输入全节目 A/B 的显存采样和计时汇总。

历史运行脚本含当时的 Data 路径与写出位置。复现时在独立 Data 目录使用相符的准备输入、检查点和预算，切勿在源码目录直接运行 2A 脚本。结果、日志和冻结输入由节目绑定定位。
