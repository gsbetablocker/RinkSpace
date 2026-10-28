# 印刷边界诊断源码保全

这里保存 4 个仍被现有诊断链依赖、原来仅位于 Data 的项目源码文件，目的是避免核心诊断方法随运行目录丢失。原文件逐字节保留，来源、SHA-256、复制期间的文件状态和直接依赖见 [source_preservation_manifest.json](source_preservation_manifest.json)。

这些源码是历史诊断方法副本；对应 `scripts/diagnostics/` 入口仍按原绑定从 Data 加载辅助模块。保全副本未替换正式求解器，也不构成当前 native 拟合入口。

依赖顺序如下：

1. 已有 `calibration/solver_runtime/` 安装项目源码映射，并提供原模块名、数值方法和原生后端契约。
2. `verify_actual_printed_extent_candidate.py` 导入已有 `scripts/diagnostics/verify_current_native_primitives_20261007.py`，并以模块名 `structured_programme_source_candidate._data_printed_extent_candidate` 动态加载同目录的 `native_board_edge_source.py`。
3. `audit_user_left_goal_line_and_ad_edges_candidate.py` 复用上一步的状态装配函数；`existing_LOW_circle_material_owner_light.py` 复用已有求解器中的圆形、材料点、广告位置与模板算子。
4. 外层用户事实追加和审计脚本调用这些辅助方法。

本目录的 `native_board_edge_source.py` 是当时的诊断候选，SHA-256 为 `0390ff4df4f1c9188fddf9afcc7eb3d38c8ec85af4a06c829b4de7b31c238f1f`；它与 `calibration/solver_runtime/legacy/structured_programme_source_candidate/native_board_edge_source.py` 的正式版本不同，不能用它替换正式文件或更新正式源码登记。

复制保全不等于直接可运行。旧文件保留了 `D:/OneDrive/codex_projs/RinkSpace` 绝对路径、原 Data 资源地址和基于 `__file__` 的相邻资源语义；在以后切换入口时，需要分别桥接项目根、运行资源根、旧裸模块名和上述动态加载的模块名。尤其 `verify_actual_printed_extent_candidate.py` 会在导入时加载候选算子，`audit_user_left_goal_line_and_ad_edges_candidate.py` 会在导入时核验原生后端；不要通过随意导入这些副本来判断迁移成功。

原视频、源观测、完整状态、相机数组与运行产物留在 Data。本目录只提供方法源码及来源，未形成可独立运行的完整诊断包，也不表示定位准确性已验证。
