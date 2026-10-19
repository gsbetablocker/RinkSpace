# 既有完整周期对照的 baseline 执行辅助版本

这里保全一份 12,914 字节的历史有限差分调度辅助源码。现有 `scripts/diagnostics/start_complete_cycle_comparison_20261010.ps1` 明确选择它，`calibration/solver_runtime/execution_acceleration.py` 的 baseline 分支通过 `RINKSPACE_BASELINE_FD_SOURCE` 加载它，并要求 SHA-256 为 `761fb8a87b612938a57533876c50f6e860106d48bb4b336a579450529262c82c`。当前维护版本的源码散列不同，因此保留这个被明确选择的版本有助于读懂和重建历史对照。

本次只复制这一小文件，没有复制整个 `.bootstrap/execution_sources/` 快照。来源、别名、SHA、大小、复制期间的文件状态和直接依赖记录在 [source_preservation_manifest.json](source_preservation_manifest.json)。副本逐字节匹配稳定原件及现有选择器要求的散列。

`runtime_activated=false`。原 selector、环境选择、Data 原件、求解器源码、配置、profile 和来源绑定均未改。副本不能当作当前维护实现；其相对导入需要原 `calibration.solver_runtime._complete_cycle_baseline_FD` 模块身份。它依赖已有项目的 `absolute_geometry_fd_pool_candidate` 和 NumPy。保全源码不批准新计算，也不证明历史比较能够在今天的其余代码版本上直接重现。

实际重放还需要对应完整检查点、truth-restoration 状态、源观测、已认证原生后端及环境；它们仍属于 Data 资源。需要外部 `--source-replay-module` 的诊断入口仍必须由调用者明确提供实际回放模块和该模块邻接的绑定 JSON。本次未推断或替换这些外部输入，未读取大数组或视频、哈希大型资源、运行模块、测试、求解或编译。

顺带核实的 `D:/CodexData/RinkSpace_Data/station_columns_runtime_draft_20261012/finite_rectangle_station_columns_runtime.py` 已与项目维护版本逐字节相同（19,565 字节，SHA-256 `69e7139c583f90eda4a414b273a0a6bfd360a2460ee9aebe67d1906f79d1e9d0`）。该结论记入来源清单，本次没有再复制它。

保全这些执行方法不改变源点资格：三维场地拟合仍只允许原片核实的冰面或围挡本体静止实体与材料，归属待核点继续排除。
