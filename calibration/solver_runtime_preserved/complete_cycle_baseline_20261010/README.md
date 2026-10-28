# 历史完整周期 baseline 辅助版本

本目录保存历史有限差分调度辅助源码；[来源清单](source_preservation_manifest.json)登记原地址、别名、SHA、大小和依赖。原选择器 `scripts/diagnostics/start_complete_cycle_comparison_20261010.ps1` 通过 `RINKSPACE_BASELINE_FD_SOURCE` 选择该文件，要求 SHA256 `761fb8a87b612938a57533876c50f6e860106d48bb4b336a579450529262c82c`。

该副本属于[历史 solver_runtime](../../solver_runtime/README.md)，不是当前 native LM 实现。相对导入需要 `calibration.solver_runtime._complete_cycle_baseline_FD` 模块身份，依赖既有 `absolute_geometry_fd_pool_candidate` 和 NumPy。

历史对照还需要对应完整检查点、truth-restoration 状态、源观测、已认证后端与环境。要求外部 `--source-replay-module` 的入口必须使用匹配模块及邻接绑定 JSON；保存方法源码不表示这些运行资源已经齐备。原执行结果与额外资源核对事实留在来源清单中。
