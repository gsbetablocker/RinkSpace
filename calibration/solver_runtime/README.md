# 历史 named66／v12 求解运行时

本目录保留 named66／v12 数值核心、原生内核、事件适配和冻结输入接口，属于历史后端。当前 `rebuild` 的自有 C++／CUDA 连续 LM 入口见 [native](../../native/README.md)及[实现状态](../../docs/REBUILD.md)；两者的参数图、检查点和库身份不能互换。

## 样本入口

在项目根运行 `START_FAST_SOLVER.ps1`，默认配置为 [okada_named66.json](../../configs/solver_runtime/okada_named66.json)。它描述对应历史样本，不是当前四镜头实验入口。源码与配置在项目，源包、完整状态及匹配 DLL 在 Data。

| PowerShell 参数 | 用途 |
|---|---|
| `-InputConfig` | 与驱动、源身份及参数图相符的配置 |
| `-Seed` | 完整续算检查点 |
| `-Output` | 新的 Data 结果目录 |
| `-AcceptedSteps` | 可选阶段审阅，默认 0 使用数值停止条件 |
| `-VerifyBackendsOnly` | 只核验已有原生后端 |

~~~powershell
& .\.venv\Scripts\python.exe -B -m calibration.solver_runtime status
& .\.venv\Scripts\python.exe -B -m calibration.solver_runtime verify-backends
& .\.venv\Scripts\python.exe -B -m calibration.solver_runtime named66 --output <Data中的新目录> <明确的输入参数>
~~~

named66 与 v12 各自转交原驱动，依赖对应完整输入、状态和 profile。命令可调用不等于历史候选已通过定位验收。

## 数值接口

~~~python
from calibration.solver_runtime import install, configure_native_runtime, solve_joint_source

install()
contract = configure_native_runtime(stage="before_factor_construction")
result = solve_joint_source(evaluate, q, *args, **kwargs)
~~~

调用者提供残差、完整状态、源块工厂、物理 guard 与回调。接口不选择或改写源观测、FIT／HELD、sigma、profile、物理参数图或准确性结论。其他赛事需自己的数据装配与事实，不继承安卡拉数值。

`shared_parameter_bounds` 提供已声明范围，`shared_linear_constraints(q, selected_shared)` 返回 matrix、lower 与报告。边界在同一 reduced LM 步处理，实际非线性 guard 和完整真实目标决定接受。约束返回前核对原单位可行性、非负对偶、驻点及互补性；保存的活动面索引只作工作面初值，不能跳过全约束核验。

## 事件输入与物理约束

| 驱动选项 | 对应语义 |
|---|---|
| `--camera-mount-constraint-source` | 各安装独立、同机位跨曝光共享的杆臂范围与来源 |
| `--camera-pivot-exterior-constraint-source` | 围挡外机位物理域，不由此产生额外距离测量 |
| `--reinitialize-invalid-mounts`、`--reinitialize-invalid-pivot-positions` | 显式初始化非法安装；不能声称全部曝光投影等价 |
| `--initial-pose-feasibility-source` | 绑定确切状态及观测的曝光初值修复 |
| `--independent-board-pattern-placement-source` | 事件独立广告安装，缺实测位置仍为初值 |
| `--persistent-ice-long-baseline-source` | 原始锚点像素与静止冰面跨曝光对应 |
| `--static-motion-source-role-retirement-source` | 按源证据撤回误作静止场景的屏幕图形资格 |
| `--estimate-unverified-physical-geometry` | 旧完整状态尾部的未验证场景参数 |
| `--original-motion-far-board-role-source` | 有 RGB 证据的原运动点接入有限围挡 |

初始化依次处理杆臂、围挡外支点 XY 和曝光可行性。同事实、同目标续算继承完整检查点及 damping；改变输入身份或物理域时按原合同重建相应状态。

## 资源与状态合同

`legacy/` 保留原 import／pickle 名称；`source_manifest.json` 和 `event_factories_manifest.json` 登记源码、工厂及原始散列。历史源码文件名不表示其审计 main 会自动执行。

加载先检查源码 SHA，再匹配 `RINKSPACE_SOLVER_NATIVE_CACHE_ROOT` 中 build.json 与 DLL SHA 或已登记构建。普通导入和求解不自动编译；公开派生字节可能不满足原件合同，不能绕过身份检查。事件资源由配置、`install(resource_root=...)` 或 `RINKSPACE_SOLVER_DATA_ROOT` 指定，Windows 工人先安装项目 finder 和后端合同。

`--reuse-source-jacobian` 使用滞后 J；每个状态仍更新真实残差、profile、稳健权重与 H/g，只有 fresh J 可判断数值收敛。`--adaptive-source-backtracking` 的步长和 damping 随完整检查点保存。完整 r/J 默认不落盘，显式缓存选项与实际存储策略由原驱动记录。

执行加速、并行数和缓存核验以所选配置及运行回执为准，不能由历史性能数字推断当前 native LM 的速度。停止控制读取原 `solver/control.json`；安全保存、失败和数值停止保持不同语义。历史性能和进程接口测试均不代替完整原片定位验收。
