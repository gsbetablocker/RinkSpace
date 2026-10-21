# 可复用求解器接口

本目录保存项目拥有的数值核心、原生内核与现有事件适配器。它服务 [通用项目目标](../../README.md)。通用接口可复用，已有 named66／v12 驱动仍绑定安卡拉数据与参数图，不能直接套用到所有新赛事。

## 现有样本入口

在项目根目录运行 START_FAST_SOLVER.ps1，默认读取 [okada_named66.json](../../configs/solver_runtime/okada_named66.json)。源码和启动配置在项目；实际源包、完整检查点与已认证 DLL 在 Data。

| PowerShell 参数 | 用途 |
|---|---|
| -InputConfig | 显式选择与驱动、源身份和参数图相符的输入配置 |
| -Seed | 覆盖完整续算检查点 |
| -Output | 指定新的 Data 结果目录，保留已有结果 |
| -AcceptedSteps | 可选阶段审阅参数；默认 0 使用求解器自己的收敛条件 |
| -VerifyBackendsOnly | 只核验已准备的原生后端 |

```powershell
& .\.venv\Scripts\python.exe -B -m calibration.solver_runtime status
& .\.venv\Scripts\python.exe -B -m calibration.solver_runtime verify-backends
& .\.venv\Scripts\python.exe -B -m calibration.solver_runtime named66 --output <Data中的新目录> <明确的输入参数>
```

named66 转交现有完整节目驱动参数；v12 保留另一套历史驱动，两者的参数图与检查点不能互换。输入配置、事件图和检查点必须对应。

## 通用数值接口

```python
from calibration.solver_runtime import install, configure_native_runtime, solve_joint_source

install()  # 早于旧模块名的 factor/source/driver 导入
contract = configure_native_runtime(stage="before_factor_construction")
result = solve_joint_source(evaluate, q, *args, **kwargs)
```

调用者提供实际残差、完整状态、源块工厂、物理 guard 与回调。接口不选择或改写源观测、FIT/HELD、sigma、profile、物理参数图或准确性判定。新赛事需要对应的数据装配与证据契约，不复制公共内核，也不借用安卡拉数值。

shared_parameter_bounds 给出已声明共享参数的范围；shared_linear_constraints(q, selected_shared) 返回 matrix、lower 和小型报告。它们在同一 reduced LM 步中处理边界，不事后裁剪相机、不自行添加先验；实际非线性 guard 与完整真实目标决定接受。

约束解先在当前工作面集合上分解，再检查所有原始约束；违反的原始面继续加入，只有完整原单位的可行性、非负对偶、驻点和互补校验全部通过才返回。活动面行编号可跨进程保留，用于缩短下次寻找工作面的过程；不保存旧步长、对偶或矩阵，错误或过期索引不能跳过全行校验。数值异常或未通过校验时保留完整 IPM 路径。

启用 source_stage_timing 后，日志逐次记录约束构建、全部约束求解、曲率测量、试步检查和保存耗时，包括失败调用。整体性能使用 actual_accepted_cycle_performance.json 的接受状态间墙钟；末次 QP 的 seconds 不代表整步。


named66 正式入口在构图前安装 execution_acceleration.py，并在对应 `.bootstrap/execution_acceleration_receipt.json` 记录每个执行辅助文件、原函数与变换版本的散列。六个驻留残差工人同时运行各自原有的有限范围检查，失败按原块顺序报告并排空回复。原 truth-restoration 必要条件可提前拒绝试步；任何接受仍运行完整物理检查、源代价与原接受回调。wide 人工顶线查询复用同一几何 producer 和场地合法性检查，省去无关曲线和整场景深拷贝。使用封存完整缓存时，还逐块核对当前并行残差、权重、尺度和行身份的严格字节一致性。

可选曲率修正已有通过完整原 KKT 的方向时，若更紧的修正子问题的原受限求解路径未能提供通过完整 KKT 的解，结束这次可选修正并将上一合法方向交回原线搜索。该路径覆盖原求解器的失败状态、数值异常及完整 KKT 失败，保留具体原因；不宣称数学不可行。初始 QP、没有已验证方向，以及所有实际返回解的原精度与 KKT 门槛，均保留既有处理。三处 QP 后端调用分别记录阶段、行数、状态、迭代数与墙钟。执行辅助模块不改写封存缓存或 registered producer 文件；每次运行保存辅助源码快照，新版本以真实完整周期核验。

正式入口的 QP 在 Clarabel 迭代回调中读取原 `solver/control.json`。有效的 `stop_branch: true` 作为独立控制信号回到原保存、结果与进程清理流程，不被数值回退吞掉；没有新增自动超时或迭代上限。若显式请求 `save_current_linearization_cache: true`，则在当前参数和冻结 profile 上回到原 fresh J 边界封存缓存后结束。

fresh J 前释放闲置残差进程，严格核对冻结 profile 字节保持一致；原父进程与几何有限差分路径完成并释放 FD 池后，后续原残差或检查命令再启动驻留进程。相机导出继续调用原导出函数，首次在当前已建图上对比全部输出字段，完全一致后才改用原 camera-only accessor，避免仅导出相机时同步整套运动几何。

几何有限差分任务完成后立即补充空闲工人的任务，结果仍按原组顺序交付，保留原步长、差分模板和异常顺序。私有材料快照移除四个指向完整运动图的非必需反向引用，原父对象与源记录保持不变。滞后 J 遇到原零导数行激活时，已认证的驻留残差接口可调用原父对象的精确行刷新；该局部刷新使用父进程有限差分，避免残差进程与全量 FD 池同时占用内存。不支持的接口仍走原有回退。

同一当前世界的材料差分样本由现有进程池各计算一次，供工人只读共享；逐行绑定、资格检查和差分模板仍执行。人工约束以及冰线、广告边缘的独立导数列也复用该池，保留原冰线关联、光学支持分支及逐列数学错误。同一节目调用中，镜头与安装参数复用输入字节完全相同的相机15列导数。源角色初始化在每个原 NPZ 打开上下文内只读取一次数组成员，保留全部逐行源事实检查。默认不再为每组首个 J 自动运行详细 cProfile；需要时通过 RINKSPACE_SOLVER_PROFILE=1 显式启用。这些是执行方式，性能以对应完整周期及冷启动至保存的实际墙钟为准。

同目标 fresh J 核验先认证封存源、完整参数与冻结 profile，再重新计算全部源块；逐块比较 r、mask、scale 与 CSR 的 dtype、shape 和字节，新块进入原法方程累积器。J 阶段耗时、热周期和冷启动至保存须分开报告。

## 事件输入与物理约束

| 驱动参数 | 现有作用与边界 |
|---|---|
| --camera-mount-constraint-source | 恢复各安装独立、同机位跨曝光共享的杆臂工作范围与来源 |
| --camera-pivot-exterior-constraint-source | 接入用户确认的围挡外机位物理域；无实测距离时不新增“靠近”阈值 |
| --reinitialize-invalid-mounts | 仅显式要求时初始化不合法杆臂；参考曝光光心保留不等于所有曝光投影等价 |
| --reinitialize-invalid-pivot-positions | 仅显式要求时初始化不合法支点 XY；保留其他参数后仍需重评真实投影 |
| --initial-pose-feasibility-source | 用绑定确切状态与原始观测的初值修复不可行曝光；不放宽曲线范围或改源点 |
| --independent-board-pattern-placement-source | 现有事件中保留印刷模板、独立估计 Ad25 station/Z；无实际源的广告位置仍是初始化 |
| --persistent-ice-long-baseline-source | 用原始锚点像素与当前相机建立静止冰面跨曝光对应，避免把旧模型生成的世界点当真值 |
| --static-motion-source-role-retirement-source | 按冻结源证据纠正被误当静止场景的屏幕图形；保留原始记录与行布局，退休不等于贴合正确 |
| --estimate-unverified-physical-geometry | 保留旧完整状态索引，尾部扩展未验证广告位置、真实源印刷模板的尺寸和位置、独立 Lane 右延伸；求解和导出使用同一材料模型 |
| --original-motion-far-board-role-source | 将有 RGB 证据的原运动点接到当前有限围挡；不增点、不改 sigma，不把旧未知深度当作场地事实 |

初始化顺序为杆臂、围挡外支点 XY、原曝光可行性初值。改变物理域、源角色或初值时按既有策略重建对应数值状态；相同事实与目标的后续续算继承完整检查点及 damping，避免每轮重复初始化。源行、曝光身份、权重与角色变更均可追溯。

## 代码、后端与数据位置

legacy/ 保留实际驱动需要的源码及原 import/pickle 模块名，历史文件名不意味着其审计 main() 会被运行。source_manifest.json 与 event_factories_manifest.json 登记项目拥有的源码、原始来源和散列；事件工厂加载映射到项目代码，证据包和数组继续读取 Data。

所有 C/CUDA 内核源码在项目。加载先校验源码 SHA，再匹配 RINKSPACE_SOLVER_NATIVE_CACHE_ROOT 下的 build.json 与 DLL SHA，或使用登记的现有 Data 构建；导入和普通求解不自行编译。内核源码改变后需显式构建匹配产物，不能用旧 DLL 代替。

默认事件资源根由输入配置给出。需要同结构资源时，在第一次安装前传入 install(resource_root=...) 或设置 RINKSPACE_SOLVER_DATA_ROOT；源身份与完整状态认证继续生效。Windows 工人使用 worker_entry.py，在反序列化 legacy 目标前安装项目 finder 与后端契约，避免落回 Data Python。

## 续算、并行与存储

现有冈田配置使用一份内存 J 跨步复用、六个驻留残差/profile 工人、八个几何有限差分工人、四个广告因子工人与八线程 Gram。具体值以 [输入配置](../../configs/solver_runtime/okada_named66.json) 为准，属于已有事件执行配置，不是所有赛事的固定要求。

--reuse-source-jacobian 使用 modified Newton：每个新状态仍更新 profile、真实残差、稳健权重与 H/g；旧 J 只是滞后的导数模型。预测失效、试步失败或接近收敛时重建 J，只有新 J 能判断数值收敛。--adaptive-source-backtracking 继承成功步长，完整检查点保存 next_line_search_fraction 与 damping。

--serial-source --geometry-fd-workers 8 使用轻量进程池和原始步长、源行顺序；--normal-workers 8 并行计算 CSR 行块 Gram，浮点累加顺序可能存在舍入差异。默认取消固定内存预留导致的退池；RINKSPACE_SOLVER_MEMORY_RESERVE_BYTES 可显式启用。

完整 r/J 和法方程默认不落盘，只保存完整数值检查点与小型记录。--save-initial-linearization-cache 显式保存初始完整缓存；实际策略写入 matrix_storage_policy.json，入口不自动删除历史缓存。检查点先于迭代日志保存；Windows 日志原子替换的临时占用会有限重试，失败保留材料。

## 性能与定位的证据边界

历史 v8 对照的 191.546 秒 → 17.200 秒、约 11.14 倍，针对双方复用已认证同状态完整 J/法方程的热求解周期；冷准备与首次预热另计。该倍率不能套用到新状态重建 J、跨步复用或其他赛事，也不代表定位准确。

test-transport 是实际 Windows 小型进程接口检查，会在 Data 生成测试记录；它不处理真实节目，也不验证定位或替代完整同状态性能对照。额外性能诊断由 RINKSPACE_SOLVER_PROFILE=1 显式启用。
