# Ceres／PHR 后端：已实现接口与本片接入合同

本包已经包含能够编译运行的双阶段约束求解内核、真实 Ceres 因子包装、不可变整状态快照、停止／序列化续跑，以及独立原生逐行预算审计。`rinkfit_synthetic_demo` 是验证这些机制的 12 行合成有限材料例；它不读取比赛资料，也不代表已完成本片求解。

本片的 182,017 行便携输入由 `rinkfit-start export-backend` 导出。下一步是把已核对的材料与原工程统计包编译为 `EventProblemBuilder`，链接现有后端，生成单独的 `rinkfit_event_fit`。准确的逐材料绑定见 `docs/event_integration.md`；本文件给出需要实现的 C++ 方法、数据和终止规则。

## 1. 交付边界

| 对象 | 已有行为 | 不能据此作出的结论 |
|---|---|---|
| `rinkfit_al` | 标准 C++17 PHR 数学、双阶段调度、原生审计、深快照与 checkpoint | 不包含比赛场景／统计分组适配 |
| `rinkfit_ceres` | Ceres LM、`SPARSE_NORMAL_CHOLESKY`、SuiteSparse；对全部一般约束使用外层 PHR | Ceres 自身支持一般不等式／等式 |
| `rinkfit_synthetic_demo` | 已知有限水平／垂直线段、真实端点、已声明非严格顺序、共享相机、所有合法 S 联合优化 | 比赛 1,248 帧已经拟合；示例的假定安装或几何是真实比赛测量 |
| `constraints.py` | NumPy 全行原生预算／profile／PHR 数学参考 | 大型物理求解器；完整物理重投影器；独立键与场景认证 |
| `EventProblemBuilder` / `rinkfit_event_fit` | 本地 Codex 应依据下面的合同完成 | 把示例改名、输出 capability 声明即完成物理接入 |

Ceres 官方只提供参数分量上下界，并不提供一般等式／不等式求解。返回 `false` 的因子用于真实算术域失败；把全部像素预算都这样处理会令不可行初值无法改进，且存在过早停止风险。[C1] 本实现把一般约束留给 PHR。

## 2. 编译与真实验证

在启动包根目录执行：

```bash
cmake -S . -B /tmp/rinkfit_start_solver_build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build /tmp/rinkfit_start_solver_build -j 3
ctest --test-dir /tmp/rinkfit_start_solver_build --output-on-failure -V
/tmp/rinkfit_start_solver_build/rinkfit_synthetic_demo --self-test
```

2026-10-17 的实际验证环境：GNU C++ 13.3.0、CMake 3.28.3、Ceres 2.2.0、Eigen 3.4、SuiteSparse 7.6.1。三个 CTest 项目全部通过：独立 AL 合同、物理公式、真实 Ceres 有限材料与停止续跑。最后一项的实测输出为：

| 数量 | 合成例结果 |
|---|---:|
| 原始／重播行数 | 12 / 12 |
| 初始实际 `t_hat` | 19.1884059957 |
| Phase II 后实际 `t_hat` | 0.0767564928301 |
| 原生统计 `F_px`，工程 sigma=1 px | 0.0707721005755 |
| 缩放投影 stationarity | 6.13731288013e-11 |
| 互补性 | 0 |
| 真实 Ceres AutoDiff 检查 | `log_f`、共享 `Cx`、`S`、`b` 四类列通过有限差分比较 |
| 同一目标的中断／保存／重新装载／续跑 | 通过；最终原生统计代价与连续求解相差 < 1e-7 |

测试还验证了等式 PHR 包装的残差／Jacobian、失败算术因子不能变为零残差、profile 法向误差很小时查询线仍可失败、原始键与完整行数检查，以及仅有低 cost 不能触发数值完成。

可以实际测试单独进程的停止与续跑：

```bash
/tmp/rinkfit_start_solver_build/rinkfit_synthetic_demo --stop-after 1 --checkpoint /tmp/fixture.chk
/tmp/rinkfit_start_solver_build/rinkfit_synthetic_demo --resume /tmp/fixture.chk --checkpoint /tmp/fixture_done.chk --ledger /tmp/fixture_native.csv
```

没有 Ceres 时，默认 CMake 配置会明确报 `Missing production dependency: Ceres >= 2.2 built with SuiteSparse`，不会选择 SciPy、Eigen sparse 或 dense solver 代替。已通过 `CMAKE_DISABLE_FIND_PACKAGE_Ceres=TRUE` 实际验证该拒绝路径。仅检查独立数学可执行：

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Icpp/include \
  cpp/src/augmented_lagrangian.cpp cpp/src/checkpoint.cpp \
  cpp/tests/test_augmented_lagrangian.cpp -o /tmp/rinkfit_al_reference_test
/tmp/rinkfit_al_reference_test
PYTHONPATH=src python -m unittest discover -s tests -p test_constraints.py -v
```

也可把 CMake 的 `RINKFIT_BUILD_PRODUCTION_BACKEND` 设为 `OFF`，只编译公式测试。生产 CMake 使用 `find_package(Ceres 2.2 COMPONENTS SuiteSparse)`，并在运行时再次检查 SuiteSparse 可用性；这是 Ceres 官方支持的组件检查方式。[C2]

## 3. 已实现的数学和阶段转换

`ConstraintSchema` 的前 N 个不等式按原始行顺序固定为：

\[
g_i=q_i-b,\qquad q_i=e_i^\top E_i^{-1}e_i,\qquad b\ge 0.
\]

普通曲线的 `e_i` 是有限合法 `S_i` 对应预测减原始 UV 的完整二维差。真实 profile 用原生像素单位的 `e_n` 与单独冻结的法向预算；统计残差 `e_n/sigma_n` 不进入预算。真实端点不滑动，裁剪／遮挡端点在没有实体端点证据时保持自由站位。原始重复 point ID 不自动共用站位。

Phase I 初始化 `b >= max(1, max_i q_i)`、`lambda=0`、`nu=0`、`rho=1`。后端自动添加无 robust loss 的 `sqrt(2*b+1)`，其 Ceres 半平方为 `b+1/2`。该残差的 GN 曲率是 LM 局部模型，不是线性目标的精确二阶 Hessian。Phase I 因子清单禁止统计项、任意权重的先验／平滑和 common mode；需要满足的物理关系应放在参数化或明确的约束里。

任一完整原生重播通过物理／材料／查询／根和单位预算，即可进入 Phase II，无需继续把最坏误差压至零。Phase II 精确设置 `b=1`，重置乘子和 `rho`，然后使用冻结统计模型、测量先验、时间平滑；所有原始约束继续保留。

两类包装残差为：

\[
r_{\mathrm{ineq}}=\frac{\max(0,\lambda+\rho g)}{\sqrt\rho},\qquad
r_{\mathrm{eq}}=\sqrt\rho\,(h+\nu/\rho).
\]

`add_phr_inequality` / `add_phr_equality` 接收一个输出真正 `g` / `h` 的标量 `ceres::CostFunction`、实际依赖的参数指针、该外层固定乘子和 `rho`。包装器拥有原因子，Ceres 拥有包装器。包装器始终挂 `nullptr` loss，原因子求值或导数失败会继续返回失败。`BuildInventory` 会检查全部 AL 标量因子、无 robust loss、因子清单完整性与命名参数／快照一致性。

乘子更新前，内核要求独立计算当前 AL 梯度达到配置精度；否则在同一个 `lambda/nu/rho` 下延续内层，默认最多两次内层预算。并不把 Ceres 的 `CONVERGENCE` 或很小的 cost 当成满足该条件。当前内层 AL 的精确梯度可按以下有效乘子调用同一导数接口：

\[
\lambda^{\rm eff}=\max(0,\lambda+\rho g),\qquad
\nu^{\rm eff}=\nu+\rho h.
\]

达到精度后，用旧乘子计算

\[
V=\max(\|\max(g,-\lambda_{old}/\rho)\|_\infty,\|h\|_\infty),
\]

再更新 `lambda=max(0,lambda_old+rho*g)` 与 `nu=nu_old+rho*h`。若 `V > 0.5*previous_V`，请求 `rho *= 5`；超过冻结上限返回 `NUMERICALLY_LIMITED`，不会无止境加罚。

最终数值检查分别包含真实原始不等式、等式、乘子非负、互补性、缩放投影拉格朗日梯度；Phase I 还检查预算变量的下界法锥。当 `b>0` 时它包含 `1-sum(pixel_lambda)`，在 `b=0` 时不错误要求乘子和恰等于 1。默认算术配置定义在 `RuntimeLimits`，由事件运行配置冻结；目前为 v1.1 的工程起点，尚不是本片实测收敛保证。

## 4. `EventProblemBuilder` 必须逐项实现的方法

公共头文件是 `cpp/include/rinkfit/augmented_lagrangian.hpp`。`synthetic_demo.cpp` 给出全部纯虚方法的一份可编译实现；事件适配应复用接口结构，把合成资料和因子替换为明确的全片生成模型。

| 方法／对象 | 事件实现的具体责任 |
|---|---|
| `identity()` | 绑定全部原始数组／键、模型／共享 shape 版本、E／profile 预算、W／covariance provenance、变量／约束尺度、目标版本的 SHA-256 身份。不要只用进程编号。 |
| `observations()` | 返回全部 182,017 个 `ObservationSpec`，顺序与 `source_rows=0..N-1` 一致；`key` 用 `(shot_id,native_pts,source_point_id)`，原始 UV 不改。普通行填冻结 SPD `E`；只有实际 profile 才填单位法向、原始有限查询区间和像素半径。 |
| `schema()` | 前 N 行严格对应像素 `q-b`；追加明确核对的非严格顺序、其他物理不等式、必要 profile／RS 等式。每条约束有稳定 ID，尺度与语义冻结，后面这些关系不乘或除以 b。 |
| `budget_parameter()` | 返回稳定的一个 double 地址；命名注册为 `budget_sq`，纳入快照；Phase II 必须恰为 1。 |
| `parameter_blocks()` | 注册全部 Ceres 参数块，包括固定块、共享几何、材料 shape、时间节点、光学、所有合法 S 与等式见证；地址和大小在一个 Ceres Problem 的生命周期内稳定。 |
| `capture_state()` | 深复制所有上述数值到 `StatePayload.parameters`；数值以外的几何拓扑、材料选择、关联身份放 `discrete`；可切换的等价根 chart 放 `root_charts`。没有真实场景含义的派生缓存不作为新自由量。 |
| `restore_state(snapshot)` | 严格验证布局与身份，然后原样复制全部块和分支；原有 vector 使用预分配存储，避免恢复时使 Ceres 指针失效。不得“顺便”clip S、改相机或修 UV。 |
| `replay_native(snapshot)` | 从传入的不可变完整状态重播全部原始行；独立求真实隐式根／查询交点、合法有限材料／顺序、正深度与光学域，输出同一顺序的原始 key、预测、station、branch、失败原因与物理状态。不得读取另一个活动状态或减 fitted common mode。 |
| `evaluate_constraints(snapshot)` | 在当前显式／隐式 chart 上重算每个真实 g/h，输出 snapshot token。未恢复的等式仍可为非零，不能换成“无交点所以零残差”。 |
| `lagrangian_gradient(snapshot,phase,lambda,nu)` | 用 fresh 完整导数计算 `grad(F_phase + lambda*g + nu*h)`，包含全部几何、时间、光学、S、aux 和统计相关性依赖；映射到实际缩放切空间／边界法锥，返回全部投影分量、尺度身份和当前 state token。 |
| `build_ceres_problem(problem,objective)` | 添加当前阶段所有需要优化的物理量、manifold、闭 bounds、基底消规范、完整原始因子及其 PHR 包装；返回真实 `BuildInventory`。phase I 的预算残差由 backend 添加，其余基底代价只在 phase II 添加。 |
| `root_chart_restart_requested()` | 因消元 root/query chart 失效时发出线程安全请求；残差求值内不修改物理状态／分支图。 |
| `rebuild_equivalent_root_chart()` | 在已接受物理状态上加入等价显式见证和等式，随后重建问题／导数。非辅助参数和 `discrete` 必须逐值不变；内核会核对，按约束 ID 保留既有乘子，新增约束从零开始。 |

`GradientReport.complete_current_derivatives=true` 是适配器的履约声明，不是用名字自证。事件集成必须对实际块作有意义的方向导数验证。对于流形，不能把 ambient 梯度直接塞进普通 box helper；应使用 manifold 的切空间 Jacobian、真实活动边界和冻结尺度。`projected_box_gradient` 只实现欧氏 box 下的 `z - projection(z-grad_z)`。

### 参数布局的首版要求

固定场地规范 `Z=0`、命名原点／长轴／手性、`L=1`。安装／轴／固定头部杠杆是整段共享参数。`alpha/beta/log_f` 采用覆盖整段和半读出边界的局部 cubic 控制块；`alpha(t_ref)=beta(t_ref)=0` 是函数值精确消元，不是盲目固定一个 spline 系数。一个共享 `tau`，不另加逐帧自由速度或独立平移修正。

每个材料的 shape 控制块保持局部、跨帧真正共享。普通站位按原始行独立，只有确有同一物质点轨迹证据才共享；真实端点固定，普通 S 的闭区间允许取端点及非严格相等顺序。有歧义／活动边界／相关组耦合的 S 直接保留在一般稀疏图中。首版没有自写非线性站位消元；也没有声称所有长期轨迹、共享 shape、完整 W 的图都是带状或 O(N)。Ceres sparse Cholesky 使用实际稀疏图与填充排序。[C3]

如需根恢复，`ParameterValues.equality_auxiliary=true` **只标记**显式 `v` 或 profile `alpha` 等同一生成方程的见证。相机、S、shape 或图像 warp 不能借此豁免“同一物理状态”检查。最终重播的 `true_root_valid` 需要真实相关非奇异根，并检查 `root_witness_error_px`；profile 还由审计器直接核对 `prediction-source_uv-alpha*n` 和原查询区间。

## 5. 读取 Python 导出的事件包

```bash
rinkfit-start export-backend --run-dir /path/to/run
```

`backend_input/manifest.json` 的 schema 为 `rinkfit.event_packet.v1`。C++ 读取前必须验证文件 SHA-256、dtype、shape、精确字节数和原始键，不通过就报告数据绑定错误。二进制均为 C 连续、little endian，无 pickle：

| manifest 数组名 | 文件 | 类型／布局 | 绑定用途 |
|---|---|---|---|
| `native_uv` | `uv.f64` | N×2 `<f8` | 原始 UV；始终在原生像素中心定义 |
| `source_rows` | `row.u64` | N `<u8` | 必须严格为 `0..N-1`，所有账本均使用此行索引 |
| `frame_index` | `frame.i32` | N `<i4` | 1..1248；time spline 支持选择 |
| `native_pts` | `pts.i64` | N `<i8` | 相对时间 `(PTS-256256)/12800` |
| `material_index` | `material.u32` | N `<u4` | `string_tables.json/material_labels` 的索引 |
| `point_id_index` | `point_id.u32` | N `<u4` | `string_tables.json/source_point_ids` 的索引；不是独立主键 |
| `engineering_budget_px` | `budget.f64` | N `<f8` | 当前普通 2D 行的冻结工程半径；`E=radius² I` |

`seeds.json/world_seeds.npz/camera_seeds.npz` 是条件初值和材料拓扑来源，不是冻结的真实三维测量或逐帧 pose 观测。建议在 Python 的事件编译步骤把已绑定场景参数、闭区间、拓扑、共享 shape 块与曲线类型导出成带 hash 的 `scene_layout.json` 和明确 dtype 的数组；C++ 不依赖旧 Python owner 的运行副作用。保存几何 seed 的来源与释放表，再共同优化确属未知的量。

**当前输入明确没有已验证 run order 和真实 profile 元数据，也没有完整 grouped W。** 因此不能从数组顺序制造顺序关系，不能给普通曲线捏造 profile，也不能拿 `source_sigma.npz` 的标量值替代原完整白化算子。前两项保持 ordinary finite-curve 语义并保留全部观测；完整 W／组成员／rank／协方差来源需要从本地现有项目适配器恢复后才能宣称执行了所要求的 Phase II。Phase I 不依赖统计目标，可以先运行可行性。缺少真实语义证据不意味着要求用户重新标注已经审核过的 UV／材料身份。

事件二进制应接受：

```text
rinkfit_event_fit --capabilities-json
rinkfit_event_fit --packet /path/to/run/backend_input/manifest.json --run-dir /path/to/run
rinkfit_event_fit --packet /path/to/run/backend_input/manifest.json --run-dir /path/to/run --resume /path/to/checkpoint
```

当前 Python launcher 核对的 capability 字段为 `schema="rinkfit.event_backend.v1"`、`backend_kind="physical_event"`、`shot_id="ankara_2026/okada_mei_sp/shot1"`、`solver="Ceres"`、`whole_shot_rows=182017`、`independent_native_replay=true`、`fixed_shared_camera_head=true`、`synthetic_demo=false`。只有真实事件适配完成后才能输出这些能力。`rinkfit_synthetic_demo --capabilities-json` 明确输出不同的 fixture schema 和 12 行，不能通过事件启动门。

## 6. 最小调用、停止与 checkpoint

在事件目标中链接现有库：

```cmake
add_executable(rinkfit_event_fit cpp/src/event_main.cpp cpp/src/event_problem_builder.cpp)
target_link_libraries(rinkfit_event_fit PRIVATE rinkfit_ceres)
```

完成上述 `EventProblemBuilder` 后的调用形态如下；包内 `FiniteLineFixture` 是它的可运行对照：

```cpp
rinkfit::CeresInnerSolver inner;
rinkfit::RuntimeLimits limits;  // 从冻结事件运行清单装载全部字段。
rinkfit::RunHooks hooks;
hooks.stop_requested = [&] { return stop_flag.load(); };
hooks.on_outer_checkpoint = [&](const auto& state, const auto& outer,
                                const auto& replay, const auto& audit,
                                const std::string& reason) {
  // 写同一个 state.token 的全部原生行账本、物理状态与数值进度。
  // 原子保存：临时文件 -> flush/fsync -> rename；由事件 runner 负责。
};
rinkfit::RunCheckpoint resume;
// 如续跑，用 read_checkpoint(stream) 读取；首次运行传 nullptr。
auto result = rinkfit::solve_two_phase(builder, inner, limits, hooks, resume_ptr);
rinkfit::write_checkpoint(stream, result.checkpoint);
```

`RunHooks.on_iteration` 在 Ceres 内部每次迭代收到最新的**整状态接受快照**、当前固定 AL 外层与同目标代价／实际下降／relative decrease／信赖域半径。要记录第一轮全数据联合 iteration 时，在该 hook 对收到的快照调用 `builder.replay_native(state)` 并写全部账本；函数返回后继续求解，不能把这一轮诊断当成完成。Ceres 的 `update_state_every_iteration=true` 是官方保证用户回调前参数已更新的开关，本实现已启用。[C4]

停止标志只负责在完整状态边界退出。Ceres 的 `USER_SUCCESS` 在这种情况下被内核明确翻译为 `STOPPED_WITH_COMPLETE_CHECKPOINT`，不会成为精度成功。中断内层的 multiplier、rho、阶段、外层序号保持原目标，续跑重新线性化同一目标。完成外层更新后的 checkpoint 则存下一外层目标，不存半更新的 lambda/rho。

快照深拥有所有参数和离散元数据，无可变数组 alias；还保存全段最佳物理状态，续跑重新独立重播它。状态 token 是确定性 FNV freshness／损坏检查，**不是密码学 SHA-256**；冻结来源／模型使用事件 manifest SHA-256，持久化 checkpoint 文件的 SHA-256 和原子提交仍由 runner 负责。重新启动必须使用同一运行限制指纹；改统计定义、预算、模型或数值限额属于新的明示配置版本，不能无记录续跑。

## 7. 原生审计和完成状态

`audit_native` 根据 `ObservationSpec` 的原始 UV／E 与 `NativeReplay` 的原生预测直接计算 q。原始行数、键顺序、state token 不一致时 `identity_complete=false`；非法投影／真实根失败会保留整行且令 q 和实际 `t_hat` 为无穷。材料、顺序、query、aux-root 一致性与物理结构分别检查，即使像素 q 很小也不会消失。

Python 的 `audit_native_budget` 使用同样原生预算语义，要求显式传入完整 `expected_rows`、投影有效、有限支持、根有效数组与物理状态。它不接受 robust weight、fitted common-mode correction 或统计 sigma 参数。原始 composite key 多重集和“预测确实来自所保存物理状态”由调用层独立验证，不能仅靠数组 q 通过。

`RunResult.status == kPrecisionReady` 只在 Phase II 完成、fresh KKT 通过且最后独立原生重播通过后返回，对外名称为 `PRECISION_READY_FOR_S6_EXPORTS`。事件 runner 仍要做 v1.1 S6 的 nuisance 重优化弱方向剖面、网格包络、全部 1,248 帧原生 overlay、逐点／帧／材料完整账本，以及四类状态分别报告。成功进程退出、单次 Ceres completion、总 cost 降低、仅有效子集通过都不是整个任务完成。

`NUMERICALLY_LIMITED`、`INNER_ACCURACY_UNFINISHED`、预算耗尽或局部 KKT 在单位预算外，只能报告当前具体状态和数值原因。所有超界原始行、失败根、活动约束和经过测试的候选都应保留；这些状态不证明全局无解。

## 一手接口依据

- [C1] Ceres 官方 Modeling FAQs，一般约束能力边界：<https://ceres-solver.readthedocs.io/latest/modeling_faqs.html>
- [C2] Ceres 官方 Installation，`find_package` 组件及版本：<https://ceres-solver.readthedocs.io/latest/installation.html#specify-ceres-components>
- [C3] Ceres 官方 Solving Non-linear Least Squares，LM／Sparse Normal Cholesky／排序：<https://ceres-solver.readthedocs.io/latest/nnls_solving.html#sparse-normal-cholesky>
- [C4] Ceres 官方迭代状态及回调语义：<https://ceres-solver.readthedocs.io/latest/nnls_solving.html#iterationcallback>

本实现的阶段、原生全量通过与停止条件以本包冻结的 v1.1 SPEC §7、§9、§10 和 `design_contract.json` 为权威；Ceres 文档用于核对实际公开 API，不作为本片已收敛的证据。
