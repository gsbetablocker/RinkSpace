# solve2 历史 Ceres 批量桥

本目录保留 solve2 的 Ceres 批量桥，物理模型、投影、源时钟、实体、联合噪声和约束由对应 rinksolve 上游负责。它属于历史后端，与当前 [native C++／CUDA LM](../../../native/README.md)使用不同的布局、状态和构建合同。

## 构建与加载

从项目根执行：

~~~powershell
& calibration\solve2\native\build.ps1
~~~

公共依赖为项目 environments/cg22p 下的 Ceres 2.2.0 和 Data/runtime_tools/rf_deps 下的 Eigen、glog、gflags。MSVC 2022 14.44 编译。源码留项目；二进制、响应文件、构建日志和清单留 Data/runs/solve2_native。源码或构建入口变化产生新目录，不覆盖已加载 DLL。普通运行只加载所选构建清单与对应数据，不重放旧生产链，也不自动编译。

Python 入口为 rinksolve.native_backend。NativeBackend(build_manifest) 加载明确的桥及公共依赖。成功加载只证明 ABI 可用；没有宣称模型拟合合格。

## 最低可运行例子

~~~python
import numpy as np
from rinksolve.native_backend import NativeBackend, BatchLayout, SolveOptions

backend = NativeBackend(build_manifest)
layout = BatchLayout(
    block_ids=("x", "y"), block_sizes=(1, 1),
    factor_ids=("sum", "product"), residual_sizes=(1, 1),
    factor_blocks=((0, 1), (0, 1)),
)

def evaluate(x, residual_out, jacobian_out, new_point):
    a, b = x
    residual_out[:] = [a + b - 5, a * b - 6]
    if jacobian_out is not None:
        jacobian_out[:] = [1, 1, b, a]

result = backend.solve(
    layout, np.array([1., 4.]), evaluate,
    options=SolveOptions(linear_solver="sparse_normal_cholesky"),
)
assert result.usable
# result.parameters 为独立只读新数组，约为 [2., 3.]。
~~~

## 稀疏数据与 rinksolve 模型对接

1. model.pack_state(state, selected_block_ids) 返回数值向量和 ParameterView；每个 native 块保留 id、semantic_block_id、component_offset、size、vector_offset 和 owner。cubic controls 自动拆成每块三值，不能把 3,756 维整块接到每个像素。
2. BatchLayout.block_ids/block_sizes 直接取这些 view 的 id/size；factor_blocks 列出各 factor 真正涉及的 native 块索引。交点、共享广告安装与固定 station 的身份不能在桥里改成逐帧变量。
3. 每 factor 的 residual_size 是完成正确联合白化后的实际残差维数。联合 W / Query 的相依性由上游保留；桥不拆成独立的像素损失，也不增加鲁棒核。
4. evaluate 每次参数点只被调用一次，覆盖所有 factor。先 model.unpack_state 原状态和这些 view，再构造同一新 ModelBundle/ProjectionOperator；GPU kernel 可以完成全批前向和解析导数。
5. residual_out 按 factor 顺序拼接；Jacobian 按 factor → factor_blocks 的列出顺序 → row-major [residual_size, block_size] 拼接。layout.jacobian_offsets 和 jacobian_block 提供精确视图。即使某块本次为 constant，当前 ABI 仍要求写全布局，避免模式隐式变化。
6. 源数组始终不改。返回参数经 model.unpack_state 生成新不可变修订；没有通过 owner/尺寸/范围核对的向量不能写回模型。

## 运行、线程与错误语义

Ceres EvaluationCallback 每个参数点先做一次批量计算，各 CostFunction 在 C++ 工作线程仅复制对应残差/Jacobian。Ceres 原生线程完成稀疏装配和数值求解，避免逐 factor Python/GIL 回调。num_threads 显式可配置。同一 NativeBackend 实例串行化 solve；各运行拥有独立 callback、数组、layout 和结果缓存。回调中递归调用同一实例会立即报错，避免重入锁造成死等。

回调参数是借用只读视图，输出是借用可写视图。它们只在本次回调期间有效。GPU 异步任务必须在返回前同步，不能保留这些指针供随后写入。new_point=False 可用于复用同一参数点的计算；残差请求与 Jacobian 请求仍须完整写出各自输出。

输出预填 NaN。任何未填值、非有限结果或返回 False 均作为该参数点计算失败交给 Ceres；不得在回调里静默丢行、改观测或夹取像素。参数边界检查、constant 标志及精确 int32 布局均在进入 native 前核对。Python 回调异常会记录 callback_errors，绝不跨 C ABI 抛出。调用者必须检查 result.usable、termination、callback_failures、callback_errors 及物理验证结果。

on_iteration(event, current_parameters) 每次收到 Ceres 迭代结束时的当前状态；iteration=0 标明 initial_state，不能计为已取得一次实际改善。回调可将接受状态原子保存为新 checkpoint。返回 0/None 继续、+1 表示调用者主动满足停止条件、-1 中止。日志路径、暂停/停止状态及持久化由上层运行控制统一处理。

当前这是非线性最小二乘桥，不是通用等式/不等式约束求解器。硬等式必须由明确可行参数化或外层约束算法实现，不能私自以大权重残差代替。Schur ordering 的首个非空自由变量组必须独立：每 factor 最多连接其中一个块。联合 W factor 若同时连接多个 station/row 辅助量，不能假装这些变量独立消元。

## 本机已实际验证

Data/runs/solve2_native/verification.json 记录实际非线性、稀疏 Schur、边界/常量、失败处理、当前状态回调、布局/精确类型与线程隔离测试。

`Data/runs/solve2_native/scale_benchmark.json` 保存已知解合成问题的规模和性能；未计入真实投影／联合噪声成本，也未使用 GPU，不能推断真实节目速度或验收状态。


## 显式约束外层

rinksolve.constrained 使用 PHR augmented Lagrangian。ConstraintFamily 声明 equality(h=0) 或 less_equal(g<=0)、同量纲 unit、正 normalization_scale、scale_source、authority_refs 和 family 的 factor_indices；约束值、Jacobian、原 E 及其来源由调用者负责。工程 E 不是统计 sigma。目标与约束同 native parameter views，参数各分量也要显式 parameter_scales。

q=raw/scale；等式采用 sqrt(rho)*(q+lambda/rho)，不等式采用 sqrt(rho)*max(0,q+lambda/rho)。不等式 lambda 始终非负。每个家族的有限 penalty、增长规则、上限、可行性/站立性/互补性阈值、inner 站立性阈值和 inner Ceres 参数均由 ALOptions 明确给出，没有硬编码物理默认值。

完整 ALState 包含当前 x、归一化乘子、逐家族 rho、上轮 shifted violation、outer index、所有输入/模型/约束 identity、数值 policy、实际 inner 结果和变化历史。state.save 在 Data 写新 NPZ generation 后原子更换 manifest；load 只读当前完整 checkpoint，不读旧生产/历史审计。无约束时仍保存完整结构，空 multiplier/penalty 明确为零长度。

inner 达到 maxiter 或浮点 function tolerance 均不能代替实际 projected augmented gradient 检查；未达到声明 inner 精度则返回 inner_incomplete，保留最新接受 x 和原 lambda/rho，可继续同一个子问题。惩罚到有限上限且不可行时返回 penalty_limit_requires_review，不能无限加权或称为完成。外层 chunk 耗尽同样只是可续状态。诊断同时报告原单位 violation、归一化 feasibility、projected KKT stationarity、dual sign、complementarity 与排除 initial_state 后的 actual accepted steps。numerically_feasible_stationary 只代表当前数学问题通过声明数值标准，不能代替完整原片检验。

`Data/runs/solve2_native/al_verification.json` 与 `al_test_output.txt` 保存已知 KKT、单位变换、原子续算、停止与失败恢复检查。早期失败保存在同目录原记录中；数值通过只针对所声明合同，不改变原片误差预算或物理约束。

Explicit numerical unit metric: `NativeBackend.solve(parameter_scales=S)` uses the additive full-rank chart `Plus(x,d)=x+S*d`, `Minus(y,x)=(y-x)/S` (same ambient/tangent size), Ceres Jacobi scaling disabled and both LM diagonal limits fixed to one. Residual/Jacobian callbacks and bounds remain in original ambient units; no residual, prior, or frozen degree of freedom is added. The old `rs_batch_solve` ABI retains public Ceres defaults; the new `rs_batch_solve_scaled` export requires explicit finite positive scales. Runtime reports the actual chart ABI, metric flags, and scales hash.
