# 原生数学核心

C++17 CPU 库及可选 CUDA 评估、线性化和连续 LM 求解，当前 ABI 2。Python 经 [core/native.py](../src/rinkspace/core/native.py) 调用同一数学；完整签名、参数布局和逐点状态以 [core_api.h](include/rinkspace/core_api.h) 为准。调用方分配数组，失败点保留状态，不钳回有效像素。

几何、实体、相机与投影分别在 `include/rinkspace/geometry.hpp`、`entities.hpp`、`camera_state.hpp`、`projection.hpp`；CPU C ABI 实现在 `src/core_api.cpp`；可选驻留 GPU 入口在 `src/core_gpu.cu`，薄绑定为 `core/gpu.py`。当前能力与 GPU／LM 缺口只在 [REBUILD](../docs/REBUILD.md) 维护，领域绑定与 Python 入口见 [INTERFACES](../docs/INTERFACES.md)。同一 handle 已支持当前原生路径的 UV 残差、完整相机／station 导数、相关组白化、连通组消元及连续 LM。接受后重算线性化，拒绝保留状态；支持安全停止及同一 handle 继续。磁盘完整 checkpoint/resume 尚未实现。

## 构建与检查

在项目根复用已有 Python、CMake 与编译工具链，输出放源码树外：

```powershell
python scripts/build_core.py --out D:\CodexData\RinkSpace_Data\build\core
```

可用 `--cmake <现有cmake.exe>` 指定工具。输出 `build.log/build.json`，后者登记实际库路径和能力；脚本不安装工具链。也可直接运行：

```powershell
cmake -S native -B D:\CodexData\RinkSpace_Data\build\core -A x64
cmake --build D:\CodexData\RinkSpace_Data\build\core --config Release
ctest --test-dir D:\CodexData\RinkSpace_Data\build\core -C Release --output-on-failure
```

原生检查验证数学与状态合同，不代替 [VALIDATION](../docs/VALIDATION.md) 的真实原片验收。尺寸和安装由调用方传入，不因前向成功而成为已验证米制。

CUDA G1／G2 使用已有 MSVC/CUDA 环境，在开发者命令环境执行：

```powershell
.venv-gpu/Scripts/python.exe scripts/build_core.py --out D:\CodexData\RinkSpace_Data\build\g1_20261025 --cuda --generator Ninja --cmake "<已有 cmake.exe>"
```

模型、相机、观测和关联由 native handle 持有，CPU/GPU 共用当前 `shared_entity_definition`、`sample_planar_primitive` 和 `project_motion`。FP64 且关闭 CUDA FMA；设备端有限数判据与 CPU 等价，host-only 调用编译警告作为错误。G1 评估不更新参数；显式附加 G3 后才可执行候选。不支持的实体返回行状态 11。复现入口 `scripts/evaluate_g1.py --help`，数据范围与结果见节目 `g1.json`。

`rs_evaluate_motion_kinematics`／`NativeCore.evaluate_motion_kinematics` 直接读取同一 native 样条的 pan、tilt、log(f) 及一、二、三阶时间导数，包含任意帧间与行曝光时刻。`motion_derivative_extrema` 返回指定连续时间段的精确速度／加速度／jerk 峰值、时间和区间；三次样条是 C2，结点处 jerk 的单侧身份由区间保留。这些诊断不平滑或改写轨迹。构建显式登记公共头文件依赖，避免本地化 MSVC include 输出未被 Ninja 解析时 CPU 使用旧数学实现。

`configure_camera_envelope` 的可选 `max_speed`、`max_acceleration`、`max_jerk` 各为 pan／tilt／log(f) 三元组，使用弧度和秒。native CPU 与 GPU 共用逐三次区间的解析极值检查，覆盖帧间和完整行曝光 halo；GPU 在同一 LM 候选中沿解析约束梯度修正可行步，并重新检查真实像素目标。没有新增源点权重、运动残差或事后滤波。None／正无穷明确表示未设限，零、负值、NaN 拒绝；历史十项静态 envelope 保持兼容。接口具备能力不等于某次运行已经选择有依据的限值，实际配置必须由运行记录确认。当前 S01 失败候选仍未应用这些动态限值，不能称为物理轨迹已修好。

G2 的 `motion_derivatives.hpp` 使用当前几何、完整相机和投影公式的局部方向导数，并在已认证 RS 根上作隐式微分。逐点像素入口 `rs_g2_attach_pixels`／`configure_pixel_linearization` 直接累计 `0.5 * sum((u-u_obs)^2+(v-v_obs)^2)`，不接收 W、sigma、协方差、稳健阈值、旧查询约束或像素容许范围；每个点的两个残差直接进入同一 GPU LM。station 的数值步长缩放只改变求解坐标，不缩放像素误差。旧相关入口 `rs_g2_attach` 保留供显式使用旧相关目标的调用者与历史回归，不是逐点入口的内部回退。

`src/gpu_linearization.cuh` 复用同一稀疏导数、块累计和消元代码；逐点入口不创建或读取白化矩阵，也不作 Huber 降权。两入口沿用 G1 handle，正常调用仅回传 4 个 double 摘要。固定点没有 station；普通曲线点沿当前有限实体寻找对应位置，各点的像素平方误差不会跨点相互抵消。真实同一材料点的共享关系须由调用者明确提供。G2 线性化本身不更新参数；旧窗口记录与复现入口 `scripts/evaluate_g2.py` 保留。

逐点模式的 `rs_g3_refine_stations`／`refine_curve_stations` 在 GPU 上按当前实体和曝光缓存曲线投影，寻找更近的位置，再用解析切向导数及真实投影作下降搜索。每个普通点只有在其自身像素平方距离下降时才更新 station；相机、场景、UV、时间、实体身份、固定角点及跨曝光共享材料点均不改变。缓存采样用于数值搜索，不是新增观测。S01 在初始状态及每次接受的联合步之后调用此步骤，并将额外的真实代价下降单独写入日志，下一次 LM 从更新后的对应重新线性化。检查见 `tests/test_gpu_pixel_residuals.py`；完整镜头准确性仍须看实际原片。

G3 可选构建在上面的命令追加 `--cudss <项目绝对路径>/environments/cudss/0.8-cu13`。`gpu_solver.cuh` 使用现有 cuSOLVER 做带阻尼的连通局部块分解，使用 cuDSS 解同 stream 的保留 CSR；C++ 完成回代、可行更新、同目标候选评估及接受／拒绝。`rs_g3_step` 只做一个候选，拒绝不覆盖 accepted state；仅显式导出状态或诊断。已完成的真实范围和失效记录见节目 `g3.json`，不是完整 solve/resume 生命周期。

独立逐点曲线的画面上下沿条件现直接进入 GPU 联合消元：触及边缘时，station 增量成为相机／场景增量的仿射函数，其耦合进入同一阻尼方程。局部有限支持、分段端点与画面边缘按当前联合步的乘子检查释放，不把一次触边永久当作固定条件。数值上保留零步可行，避免固定向内偏移制造无法随阻尼消失的上升步；最终仍以原隐式投影与原像素目标核验。共享材料点的联合身份不拆分，旧相关入口不自动套用独立点的画面边缘消元。数学对照覆盖上下沿、多个边缘的释放、曲线端点的释放，以及原有稀疏系统、停止／继续和约束回归。

## 当前模型的 GPU 求解入口

`session.gpu_geometry.NativeSceneGPU(scene_path, core)` 只编译显式选定的当前 native 模型。冰面、连续围挡和广告使用共同几何；广告保留原材料 station 与形变分界，展开到围挡时使用原 BoardCourse。安卡拉当前有 624 条可定位路径（64 冰面、2 围挡、558 广告）；Ad15／Ad16 保留模型中的未定位状态，不补造安装。

调用 `resident(camera=..., controls=..., path_ids=..., stations=..., pts=..., observed=..., scene_directions=...)` 获得共同 GPU handle。PTS 必须是整数，先相减再换算秒；路径引用该模型。`SceneParameterModel(..., optimize_scene=True)` 显式开放可用的共享场地与广告参数；不传 `scene_directions` 时几何才冻结。`configure_camera_parameters(...)` 把完整物理相机的基础 27 维局部 chart（选用原有广角 Brown 扩展时为 31 维） 接入联合方向，包括安装姿态、两转轴、安装位置、杆臂、轴间距、随变焦的入瞳与光学参数、读出时间；广角扩展使用所选结点下方的平方 hinge，并对 k1、k2、p1、p2 分别开放系数。旧 38 项 rig 保持兼容，完整扩展打包为 43 项，新增顺序为结点 log-f 与四个系数。相机安装关系和已确认范围另由约束接口传入。

`configure_linearization(...)` 传入活动方向、局部 station 身份及完整 W，`configure_solver(...)` 传入边界和实际可用内存，`solve(progress=...)` 在 C++ 内连续迭代。`export_state()` 导出最后接受的完整 rig、相机控制点、station 和共享场景数值。数据集仅提供源读点、源时间、关联和权重，没有覆盖模型或相机参数的隐含步骤。示例及回归见 `tests/test_gpu_geometry.py`、`tests/test_gpu_joint_camera.py`、`tests/test_gpu_joint_scene.py`。

原始图像查询通过 `configure_profiles(..., mode="exact")` 接入；查询交点由 native 隐式求解与求导，不能另设自由查询 station。显式 `feasibility_restoration` 阶段只用于恢复交点有效域，必须标明尚非最终查询约束目标。候选步若曲线位置越出传感器的严格预测行范围，可在同一曲线上恢复局部位置；最终仍用严格滚动快门投影、完整 W 与实际候选增量判定接受，不能删除失败点或钳改像素。

`request_stop()` 可由其他线程请求安全停止；进度回调返回 `False` 也可停止。再次调用 `solve()` 保留原阻尼和接受状态继续。停止、浮点停滞和数值收敛分别报告；回调异常先停止再抛出。工程测试不表示真实节目拟合或人工验收通过。

## 稀疏容量工程检查

`scripts/benchmark_gpu_sparse.py --library <本次CUDA/cuDSS库> --scene <当前field/model.json> --out <Data中的新运行目录> --rows 2000000 --controls 800 --group-size 16` 在同一 resident／连续 LM 入口使用完整相关组与共享 station。依赖已有 CUDA 13 运行库；保留逐次迭代、设备全局显存采样、进程工作集峰值和库哈希。原数据集不参与，目录已有运行记录时拒绝覆盖。实测规模、限制和结果见 [REBUILD](../docs/REBUILD.md#百万观测容量与停止判据修复2026-10-26)，不能把合成近真值输入的耗时当作真实节目的性能保证。
