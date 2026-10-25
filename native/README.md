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

G2 的 `motion_derivatives.hpp` 使用当前几何、完整相机和投影公式的局部方向导数，并在已认证 RS 根上作隐式微分；`src/gpu_linearization.cuh` 负责完整 W、按组秩归一的 Huber IRLS 和连通局部块累计。`rs_g2_attach/linearize` 沿用 G1 handle，正常调用仅回传 4 个 double 摘要；块导出及少量导数／白化探针须显式调用。固定点没有 station；共享 station 与相关组构成连通分量。G2 的 linearize 不求逆、不消元、不更新参数；实际秩亏和活动方向见节目 `g2.json`，复现入口 `scripts/evaluate_g2.py --help`。

G3 可选构建在上面的命令追加 `--cudss <项目绝对路径>/environments/cudss/0.8-cu13`。`gpu_solver.cuh` 使用现有 cuSOLVER 做带阻尼的连通局部块分解，使用 cuDSS 解同 stream 的保留 CSR；C++ 完成回代、可行更新、同目标候选评估及接受／拒绝。`rs_g3_step` 只做一个候选，拒绝不覆盖 accepted state；仅显式导出状态或诊断。已完成的真实范围和失效记录见节目 `g3.json`，不是完整 solve/resume 生命周期。

## 当前模型的 GPU 求解入口

`session.gpu_geometry.NativeSceneGPU(scene_path, core)` 只编译显式选定的当前 native 模型。冰面、连续围挡和广告使用共同几何；广告保留原材料 station 与形变分界，展开到围挡时使用原 BoardCourse。安卡拉当前有 624 条可定位路径（64 冰面、2 围挡、558 广告）；Ad15／Ad16 保留模型中的未定位状态，不补造安装。

调用 `resident(camera=..., controls=..., path_ids=..., stations=..., pts=..., observed=...)` 获得共同 GPU handle。PTS 必须是整数，先相减再换算秒；路径引用该模型。随后调用 `configure_linearization(...)` 传入活动方向、局部 station 身份及完整 W，`configure_solver(...)` 传入边界和实际可用内存，`solve(progress=...)` 在 C++ 内连续迭代，`export_state()` 导出最后接受的相机控制点与 station。几何在一次 handle 中冻结；没有读取或转换点数据集的隐含步骤。可运行的全模型调用示例见 `tests/test_gpu_geometry.py::test_all_current_paths_run_through_continuous_gpu_solver`。

`request_stop()` 可由其他线程请求安全停止；进度回调返回 `False` 也可停止。再次调用 `solve()` 保留原阻尼和接受状态继续。停止、浮点停滞和数值收敛分别报告；回调异常先停止再抛出。工程测试不表示真实节目拟合或人工验收通过。
