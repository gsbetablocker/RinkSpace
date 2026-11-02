# 原生数学核心

C++17 CPU 库及可选 CUDA 评估、线性化和连续 LM 求解，当前 ABI 2。Python 经 [core/native.py](../src/rinkspace/core/native.py) 和 [core/gpu.py](../src/rinkspace/core/gpu.py) 调用同一数学。C ABI、参数布局和逐行状态以 [core_api.h](include/rinkspace/core_api.h) 为准；调用方分配数组，失败点保留状态。

领域绑定见 [INTERFACES](../docs/INTERFACES.md)，能力及缺口见 [REBUILD](../docs/REBUILD.md)。磁盘完整 checkpoint／跨进程恢复尚未实现。

## 构建与检查

在项目根使用已有 Python、CMake 与编译工具链，输出放源码树外：

~~~powershell
python scripts/build_core.py --out D:\CodexData\RinkSpace_Data\build\core
~~~

`--cmake <现有cmake.exe>` 可指定工具。脚本输出 `build.log` 和登记实际库路径、构建能力的 `build.json`，不安装工具链。直接 CMake 构建：

~~~powershell
cmake -S native -B D:\CodexData\RinkSpace_Data\build\core -A x64
cmake --build D:\CodexData\RinkSpace_Data\build\core --config Release
ctest --test-dir D:\CodexData\RinkSpace_Data\build\core -C Release --output-on-failure
~~~

CUDA 构建复用已有 MSVC／CUDA。脚本在 Windows 上先检查原生子进程能否进入项目、输出和临时目录，以及读取源码、写入输出；再按需用 `vswhere` 找到现有 `vcvars64.bat` 并载入 MSVC 环境。也可用 `--vcvars <现有vcvars64.bat>` 指定安装。CMake、Ninja 可从该开发环境发现：

~~~powershell
D:\CodexData\RinkSpace_Data\runtime_tools\vision_cuda\Scripts\python.exe scripts/build_core.py --out D:\CodexData\RinkSpace_Data\build\core_cuda --cuda --generator Ninja
~~~

连续 GPU LM 另需 cuDSS：在脚本命令追加 `--cudss <外部Data绝对路径>/runtime_tools/cudss/0.8-cu13`。直接 CMake 使用 `-DRINKSPACE_CUDSS_ROOT=<cuDSS目录>`。依赖版本与本机配置见[环境说明](../environments/README.md)。

追加 `--check-toolchain` 仅检查环境，不配置或编译；完整构建使用同一命令去掉该选项。`toolchain.json` 记录实际工具路径与版本，`build.json` 记录构建命令；重试时 `build.log` 追加保留失败日志。Windows 的编译器探测使用与产物一致的 Release 配置，仍执行真实编译和链接。

`--cuda-fast-compile min` 可显式选择较少削弱运行优化的编译路径；默认 `0` 仍保留完整优化。不同选项必须在相同输入上实测，不能按编译速度推断求解速度。

原生检查验证数学、导数与状态合同；尺寸和安装由调用方传入，前向成功不等于米制准确或真实镜头验收通过。

小型 CPU 约束对偶通过 `coupled_dual.hpp` 复用乘子未变化时的残差，并交错计算两条独立的有序归约；精确重复的浮点状态按完整周期跳过，保留原第 4096 次状态和失败判定。`rinkspace_coupled_dual` 检查新旧乘子、KKT 和返回次数逐位一致，可传入 `RINKSPACE_TRACE_DUAL_INPUT` 保存的困难样本作同输入检查。此优化不改变物理边界、精度或像素目标，局部同输入提速不代表完整拟合具有固定加速倍率。

## 共同几何与投影

CPU／GPU 共用 `geometry.hpp`、`entities.hpp`、`camera_state.hpp` 和 `projection.hpp` 中的实体、完整相机与投影。几何包含冰面、连续围挡、固定点和广告；不支持的实体返回逐行状态。

共享冰面几何选项 16（须同时启用坐标架选项 2，且采用 layout35）将两条球门线构造成 `x=±d`，两端半椭圆共用 `ax/ay`，球门区只包含有限圆弧，梯形四边共用两项半开口跨度并关于两条场地中轴镜像。CPU/GPU 使用同一构造和导数；旧选项 0–15 的参数含义保持不变。具体参数槽见 `core_api.h`，赛事启用状态由当前 scene 声明。

共享冰面选项 64（同时启用原选项 1–32）使用 layout146：前 39 项为原共同名义参数，后 107 项是相对名义量的有界偏差。尺寸允许 ±1%，位置和斜率的归一化定义见 `domain/ice_tolerance.py` 和 `ice_tolerance.hpp`。图元类型、接点、分支与严格顺序保留；206 个有限关系面在 CPU/GPU 前向、可行步和停止判据中采用同一函数及自动导数，偏差盒边界由 native 强制交集。全部偏差为零时复现选项 63。`rinkspace_ice_tolerance` 检查全部偏差与关系面导数、接线和边界，Python 回归另检查 CUDA 与有限合成拟合。

`ice_relation_meta.hpp` 为 206 个原关系维护有限类型、精确同义面和结构支持。内部修正去除 14 个重复面，原 206 面仍是最终域检查；136／147 等仅近乎平行的面不会合并。仿射偏差界可在同一 cuDSS 分解下按多 RHS 求响应，并进入允许工作面释放的小 QP。容量按真实 RHS 数和约束面数增长，计入设备预算。

`ice_actual_chart.hpp`／`ice_actual_witness.hpp` 支持八个实际 X 和 32 个实际 log-radius 的等价坐标，消去六个名义见证自由度，未固定其余几何零方向。输出仍为旧 layout146，见证必须通过原 inclusive ±1% 盒和全部关系。名义 X 使用极值区间中点，半径使用共同合法区间，逆变换采用稳定差分；失败不裁剪实际几何或放宽容差。联合 QP 的越界分离点可暂用未受界偏差表达几何，最终状态仍须原域合法。CPU 三项 `ice_relation_meta`、`ice_actual_witness`、`ice_actual_chart` 与 `tests/test_ice_actual_chart_gpu.py` 分别核验元数据、见证、导数和 CUDA 约束步；合成测试不替代真实节目验证。

历史 ±5% 冰面候选的独立源码保存在 [ice5_joint_soc](../native_experiments/ice5_joint_soc/)，差异见 [ice5_joint_soc.patch](../native_experiments/ice5_joint_soc.patch)。本节构建入口与当前核心回归使用 `native/` 中的 ±1% 实现；实验树保留供追查原候选。

隐式滚动快门求解使用同一源时钟、光学域和唯一曝光根判定。有限的画外预测仍按同一方程评估和求导，图像矩形不约束 station、相机或共享场景。图像裁切只用于显示，不能钳改残差或丢掉失败点。

`evaluate_motion_kinematics` 返回连续 pan、tilt、log(f) 及一至三阶时间导数；`motion_derivative_extrema` 求连续区间的解析峰值。运动边界可通过 `configure_camera_envelope` 配置，覆盖帧间和完整行曝光 halo。动态限值由实际运行明确给出，接口本身不提供经验证的硬件数值。

## 当前模型的 GPU 求解入口

`session.gpu_geometry.NativeSceneGPU(scene_path, core)` 编译显式选择的场景，按本次观测选择路径。广告沿原 BoardCourse 安装，保留材料 station 和形变分界；未定位广告不生成安装。当前资产库存见[场景索引](../events/ankara_2026/scene/README.md)，拟合参数快照由各次运行另行绑定。

调用 `resident(camera=..., controls=..., path_ids=..., stations=..., pts=..., observed=..., scene_directions=...)` 获得驻留 handle。PTS 必须是整数，先相减再换算秒；多机位可传 `rigs` 和逐行 `camera_ids`，分镜头运动可传显式秒坐标 `motion_times`。`SceneParameterModel(..., optimize_scene=True)` 开放共享场景方向；未传场景方向时几何冻结。

`configure_camera_parameters` 接入完整物理相机的局部方向：基础 chart 每机位 27 维，选择广角 Brown 扩展时为 31 维；多机位按 rig 顺序堆叠。安装姿态、轴、位置、杆臂、轴间距、随变焦入瞳／光学和读出时间共同进入求导。38 项基础 rig 与 43 项扩展 rig 的布局见 C ABI 和接口合同。

| 入口 | 目标 |
|---|---|
| `configure_pixel_linearization`／`rs_g2_attach_pixels` | 原始点 `0.5 * sum(du² + dv²)`，不接收 W 或 Huber，不隐含降权 |
| `configure_linearization`／`rs_g2_attach` | 显式配置的相关 W、组级 Huber 与局部连通关系 |
| `configure_profiles(mode="exact")` | 原始图像查询交点的隐式求解与求导 |

station 的数值缩放只改变坐标。固定点无 station，共享材料点保留共同身份。原始查询不能另设自由查询 station；`feasibility_restoration` 仅用于显式有效域恢复，不是最终查询目标。

`refine_curve_stations(mode="global"|"local")` 在当前实体上寻找像素距离更小的对应位置，保持原 UV、PTS、实体、相机和场景不变。可选 `profile_trial_stations` 在联合候选判定前搜索对应，仅适用于独立原始像素模式。缓存采样是搜索工具，不是新观测。 独立像素站点的局部细化以真实 native cost 做有区间保护的抛物线插值／黄金分割，减少非零残差圆弧中切线 Gauss–Newton 的往返；已达数值精度的光滑点跳过额外搜索，折角仍完整检查。有限支持、周期展开支持和固定／共享站点身份继续由原调用约定约束。

`configure_solver` 接收参数边界和可用内存；`solve(progress=...)` 在 C++ 内连续完成线性化、连通局部块消元、cuDSS 稀疏求解、可行更新及真实目标接受／拒绝。曲线有限支持和物理参数边界进入联合约束，图像边缘不进入可行域。接受后重算线性化，拒绝不覆盖接受状态。

`request_stop()` 或进度回调返回 `False` 可安全停止；再次对同一 handle 调用 `solve()` 保留阻尼和接受状态继续。`export_state()` 导出 rig、控制点、station 和共享场景，不包含完整优化器磁盘恢复状态。安全停止、浮点停滞、缺乏进展和数值收敛分别报告，数值收敛不等于人工验收。

## 性能诊断

`performance_metrics()` 返回原生累计耗时、KKT 检查／缓存命中次数、活跃面数、QP 线搜索次数、预测事件时间、候选残差缓冲大小及阻尼贡献。`solve_api_wall_ms` 包含最终证书诊断；`native_controller_wall_ms` 不含它。GPU 事件区间与主机墙钟可能重叠，不能相加。`trial_station_search_queued_wall_ms` 包含调用时尚未完成的同流工作。

原始像素候选直接累计两维残差；相关组仍用原 W／Huber，只保存 rank 个候选残差。秩二预测以 Jd 计算，角点仅刷新受影响的 station 白化列，原相关组权重完整保留。曲线搜索把时间方向先收缩到单方向 AD，保留原滚动快门根和光学域。

驻点证书按已接受状态的 GPU 线性化版本、请求精度与执行路径缓存；拒绝步可以复用，接受、曲线细化或重线性化后重新计算。证书保存在独立缓冲，试探步不能覆盖它。失败和中断不进入缓存。状态更新预编译精确非零参数映射，保持列顺序；相机按机位并行更新，约束 Gram 的各交叉积独立并行，广告最差面的归约保留原来的平局顺序。曲线精修复用刚接受候选的滚动快门投影根；同一物理机位与曝光共用相机姿态和样条变化率缓存，每次候选重建，逐点深度、光学域、全读出区间收缩界及画外根扩展仍独立检查。小型对偶在原 KKT 精度满足时立即结束。相机与广告面并行枚举后按原顺序压紧；`RINKSPACE_STATIONARITY_FACE_BACKEND=serial` 保留串行参考路径。

`RINKSPACE_STATIONARITY_BACKEND=cpu` 用原生 CPU 执行同一 FP64 锥投影，按精确非零列计算，不按阈值丢弃法向分量；几何、导数、Schur 与 cuDSS 仍在 GPU。`gpu` 保留设备参考路径；`RINKSPACE_STATIONARITY_CPU_SPARSE=0` 和 `RINKSPACE_STATIONARITY_CACHE=0` 可用于对照。可选稀疏索引不足主机预算时沿用稠密算法。

保留参数边界按列并行检查，并以原串行顺序确定有限值平局和 NaN 失败面；实际冰面仍检查全部原始关系。跨镜头连续焦段的三次样条极值按机位并行归约，每次重建候选后刷新，包含未拍到焦段内的入瞳极值；CPU 原完整扫描仍用于独立检查。围挡弧长缓存的 512 个独立积分区间在 GPU 并行计算，仍使用原 16 点积分及按区间先后顺序的前缀累加。

小对偶满足 KKT、但原始边界仍因舍入返回同一个近零负值时，保留当前非零下降候选，转入同一 Schur 度量的局部可行性精修；必要时增加向内的舍入保护量。原始物理域与像素增益检查不变。`coupled_roundoff_restorations` 与 `coupled_roundoff_inset_updates` 记录这两条路径，避免把重复同一数值切面误作需要提高全局阻尼的失败。

极小下降不会自动判为收敛。CPU 辅助驻点路径通过 `gpu_joint_seed.cuh` 复用原生 NNLS 工作集，从完整活跃锥选择联合 Schur 步的初始面；只去除零法向和精确重复，不将触及约束的坐标逐个冻结。NNLS 仅选择面，响应与乘子在实际 Schur 度量中重求，原分离器继续覆盖全部物理域。二次线搜索可使用保持同一模型下降的二阶可行性修正，最终仍核对原像素增益。普通路径已包含联合活跃面，极小预测时直接尝试投影梯度方向，避免以另一个模式编号重复相同步；该方向的二次线搜索会在导数仍为负时扩展区间，不把任意归一化倍率 1 当作步长上限。小对偶回退可使用工作集尝试中原 KKT 残差更小的可行中间解。若失败残差已落入原 FP64 有序内积的前向误差界，则仍记录为未满足原 KKT，直接转入原可行性恢复，避免在舍入噪声上消耗慢回退；`coupled_roundoff_dual_repairs` 记录此路径。只有原完整 KKT 通过时才把对偶标为求准，候选更新始终另行通过全部原物理域与原目标增益检查。恢复位于完整 station 搜索前，阻尼主导时继续原 LM 恢复。最终收敛必须通过原目标的约束梯度检查；不同坐标的梯度范数不直接比较，实际冰面坐标结果另保留旧坐标驻点检查。

## 工程检查

`scripts/evaluate_g1.py`、`evaluate_g2.py` 和 `evaluate_g3.py` 保留窗口级评估入口；历史结果按相应输入和库身份解释。

`scripts/benchmark_gpu_sparse.py --library <CUDA/cuDSS库> --scene <field/model.json> --out <Data中的新运行目录> --rows 2000000 --controls 800 --group-size 16` 使用同一驻留／连续 LM 入口检查合成稀疏容量。规模与限制见 [REBUILD](../docs/REBUILD.md#工程检查)，不作为真实节目性能或准确性保证。
