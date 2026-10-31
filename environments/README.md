# 按能力准备环境

使用 Python 3.11 或更新环境。源码、安装环境、编译库与赛事素材分别准备；本机环境不随仓库分发，也不等于空白机器已完成重建。许可见 [LICENSE](../LICENSE.md)，资产来源见[第三方说明](../docs/THIRD_PARTY_ASSETS.md)。

| 能力 | 安装或准备 | 范围 |
|---|---|---|
| 离线审阅归档 | Python 标准库，运行 `scripts/read_offline_review.py` | 不解码视频、不执行归档代码 |
| 基础几何、运动、拼接、合成 demo | `python -m pip install -e ".[dev]"` | 基础测试见根 README |
| 高级 CPU 数值 | `python -m pip install -e ".[dev,advanced]"` | 增加 SciPy、quadprog、Clarabel；具体模块可能另需 native 库 |
| 采集 | `python -m pip install -e ".[acquisition]"` | 增加 yt-dlp，素材使用仍遵循来源条件 |
| 媒体工具 | PATH 中的 ffmpeg／ffprobe | `doctor` 报告实际路径 |
| 当前 native CPU | C++17 编译器、CMake，运行 `scripts/build_core.py` | 同一几何与相机前向；[构建说明](../native/README.md) |
| 当前 native CUDA／连续 LM | MSVC、CUDA；连续求解另需 cuDSS | 自有 C++／CUDA 路径，完整磁盘恢复尚未实现 |
| 视觉 GPU | Data 的 `runtime_tools/vision_cuda` 与 `cache/pretrained` | [原隔离验证](gpu-local-validation.json)不等于取点准确性验收；现位置见下文 |
| 人体 | Data 的 `runtime_tools/body_cuda` 与 `runtime_tools/body` | [上游与补丁身份](body-upstream-provenance.json)；完整推理及部分编译扩展仍有缺口 |
| 历史 Ceres 后端 | [rinkfit](../calibration/rinkfit/README.md)、[solve2](../calibration/solve2/native/README.md)、[原生 recipe](native-build-recipes-observed.json) | 与当前 native LM 的构建和状态合同分别解释 |
| 历史 GTSAM | [独立候选配置](gtsam-reconstruction-candidate.yml) | NumPy 1.26／GTSAM 4.2.2 栈，与主 NumPy 2.x 分开，候选未验收 |

`INSTALL_LOCAL.cmd` 安装 dev／acquisition，不含 advanced。setuptools 只发现 `src` 下的包；`calibration` 中的历史模块需要完整源码 checkout，普通 wheel 不包含全部赛事运行依赖。

## 当前本机 GPU 与人体环境

视觉解释器为 `<Data>/runtime_tools/vision_cuda/Scripts/python.exe`，人体解释器为 `<Data>/runtime_tools/body_cuda/Scripts/python.exe`，包版本分别管理。SAM2／YOLO 等通用权重位于 `<Data>/cache/pretrained`，GVHMR／PyTorch3D 与人体上游源码位于 `<Data>/runtime_tools/body`；ViTPose helper 的源码与默认加载入口为仓库内 [source_helpers/run_vitpose_h.py](../calibration/program_tracking/body_tracking/source_helpers/run_vitpose_h.py)。旧逻辑地址 `.venv-gpu`、`.venv-body`、`models/cache` 和 `environments/body` 由 AssetStore 解析到 Data，不在 OneDrive 留运行副本。Data 根使用现有存储配置，不另设环境专用盘符。

[gpu-local-validation.json](gpu-local-validation.json)保留 2026-10-25 的文件身份、包版本、当时模块／DLL 路径和本地模型加载检查；2026-10-30 的 Data 迁移记录保存在 Data 的 `maintenance/gpu_to_data_20261030`。历史清单中的路径是当时的来源地址，不是当前运行入口。系统 Python、NVIDIA 驱动和 CUDA 工具链仍是外部系统依赖；迁移不重新安装或升级它们。

FootMR、CoTracker 的固定源码子集、逐文件 SHA、FootMR 的本地 namespace 接线及 RTMW／FootMR／CoTracker 权重来源保存在 [body-runtime-sources.json](body-runtime-sources.json)。[prepare_body_sources.py](prepare_body_sources.py) 默认只检查现有文件；追加 `--fetch` 才取得缺失的固定版本源码，不覆盖已有不同字节，也不安装包或下载权重：

~~~console
python -B environments/prepare_body_sources.py --data-root <外部Data绝对路径>
python -B environments/prepare_body_sources.py --data-root <外部Data绝对路径> --fetch
~~~

该入口恢复当前选用的源码子集与接线，不表示完整人体环境已在空白机器验证。上游及权重许可分别见[第三方记录](../docs/THIRD_PARTY_ASSETS.md)。

当前连续 GPU LM 使用项目内 `environments/cudss/0.8-cu13`，身份见 [cudss-local.json](cudss-local.json)。包装脚本使用 `--cudss <目录>`，直接 CMake 使用 `-DRINKSPACE_CUDSS_ROOT=<目录>`。CUDA／cuSOLVER 来自系统 `CUDA_PATH`；本机 CUDA 13 运行 DLL 位于其 `bin/x64`。实际加载的库由各构建／运行清单绑定。

## 基础检查与完整回归

~~~console
python -m rinkspace doctor
python -m rinkspace demo --out ../rinkspace-demo
python -m pytest tests/test_geometry.py tests/test_motion.py tests/test_mosaic.py tests/test_demo.py -q
~~~

基础范围仅使用合成数据。全体 pytest 还收集高级数值、native 和本机历史实验；缺失依赖或材料应如实报告，不能算通过。工程测试不替代[真实镜头验证](../docs/VALIDATION.md)。

## 环境与构建记录

| 记录 | 适用范围 |
|---|---|
| [CPU wheel 锁](requirements-cpu-validated-windows-lock.txt)、[安装脚本](install_cpu_isolated.ps1)、[smoke](smoke_cpu_environment.py) | Windows AMD64／Python 3.11 的固定 wheel；脚本使用显式 Data 和新环境目录 |
| [CPU 重建记录](reconstruction-validation.json) | 2026-10-11 的依赖、轻量数学和 CLI 检查，不含真实视频或 GPU |
| [历史观察版本](observed-python311-windows.json)、[数值依赖子集](requirements-solver-observed.txt) | 本机历史观察，不是跨平台锁 |
| [原生 recipe](native-build-recipes-observed.json)、[资源快照](runtime-dependencies-current.json) | 历史源码、编译配置与产物身份，不是当前库的重新认证 |
| [历史原生支持](native_support_preserved/round3_legacy/README.md)、[保全清单](native_support_preserved/round3_legacy/source_preservation_manifest.json) | reference_callback／oneTBB 等历史源码与构建身份 |

独立 CPU 安装使用 `-BasePython`、`-DataRoot`、`-EnvironmentDirectory`，可选 `-OfflineWheelhouse`；其他平台需单独验证。历史重建记录绑定当时源码。

## 历史后端与资源解析

`solver_runtime` 按原源码 SHA 和 `RINKSPACE_SOLVER_NATIVE_CACHE_ROOT` 中匹配的 build.json／DLL 加载，普通导入或求解不自动编译。Brown4 等 wrapper 固定历史身份，源码变更或公开派生字节不自动满足该合同，不能绕过校验或借用旧 DLL。具体 API 见[历史运行时](../calibration/solver_runtime/README.md)。

reference_callback 需要匹配的 Python／C++ ABI、GTSAM、Eigen、Boost、pybind11 和 oneTBB。原始构建、源码与许可保留原字节，README 说明历史接口；两份历史 README 原文存于既有 Data 归档，原件与公开说明身份见保全清单。原 smoke／scaling／第三轮报告未随仓库分发，不作为当前构建前置条件。

源身份、PTS、完整状态／profile、W、物理 guard 与库身份必须匹配所选历史运行。当前 AssetStore 使用 [storage.defaults.json](../configs/storage.defaults.json)，本机覆盖见 [storage.example.json](../configs/storage.example.json) 或 `RINKSPACE_DATA_ROOT`；不要求普通 clone 拥有作者盘符。[ResourceRegistry](../docs/PORTABLE_RESOURCES.md)是独立准备工具，尚未替代活动运行路径。
