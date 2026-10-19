# 环境、原生依赖与重建边界

2026-10-13 的补充保全、现存外部依赖及本次仅查元数据的边界见 [运行依赖说明](../docs/RUNTIME_DEPENDENCIES.md)；新增历史构建小文件的逐项来源见 [原生支持保全清单](native_support_preserved/round3_legacy/source_preservation_manifest.json)。下文保留 2026-10-11 的观察与独立 CPU 验收，不代表后续完整运行已验证。

返回项目唯一入口：[START_HERE.html](../START_HERE.html)。本目录记录 2026-10-11 现有运行环境、独立重建结果及缺口，服务代码保全与后续公开仓库。初始快照只读取现有解释器元数据、源码、构建清单和 DLL 字节；本轮另在全新 Data 目录安装 CPU 依赖并做轻量验收。现有 `.venv`、活动配置与隔壁任务的模型和输入均未修改，未执行原生编译、赛事求解、视频处理或 GPU 工作。

## OneDrive 应保存什么

源码、依赖定义、构建方法、必要配置和小型参数是可保全的项目资产。现有 `.venv/` 是本机运行安装，内含绝对解释器路径及平台二进制，不适合直接作为 Git 环境交付；它继续保留在原位供现有任务使用。Git 仓库保存环境定义，其他机器依定义创建自己的环境。原片、批量帧图、大数组、完整检查点、编译产物与日志仍属于 Data，不能把它们缺失误报为源码缺失。

本目录新增内容均不改变现有 `pyproject.toml`、启动入口、求解器或活动配置：

| 文件 | 内容与用途 |
|---|---|
| [observed-python311-windows.json](observed-python311-windows.json) | 四个现有环境的完整已安装包名与版本、解释器身份及观察到的差异 |
| [requirements-solver-observed.txt](requirements-solver-observed.txt) | 主环境中基础视频/几何与 advanced 数值依赖的观察版本子集 |
| [native-build-recipes-observed.json](native-build-recipes-observed.json) | 当前项目原生源码与现有 Data DLL 的身份、真实编译参数及重建限制 |
| [requirements-cpu-validated-windows-lock.txt](requirements-cpu-validated-windows-lock.txt) | 本轮独立 CPU 环境使用的 19 个第三方 wheel 版本与 SHA-256；范围是 Windows AMD64 / Python 3.11 |
| [install_cpu_isolated.ps1](install_cpu_isolated.ps1) | 新建独立 CPU 环境，缓存、源码构建副本、wheel 与日志全部放在显式 Data 目录；已有目标直接拒绝 |
| [smoke_cpu_environment.py](smoke_cpu_environment.py) | 检查独立解释器、二进制库导入、极小合成数学及 CLI help/doctor，不读取赛事材料 |
| [reconstruction-validation.json](reconstruction-validation.json) | 本轮解释器、wheel 散列、安装版本、验证结果、来源差异与实际限制 |
| [requirements-vision-torch-cu128.txt](requirements-vision-torch-cu128.txt) | 视觉环境的独立 CUDA/PyTorch 候选层，未安装或验证 |
| [gtsam-reconstruction-candidate.yml](gtsam-reconstruction-candidate.yml) | 历史 GTSAM 加独立几何依赖的候选配置，未安装或验证 |
| [body-upstream-provenance.json](body-upstream-provenance.json) | GVHMR/PyTorch3D 真实本地来源、上游差异及人体模型重建缺口 |

版本快照只说明当前机器装了什么。它没有 wheel 散列、全部操作系统依赖、完整 conda 锁或另一台机器的重建验收，不应称为完整锁文件。

## 已完成的独立 CPU 验收

新环境为 `D:\CodexData\RinkSpace_Data\runtime_tools\repository_validation\cpu-python311-20261011`，基础解释器为 `D:\Python\Python311\python.exe`（Python 3.11.0 / Windows AMD64）。19 个第三方 wheel 下载到同目录的 `wheelhouse/`，本轮从该 wheelhouse 离线安装；项目自身从私有 Data 源码副本构建 Python wheel，OneDrive 没有因此生成 build/egg-info。

`pip check`、NumPy/OpenCV/Pillow/PyAV/SciPy/quadprog/Clarabel/PyYAML 导入、极小矩阵/稀疏乘法与恒等变换、CLI `--help` 和 `doctor` 均通过。quadprog/Clarabel 只导入和检查 API，未运行优化。脚本还实际验证了拒绝已有环境和活动 `.venv` 目标。日志、环境二进制及缓存留在 Data；[重建记录](reconstruction-validation.json) 在 OneDrive 保存精确 wheel 身份和源文件散列。

这份 CPU 依赖锁包含本轮实际使用的 wheel 散列，仍不是整个项目的跨平台锁。原生内核、FFmpeg、事件参数/完整状态、视觉与人体权重不由它提供。独立环境 `doctor` 的 `ffmpeg/ffprobe` 为 null；源码 `__version__` 报 0.3.0，当前 `pyproject.toml` 与新 wheel 元数据为 0.4.0。两项缺口已记录，未改正在使用的源码或系统 PATH。setuptools 由独立环境升级到满足当前构建要求的 84.0.0，pytest 传递依赖 iniconfig 为 2.3.1；不能把新环境视为历史运行环境的逐字节等价替代。

创建另一套环境时，DataRoot 必须是已有的 Data 目录，EnvironmentDirectory 必须是其中**尚不存在**的目录。脚本默认使用已验收的依赖散列锁，可联网下载公共 CPU wheel，也可用 `-OfflineWheelhouse` 指向已有 wheelhouse。失败会保留独立目录和日志，不能对失败目录重跑覆盖；先按日志解决原因，再明确选择新目标。

```powershell
& .\environments\install_cpu_isolated.ps1 `
  -BasePython 'D:\Python\Python311\python.exe' `
  -DataRoot 'D:\CodexData\RinkSpace_Data\runtime_tools\repository_validation' `
  -EnvironmentDirectory 'D:\CodexData\RinkSpace_Data\runtime_tools\repository_validation\cpu-new' `
  -OfflineWheelhouse 'D:\CodexData\RinkSpace_Data\runtime_tools\repository_validation\wheelhouse'
```

另一台机器应替换解释器和 Data 路径。依赖散列锁与 Windows wheel 绑定，其他操作系统不能直接套用。项目 `calibration/solver_runtime` 仍需从源码 checkout 使用，CPU wheel 的 CLI 通过不代表完整赛事求解已可移植。

## 主环境：查看工具与 CPU 求解

[pyproject.toml](../pyproject.toml) 声明 Python >=3.11，基础依赖为 NumPy、OpenCV headless、Pillow、PyAV；`advanced` 包含 SciPy、quadprog、Clarabel，`dev` 包含 pytest/PyYAML，`acquisition` 包含 yt-dlp。

当前 [INSTALL_LOCAL.cmd](../INSTALL_LOCAL.cmd) 只安装 `.[dev,acquisition]`。因此它足以表达基础工具安装，但未安装 advanced；已有 `.venv` 的高级求解依赖来自此前安装。新环境要使用这些数值模块，需要另外选择 advanced。源码 checkout 根目录也必须保留：setuptools 当前只打包 `src/rinkspace`，`calibration/solver_runtime` 并不由普通 wheel 安装提供。

以下是常规源码安装形式；本轮使用上述脚本在 Data 副本构建普通 wheel，避免 editable 安装在活动源码目录生成元数据。已有运行环境不应被替换。

```powershell
python -m venv <新的环境目录>
& <新的环境目录>/Scripts/python.exe -m pip install -e ".[dev,advanced,acquisition]"
```

需要参考本机已观察版本时，[requirements-solver-observed.txt](requirements-solver-observed.txt) 只保存基础/数值依赖子集，没有开发、采集或项目自身。新环境使用已验收的 CPU 依赖散列锁与独立脚本；完整观察快照保持原样，供比较来源。依赖安装完成仍不能独立恢复完整节目求解：还需要明确的事件源包、完整状态和已认证原生后端。

主环境实际为 Python 3.11.0，NumPy 2.4.6、SciPy 1.17.1、quadprog 0.1.13、Clarabel 0.11.1。已安装 `rinkspace` 元数据为 0.1.0，而当前项目声明 0.4.0；已安装 setuptools 65.5.0 低于构建依赖 >=69。此处记录差异，未修复正在使用的环境。

## 视觉 CUDA 环境

[bootstrap.py](../calibration/program_tracking/runtime/bootstrap.py) 会读取本项目主 `.venv`，并追加旁项目 `MotionTrail_Codex_Starter_v1/.venv-gpu`。后者当前有 PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、NumPy 2.4.6、OpenCV headless 5.0.0.93、PyAV 18.1.0；该环境自身没有观察到 SciPy 分发，可能由主环境追加路径提供。

[视觉训练代码](../src/rinkspace/vision.py) 和 [网络定义](../src/rinkspace/vision_network.py) 使用 torch，当前 `pyproject.toml` 没有声明视觉 extra。旁项目 OpenCV 5.0 也超出本项目 <5 范围，因此完整复制这份快照不是已验证的本项目视觉安装方法。本轮保存了 [CUDA/PyTorch 候选层](requirements-vision-torch-cu128.txt)：先用独立脚本创建另一个新 vision 环境的 CPU 基础层，再仅在该环境安装本文件。该文件单独使用 PyTorch 的 cu128 index，CPU 基础层使用 PyPI，以免混用索引。版本/构建来自本机观察，CUDA 驱动、硬件和权重仍需另行核验；本轮未下载 torch、加载模型或执行 GPU 工作，也未改原路径。

## 人体与历史 GTSAM 环境

[body_prior.py](../calibration/program_tracking/body_tracking/body_prior.py) 的 infer/materialize 依赖旁项目 `FSrotation/cache/g0_gvhmr_runtime` 中 GVHMR 源码、专用 venv、GVHMR/HMR2 检查点和 SMPL-X 模型。函数还明确检查解释器前缀并要求 CUDA；只有主项目 `.venv` 无法运行该路径。[observations.py](../calibration/program_tracking/body_tracking/observations.py) 另外引用 FSpathing 的 YOLO 权重与 site-packages。当前 GVHMR 环境观察到 PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、PyTorch3D 0.7.8、smplx 0.1.28、gvhmr 1.0.0、ultralytics 8.2.42，完整列表见 JSON。

这些上游代码、特殊编译扩展、模型文件及许可需要分别登记。本项目快照没有把它们变成 RinkSpace 拥有或已可公开再分发的资产。GVHMR 环境的 PyAV 13.0.0 低于主项目 >=14，与主环境分开记录。

[人体来源记录](body-upstream-provenance.json) 已核对 GVHMR 的 editable 本地安装来源与 upstream HEAD `6ec3ca39336c50492c0fae65fba2fb831fc7d866`，以及 PyTorch3D 本地源码 HEAD `75ebeeaea0908c5527e7b1e305fbc7681382db47`。这不是完整源码保全：上游工作树改动与真实 Windows 编译参数还未核实。GVHMR 原 requirements 使用 torch 2.3/cu121、NumPy 1.23.5 和 Python 3.10 的 Linux PyTorch3D wheel，与当前 Windows/Python 3.11/CUDA128 安装明显不同，不能盲目 `pip freeze` 或原样执行上游 requirements。SMPL-X、HMR2、GVHMR 和 YOLO 模型须分别保留来源、身份与获取条件；本轮未读取或复制完整权重。

[历史 GTSAM 定义](../calibration/program_tracking/runtime/environment.yml) 记录 conda-forge、Python 3.11、gtsam 4.2.2、NumPy <2 和 OpenBLAS。现有 conda 环境实际 Python 3.11.16、gtsam 4.2.2、NumPy 1.26.4；源码中的 GTSAM、gtsam_unstable 与历史原生边缘模块需要该独立栈，不能与主环境 NumPy 2.x 当作同一安装。已有 GTSAM 原生 build.bat 还引用另行准备的 Eigen、Boost headers、pybind11、TBB include/lib 及 python311.lib；environment.yml 未完整定义这些构建资源。

[独立 GTSAM 候选配置](gtsam-reconstruction-candidate.yml) 固定已观察的 Python/GTSAM/NumPy 核心版本，并补充不依赖主 `.venv` 的 NumPy 1.26.x 几何层。新增 SciPy/OpenCV 等组合尚未安装验收，conda build strings 和 native/vendor 工具链也未锁定。未来只能 `conda env create --prefix <新的 Data 目录> --file environments/gtsam-reconstruction-candidate.yml`；不能更新现有环境或调用 bootstrap 追加 NumPy 2.x 的主环境。这里保留的是可审查重建候选，不能宣称历史 GTSAM/native 已重建成功。

## 原生内核

[原生 recipe 快照](native-build-recipes-observed.json) 保存九个内核的当前源码位置、源码/DLL SHA-256 和现有编译参数。八个 build.json 的源码与 DLL 逐项匹配；Brown4 两项匹配其 wrapper 固定散列。这是文件身份核对，没有执行后端加载或数值验收。

CPU recipe 使用 MSVC、`/O2 /fp:strict /LD /TC`；motion/scatter/CSR merge 额外保留 `/openmp /std:c11`，normal query 保留 `/openmp`，projection guard/value 保留 `/std:c11`。不能为了重建改成 fast-math、去掉严格浮点要求或静默选用旧 DLL。CUDA recipe 记录 `-arch=sm_120`、`--fmad=false`、`--prec-div=true`、`--prec-sqrt=true` 等真实参数；它是已有实验内核，不能据此声称完整求解器已接 GPU。

[config.py](../calibration/solver_runtime/config.py) 优先查找 `RINKSPACE_SOLVER_NATIVE_CACHE_ROOT/<kernel>/<source_sha256>` 下的 build.json 和 DLL，也保留已认证旧目录兼容。普通导入与求解不自行编译。现有 recipe 的编译器和旧资源路径是本机路径，需要在未来重建流程中显式配置。

[正式后端契约](../calibration/solver_runtime/legacy/structured_programme_source_candidate/runtime_backend_contract.py) 要求 nearest_segment、motion、projection_guard、csr_scatter 及 Brown4。Brown4 wrapper 固定核验历史 DLL SHA-256，重新编译不能预设字节相同；即使新 build.json 自洽，也可能无法通过这个固定条件。公开前需要保全已认证产物及可审查的构建信息，或在独立环境完成明确的兼容验证并调整相应契约。此次未绕过核验，也未把未经验证的新编译脚本当作可复建交付。

[现有 CSR 构建脚本](../scripts/diagnostics/build_programme_csr_three_span_merge_native_20261010.py) 读取 Data 中旧 native_scatter/build.json 的 toolchain/flags。其源码已在项目，但不是与旧机器路径完全解耦的重建入口。历史 edge build.bat 同样硬编码 VS、GTSAM 和 vendor 位置；保留其来源，后续再按独立环境核验后完善。

## 公开仓库仍需完成的事项

- 把每个被现行入口读取的项目自有源码与必要小型参数纳入 OneDrive 清单，并区分源码已保全、Git 尚未跟踪与外部资源缺失。
- 为视觉、人体、GTSAM/native 各自保全可重建定义、上游版本与来源、模型身份和必要许可；冻结证据的历史字节与来源记录保持原样。
- 本次公开整理已由根 .gitignore 排除安装/编译产物；本机已有 DLL 和 .lib 继续供本地使用，公开仓库保留源码、依赖及构建定义。旧索引随旧历史整体保全，未导入新公开历史。
- 本次未在项目根观察到 LICENSE/COPYING/NOTICE 文件。正式公开前需由项目所有者明确本项目许可证并核对外部资产，不把研究引用视作代码或模型的再分发许可。
