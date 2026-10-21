# 按能力准备环境

使用自己的 Python >=3.11 环境。已有本机安装不作为可移植环境分发；完整赛事运行还需要源材料、完整状态和匹配后端。项目许可见 [LICENSE](../LICENSE.md)，第三方范围见[资产说明](../docs/THIRD_PARTY_ASSETS.md)。

| 能力 | 安装或准备 | 边界 |
|---|---|---|
| 查看离线审阅归档 | Python 标准库；运行 read_offline_review.py | 不解码视频、不执行包内代码、不运行求解器 |
| 基础几何、运动、拼接和合成 demo | `python -m pip install -e ".[dev]"` | NumPy/OpenCV/Pillow/PyAV；基础测试范围见 README |
| 高级 CPU 数值 | `python -m pip install -e ".[dev,advanced]"` | 增加 SciPy、quadprog、Clarabel；实际 profile 可能还需原生库 |
| 采集接口 | `python -m pip install -e ".[acquisition]"` | 增加 yt-dlp；材料获取和使用遵循来源条件 |
| 媒体探测与转码 | 自行配置 PATH 上的 ffmpeg 和 ffprobe | Python 包可导入不等于这两个可执行程序齐备；doctor 报告实际路径 |
| 视觉 GPU | [CUDA/PyTorch 候选层](requirements-vision-torch-cu128.txt)、硬件/驱动及对应权重 | 未完成独立 GPU 重建验收；不属于基础 extra |
| 人体 | [上游与补丁身份](body-upstream-provenance.json)、专用环境、GVHMR/HMR2/SMPL-X 等资源 | 仍有外部源码、特殊编译及模型获取条件，不能仅安装本项目重建 |
| 历史 GTSAM | [独立候选配置](gtsam-reconstruction-candidate.yml) | NumPy 1.26/GTSAM 4.2.2 栈，与主 NumPy 2.x 环境分开；候选未验收 |
| Ceres 与原生求解 | [rinkfit 构建/API](../calibration/rinkfit/README.md)、[原生 recipe](native-build-recipes-observed.json) | Ceres/Eigen/SuiteSparse 或匹配的 Windows 后端；源码 checkout、ABI 和身份条件须齐备 |

INSTALL_LOCAL.cmd 安装 dev/acquisition，不含 advanced。当前 setuptools 只发现 src 下的包，calibration 求解模块仍需要完整源码 checkout；普通 Python wheel 不包含整个赛事运行闭包。

## 基础测试与完整回归

~~~console
python -m rinkspace doctor
python -m rinkspace demo --out ../rinkspace-demo
python -m pytest tests/test_geometry.py tests/test_motion.py tests/test_mosaic.py tests/test_demo.py -q
~~~

上述范围只用合成数据。直接运行全体 pytest 会收集 SciPy、原生库和本机历史实验测试；需先准备相应依赖。没有历史材料时，依赖该材料的回放应报告缺失或跳过，不能算通过。数值测试不替代[真实镜头验证](../docs/VALIDATION.md)。

## 可复核的环境与构建记录

| 记录 | 用途 |
|---|---|
| [CPU wheel 锁](requirements-cpu-validated-windows-lock.txt)、[独立安装脚本](install_cpu_isolated.ps1)、[smoke](smoke_cpu_environment.py) | Windows AMD64/Python 3.11 的 19 个 wheel 及散列；脚本要求显式 Data 和未存在的新环境目录 |
| [CPU 重建记录](reconstruction-validation.json) | 2026-10-11 的依赖导入、轻量数学、CLI help/doctor 验证；不含真实视频、完整求解或 GPU |
| [观察版本](observed-python311-windows.json)、[数值依赖子集](requirements-solver-observed.txt) | 本机历史安装的观察值，不能当作跨平台锁 |
| [原生 recipe](native-build-recipes-observed.json)、[运行资源快照](runtime-dependencies-current.json) | 历史源码/产物身份、编译参数和元数据观察；存在性与记录 SHA 不等于重新加载/数值认证 |
| [历史原生支持](native_support_preserved/round3_legacy/README.md) | reference_callback/oneTBB 的源码、构建和来源记录 |

独立 CPU 安装脚本使用 `-BasePython`、`-DataRoot`、`-EnvironmentDirectory` 指定本机路径；可用 `-OfflineWheelhouse` 指向已有 wheelhouse。锁只适用于所列 Windows wheel，其他平台须单独验证。历史 CPU 记录绑定当时源码，不能替代当前版本检查。

## 原生和外部资源的限制

solver_runtime 先核对源码 SHA，再查找 RINKSPACE_SOLVER_NATIVE_CACHE_ROOT 下匹配的 build.json 与 DLL，或已认证旧目录；普通导入与求解不自动编译。MSVC recipe 的严格浮点设置及 CUDA recipe 的精度参数按来源保留，不能用旧 DLL 代替变更源码。Brown4 wrapper 还固定历史 DLL 身份，新编译产物不预设可通过原契约；CUDA motion 实验不代表完整求解已接 GPU。

历史 reference_callback 需要匹配 Python/C++ ABI、GTSAM、Eigen、Boost、pybind11 和 oneTBB include/lib。build_manifest 保存 vendor archive 的历史 SHA，不代表今天重新核验了 archive 或拥有完整下载/许可闭包。相应实验文件在[来源清单](native_support_preserved/round3_legacy/source_preservation_manifest.json)登记。

人体路径当前连接外部 GVHMR、PyTorch3D、YOLO/helper 和专用解释器；补丁及来源不等于完整上游树、模型或 Windows 编译链已经可移植。模型获取条件见[第三方资产](../docs/THIRD_PARTY_ASSETS.md)。

源身份、PTS、资格、完整 q/profile、统计 W、物理 guard、producer 与原生身份均须对应所选运行。旧 import/pickle 名称由 source_manifest 加载；复制文件或更换根目录不会自动改写这些契约。[可选 ResourceRegistry](../docs/PORTABLE_RESOURCES.md)仍是独立准备接口，尚未替代活动 AssetStore 和完整运行路径。

历史原生支持 README 按来源清单保留原字节，其中 smoke_report、synthetic_scaling 和旧第三轮报告不随公开仓库提供；其旧相对链接属于历史资源布局。构建身份可从 source_preservation_manifest 和各 build_manifest 查询，不能将缺失报告当作当前复现依赖。

历史 solver_runtime 的 source_manifest 固定原始源码身份，部分公开脱敏字节会被原校验拒绝。依赖该运行时的测试需准备匹配原件；纯代价保护测试按需加载 legacy，不为测试绕过生产身份检查。
