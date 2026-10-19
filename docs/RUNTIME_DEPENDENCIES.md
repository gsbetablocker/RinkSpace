# 剩余运行依赖与保全边界

2026-10-19 补充：下文是10月13日的运行依赖观察。当前九镜头技术审阅所需的模型、完整失败状态、观测/W数组、资格账本及Data独有恢复driver，现已保全到[programme/review](../events/ankara_2026/programmes/okada_mei_sp/review/README.md)的少量归档；可通过[标准库离线工具](../scripts/read_offline_review.py)直接访问，不依赖原Data地址。运行中的旧绑定仍保留；完整原片、安装环境、编译库和可再生批量产物继续外置，当前全九镜头仍未成功且保持停机。


返回项目唯一入口：[START_HERE.html](../START_HERE.html)。机器记录见 [runtime-dependencies-current.json](../environments/runtime-dependencies-current.json)，环境定义和既有验收见 [environments/README.md](../environments/README.md)。

**2026-10-13 本次环境核对仅检查元数据；小型新保全源码和参数另做字节校验。** 环境核对只读取已登记文件的存在性、大小和时间戳，以及小型源码、profile、build JSON 与既有来源清单；七个小型原生支持定义另外完成原始字节保全、源 stat 稳定性及 SHA-256 核对，其他源码和参数保全以各自来源清单为准。没有重新冻结环境，未读取 DLL、权重或大数组内容来计算散列，未导入 NumPy/torch/原生库，未安装、编译、测试、求解或解码视频。现有活动文件与运行路径未修改。文件存在、清单字段相同都不能证明加载成功、ABI 兼容、字节身份未变或完整运行可移植。

## 哪些已经保全，哪些仍是外部运行资源

| 层 | OneDrive 已有的代码或定义 | 仍需的运行资源与限制 |
|---|---|---|
| 基础 CPU 工具 | Python 项目源码、依赖定义、独立安装脚本、19 个 wheel 的散列锁与 2026-10-11 的独立 smoke 记录 | 既有独立 CPU 环境和 wheelhouse 留在 Data；本次只确认其解释器及 pyvenv.cfg 存在，没有重跑验收 |
| named66 求解 | 公共接口、拥有的 legacy 源码、来源 manifest、事件工厂、严格源适配器、profile 与必要小型参数 | 原片、源数组、完整状态、缓存、原生产物和运行证据仍在 Data；普通 Python wheel 不包含整个 calibration 源码树 |
| 原生 CPU/CUDA | C/CUDA 源码及 [九个内核的历史 recipe](../environments/native-build-recipes-observed.json) | 已登记九个 DLL 均仍存在，八个 build.json 的存储 SHA/flags 与旧 recipe 一致；今天没有验证实际源码/DLL 内容或执行加载 |
| 视觉 CUDA | 本项目视觉网络与训练代码、独立 PyTorch cu128 候选定义、已观察版本 | 旁项目 CUDA 环境仍由 bootstrap 引用；候选定义没有独立 GPU 重建验收 |
| 人体 | 连接代码、GVHMR/PyTorch3D 身份登记、GVHMR 三处本地修改补丁、ViTPose-H helper 副本与来源 | 完整上游源码树、专用解释器、特殊编译扩展、模型及小型辅助数据仍外部登记；helper 副本还未替换活动默认路径 |
| 历史 GTSAM | environment.yml、重建候选、reference_callback.cpp、历史 edge 构建脚本 | GTSAM/NumPy 1.26 栈、Eigen/Boost/pybind11 headers、oneTBB import libraries 与匹配 Python/C++ ABI 仍需独立复建 |
| 路径准备 | 可选 ResourceRegistry、profile 模板、源码/参数/附件来源映射 | 这是准备接口；活动 AssetStore 和求解器没有接入可选解析器 |

原片、批量帧图、训练数据、大数组、完整检查点、线性化及序列化缓存、日志和诊断视频留在 Data，是项目明确的存储分工。必要参数、不可替代人工事实、源身份与构建定义应在 OneDrive 有可查来源；它们的保全副本不会自动替代完整状态或改变原文件中的地址。

2026-10-13 的[参数增量清单](../events/ankara_2026/asset_preservation_manifest_20261013.json)另保全了 25 个小型 JSON 政策／绑定。它们的内部历史地址和活动 profile 仍使用 Data，不能把副本视为已换运行入口。[完整周期基线保全说明](../calibration/solver_runtime_preserved/complete_cycle_baseline_20261010/README.md)登记另保全的历史源码，文件身份与用途按该清单区分，未自动接入当前 producer。

## 选定 strict profile 与旧默认入口

本次读取的 [strict_v6_source_fit_20261013.json](../events/ankara_2026/programmes/okada_mei_sp/strict_v6_source_fit_20261013.json) 声明 `strict-finite-dataset-startup-policy.v2`，限定冰面与围挡，未确认点默认排除。其数据集 JSON 在 OneDrive，运动输入绑定和完整续算检查点仍指向 Data；完整检查点 SHA、canonical q 身份和 fresh J 条件保存在 profile 中。这里只读取这些声明，没有装载数据集、检查点或实际 producer，也不据此判断隔壁进程的实时状态或最新模型。

[严格启动检查源码](../calibration/solver_runtime/strict_model_scope_inputs.py) 已存在 v2 分支，并核对 strict graph 的显式 dataset/motion 参数及来源身份。[严格数据适配器](../calibration/solver_runtime/strict_source_dataset_adapter.py) 也在项目。不能把旧 v1 分支的未接入状态概括成当前 v2 完全未实现；反过来，源码存在也不能代替本次没有执行的完整源消费核验。

[START_FAST_SOLVER.ps1](../START_FAST_SOLVER.ps1) 的默认文件仍为 [okada_named66.json](../configs/solver_runtime/okada_named66.json)。本次读到该旧默认 policy v1 的 `all_pixel_families_routed_to_strict_input=false`、`source_review_ready_for_fit=false`，正常拟合启动必须受现有严格门禁限制，不能回退旧输入继续拟合。后续选择入口时须明确所用 profile、对应完整状态和实际来源，不由本整理文档指定新的拟合任务。

场地拟合点必须是原片核实属于冰面或围挡本体的静止实体与材料。看台、人物、字幕、倒影以及归属未确认点均排除，不能用未知深度保留在场地目标中。资源路径整理不认证点的资格，也不替代完整镜头和独立真实证据的定位验证。

## 路径、序列化与 producer 身份

[strict v6 可选模板](../configs/portable/strict_v6_source_fit_20261013.resources.json) 登记 35 个直接资源项，未解析直接地址数量为 0；它同时明确 `runtime_activated=false`、`registered_asset_verification_performed=false`、`portable_complete_run_verified=false`。这些是准备记录，不能当作完整运行可搬家的证明。

该模板是在本次参数增量保全前生成的快照，未因新副本或新增来源清单自动刷新；其中路径与身份只描述生成时的版本。

模板只处理 profile 的直接选项和 resource_root；profile 附加元数据、冻结 JSON 内部地址、检查点历史来源和嵌套材料仍需适配器处理。源码 finder 保留 legacy import/pickle 名称并依 source_manifest 校验文件，事件工厂也有来源契约。把文件复制到 OneDrive或登记别名，不会自动换掉 pickle 中的模块、原生库加载目录、材料 producer 或其认证散列。

未来独立重建需要同时核对：项目源码身份、源时钟与原片身份、观测角色/分割/不确定性、完整 q/profile/辅助状态、物理 guard 和实际 producer。可选解析器的根目录约束只能说明路径解析受控，不能说明这些条件已全部满足。当前既有运行路径保持原样，细节见 [PORTABLE_RESOURCES.md](PORTABLE_RESOURCES.md)。

## 原生加载与历史 GTSAM 定义

[config.py](../calibration/solver_runtime/config.py) 支持显式 resource_root/native cache 根，并保留已有旧目录。普通导入和求解不自动编译。[正式后端契约](../calibration/solver_runtime/legacy/structured_programme_source_candidate/runtime_backend_contract.py) 要求 nearest_segment、motion、projection_guard、csr_scatter 与 Brown4。Brown4 wrapper 固定历史源码和 DLL 散列；新机器重新编译不能预设字节相同，不能用新 build.json 绕过此条件。

已有 Data 产物继续原位保留。2026-10-11 的 recipe 身份记录与 2026-10-13 的存在性检查分别保存；今天只核对清单字段，没有把“存储 SHA 相同”写成“文件字节新验收通过”。CUDA motion recipe 是已有实验构建，不代表完整 named66 正式求解已接 GPU。

历史 `reference_callback` 的项目 C++ 源码已在 OneDrive，Data 中的 build.bat、build_manifest.json 与 README 现也按原始字节[保全到项目](../environments/native_support_preserved/round3_legacy/reference_callback/README.md)。该 build_manifest 已保存 Eigen、Boost、pybind11 三个 .conda 的**历史包 SHA**，比旧 third-party-assets 中“包散列未建立”的记录多出身份线索；本次核对了清单文件自身的字节，未重读 vendor 包。它们的精确下载 URL、source revision、完整许可/构建与 ABI 仍需分别核对，不能把历史散列当作今天的 archive 验证。

oneTBB 的原 README 已记录官方 v2023.1.0 Windows archive 散列，以及实际 conda runtime 关系；构建时仍依赖外部 include/lib。tbb_control.cpp、build.bat、README、LICENSE 四个小型定义已[保全到项目](../environments/native_support_preserved/round3_legacy/tbb_control/README.md)。它是历史独立控制实验，不能因此给 named66 正式后端新增依赖。七个保全文档／源码共 19,969 字节，[来源清单](../environments/native_support_preserved/round3_legacy/source_preservation_manifest.json)记录原件与副本地址、SHA-256、源 stat 稳定和逐字节一致性；runtime_activated=false，旧路径原样保留。完整 vendor、DLL 与环境仍在原位置。第三方 LICENSE 的适用内容按原 README 与许可原文记录，不给整个项目赋予许可证。

## FFmpeg 与外部人体路径

本次只读 PATH 查询未找到 `ffmpeg` 和 `ffprobe`。主 `.venv` 的 imageio_ffmpeg 包内确有 `ffmpeg-win-x86_64-v7.1.exe`，仅核对存在和大小，没有运行。它不自动提供 ffprobe，[media.py](../src/rinkspace/media.py) 仍调用 PATH 上的 ffmpeg/ffprobe，并在找不到时显式失败。CPU 包导入验收不能替代这两个独立可执行资源的配置。

[人体来源清单](../environments/third-party-assets.json) 已记录当前上游修订、GVHMR 修改补丁、helper、模型来源和权限缺口。本次未重新检查大模型或冻结专用 venv。body_prior 检查 FSrotation 的专用解释器前缀并切换到外部 GVHMR 目录，observations 的默认 helper 与 YOLO package 路径也仍外部引用。完整上游树、PyTorch3D 的 Windows 编译参数、内部组件修订及模型获取条件尚未构成已验收的独立重建方案；详见 [THIRD_PARTY_ASSETS.md](THIRD_PARTY_ASSETS.md)。

当前文档说明代码和定义的保全范围及完整运行的条件；Git 仓库单独恢复完整节目仍需要上述外部资源和独立运行核验。
