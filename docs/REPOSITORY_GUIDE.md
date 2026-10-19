# 仓库结构、核心资产与公开准备

返回唯一结果入口：[START_HERE.html](../START_HERE.html)。项目目标见 [README.md](../README.md)，算法与接口见 [ARCHITECTURE.md](ARCHITECTURE.md)。本文帮助第一次阅读仓库的人理解文件归属，整理记录更新到 2026-10-13；不选择最新模型，不替代正在进行的拟合核验。

## 项目在做什么

RinkSpace 从花样滑冰转播建立赛事共同的冰面、围挡、广告与相机模型，再把节目各镜头和选手轨迹绑定到同一场景及真实源时间。公共方法跨赛事复用，场地布局、尺寸依据、广告实例和相机安装事实各归对应赛事。当前安卡拉／冈田节目是开发样本，真实整镜头定位仍有未解决偏差；数值收敛与源码齐备均不代表准确性已验收。

阅读顺序是项目目标 → 架构 → 对应赛事的数据说明 → 对应节目的模型与输入 → 求解或审阅入口。需要理解求解器时先读 [公共运行接口](../calibration/solver_runtime/README.md)，无需先从上千个实验和诊断文件中猜当前入口。

## OneDrive 与 Data 的分工

| 内容 | 核心保存位置 | 阅读或使用方式 |
|---|---|---|
| 公共软件、数值方法、渲染和时间接口 | `src/rinkspace/`、`calibration/`、`scripts/` | 项目源码 checkout；当前 wheel 只打包 `src/rinkspace` |
| 环境定义、已观察依赖和原生构建信息 | `pyproject.toml`、`environments/` | 依据定义重建；现有 `.venv` 原位保留，不作为可移植 Git 环境 |
| 启动配置、存储地址与旧路径映射 | `configs/` | 现有配置含本机绝对地址，尚需独立机器的配置适配 |
| 赛事场景、安装事实、人工约束与源包 | `events/<event_id>/` | 数值只对该赛事及声明的时期有效 |
| 节目相机、源时钟、人体语义和运行 profile | `events/<event_id>/programmes/<programme_id>/` | 引用所属赛事场景；完整优化状态仍在 Data |
| 冻结源数据集及交付摘要 | 节目的 `source_datasets/` | 清单与资格政策在 OneDrive，大型 source pack、支持索引和过滤数组引用 Data |
| 最终视觉权重 | `models/` | 已有 Git LFS 属性；权重存在与被 Git/LFS 收录分别核对 |
| 原片、批量帧图、训练集、大数组、完整检查点、缓存、日志、诊断视频 | Data 的对应 `events/<event_id>/` | 不依赖公开 Git 仓库携带这些运行材料 |

人工笔划、源观测身份和不可替代的人工事实数组也属于核心证据。不能仅凭 `.npz` 后缀就把它们与可重算运动数组或完整求解检查点一起排除。此次保全清单逐项记录角色、来源和散列。

```text
RinkSpace/                         OneDrive / 未来 Git 工作区
├─ START_HERE.html                 唯一结果入口
├─ README.md                       目标与技术导航
├─ pyproject.toml                  基础及可选 Python 依赖
├─ environments/                   现有环境版本、native recipe 和复建限制
├─ src/rinkspace/                   公共软件与本地工具
├─ calibration/
│  ├─ solver_runtime/              已拥有的求解源码、身份契约与旧名字适配
│  ├─ program_tracking/            节目/人体/原片审阅代码
│  ├─ shot_tracking/               镜头级接口
│  └─ work/                        既有标定实现
├─ scripts/                        数据处理、导出与独立诊断入口
├─ configs/                        小型配置；local.json 不进入 Git
├─ events/ankara_2026/              已有开发赛事
│  ├─ models/                      注册、工作和初始化参数
│  ├─ model_constraints/           场地、广告、安装等事实
│  ├─ observations/                本次补充保全的原始源包与人工事实
│  ├─ tools/                       赛事特有的源装配辅助代码
│  └─ programmes/okada_mei_sp/      节目绑定、模型、冻结数据集与运行 profile
├─ models/                         最终神经权重
├─ docs/                           架构、算法、证据要求及本说明
└─ tests/                          接口与数值检查

RinkSpace_Data/                    本地外部运行材料
├─ .git/                           当前机器的 Git 数据库
├─ events/ankara_2026/              素材、数组、运行状态、视频与历史
├─ native/                         按源码身份登记的原生编译缓存
└─ maintenance/                    整理与只读审计记录
```

目录结构保留现有接口，没有为此次整理重命名活动模块或移动运行目录。`calibration/solver_runtime/legacy/` 中的名字还服务旧 import/pickle 身份；`legacy` 不表示这些代码可以删除。Data 中 `history/` 下的源输入或已认证 DLL 也可能仍在使用。

## 已完成的保全与独立准备

2026-10-11 首轮新增文档、环境观察记录、缺失核心资产副本与审计脚本，并在项目 README 增加阅读导航。后续已在 Data 的独立目录建立 CPU 验证环境；原运行环境、模型、profile、源清单及状态入口保持原样。本会话没有启动实际求解、训练或视频生成。

[source_manifest.json](../calibration/solver_runtime/source_manifest.json) 的既有登记闭包为 277 个 Python 模块和 8 份原生源码，四个旧赛事工厂另有精确映射。这个数字在本次元数据核对中未变，代表该登记范围，不是整个项目或新 strict 数据消费链的完整源码总数。新 strict 实现也在 OneDrive 的 `calibration/program_tracking/` 与 `calibration/solver_runtime/` 中；不能把 Data 中的所有历史 Python 文件再次整包拷贝进仓库。首轮另外保全了 Data-only 辅助代码；其中旧 `native_board_edge_source.py` 与正式版字节不同，两者不能覆盖互换。新增副本尚未切换到活动导入链，基于 `__file__` 的材料寻址仍需对应资源桥。

这四份赛事辅助源码及其依赖说明见 [印刷边界源码保全](../events/ankara_2026/tools/printed_component_extent/README.md)。另已补回 [track_scene3d_observations.py](../scripts/track_scene3d_observations.py)，恢复现有提取入口缺失的六个 helper；[来源记录](../scripts/track_scene3d_observations.provenance.json) 保留源散列及旧资源布局。该脚本 `--help` 已通过，但未运行媒体提取，其独立 main 的旧 `outputs` 路径仍需适配。

2026-10-13 另外保全了 [三份冻结 gray authority 来源方法](../events/ankara_2026/tools/strict_source_authorities/gray_review_20261013/README.md)，共 90,109 字节。副本与原件及既有冻结 receipt 的源码 SHA 一致；没有复制另外 353 份已来自 OneDrive 的源码快照。这是历史来源方法保全，尚未证明它们处于当前 v6 求解导入链，也不因保全就批准对应点进入 FIT。运行引用和原件保持原样。

当前 profile 引用的必要 JSON 源包和人工事实若已有同 SHA-256 的项目文件就复用，缺失的才复制。原件和现有配置地址保留，新增来源清单不修改求解输入。嵌套源图、原片、运动数组、完整状态等外部依赖另行登记；顶层源包保全不等于整个依赖闭包已经在仓库内。

[赛事资产保全清单](../events/ankara_2026/asset_preservation_manifest.json) 登记 105 项，其中 3 项复用已有项目副本，102 项新增保全，共 34,678,685 字节（84 个 JSON、18 个人工事实 NPZ）。源／目标散列已核对，NPZ 成员以 `allow_pickle=False` 实际读取通过。清单另外记录 853 个嵌套外部引用；其中两张旧解码 PNG 已缺失，一张历史 PNG 读取权限不足，存在性未知。此次未重新生成这些历史源图。

2026-10-13 继续沿 strict v6 与 q32 的显式引用补充 [参数增量清单](../events/ankara_2026/asset_preservation_manifest_20261013.json)：登记 114 项，新增 25 个 JSON，共 13,076,073 字节，另外 89 项复用已有项目资源。新增内容包括 10 月 12 日的源包修订、人工事实重绑定、q32 数值政策、strict motion 政策以及 v6 明示的资格过滤政策。原 18 个人工笔划／guard NPZ 均复用，没有新增大数组。两轮清单有重合项，不能直接相加作为独立资产总数。

这份增量清单记录了两个仍在 Data 的完整相机种子状态及 1,087 个嵌套外部引用；引用包括原片、帧图、数组和历史证据，不能全部称作当前必需输入。v6 的 31 个原 source pack／support index、strict motion 数组与其他运行数值仍需原 Data 材料。新增副本的源／目标 SHA 和复制前后源 stat 已核对；原 profile、数据集及运行路径保持原样，完整依赖闭包和独立机器运行尚未通过。

另外保全了 [完整周期 baseline 历史 helper](../calibration/solver_runtime_preserved/complete_cycle_baseline_20261010/README.md)，12,914 字节，其 SHA 与现有显式选择器要求一致。该副本与当前维护版不同，独立归档而不覆盖维护版或改选择器。`station_columns_runtime_draft_20261012` 的唯一源码已与 OneDrive 维护版逐字相同，不再复制。历史原生构建小文件的保全及剩余加载条件见 [运行依赖说明](RUNTIME_DEPENDENCIES.md)。

[环境说明](../environments/README.md) 保存四类环境观察值、原生 recipe 和 [独立重建记录](../environments/reconstruction-validation.json)。2026-10-11 独立 CPU 环境的依赖、基础 CLI 与轻量数学检查已通过，并保存 Windows/Python 3.11 的 wheel 散列锁；该结果不包含原生完整求解、完整视频、视觉 GPU 或人体运行。它绑定当时的私有源码构建快照，不能替代最近新增代码的验收。本轮未再次安装环境或重跑这些检查。

人体上游 helper 与 GVHMR 的三处本地修改也已保全，身份和许可证据见 [第三方资产说明](THIRD_PARTY_ASSETS.md)。公开内容的索引、工作区和 LFS 状态另见 [公开内容说明](PUBLIC_REPOSITORY_CONTENTS.md)。独立路径工具及其未接入的边界见 [可选资源解析](PORTABLE_RESOURCES.md)。

剩余环境、原生库、媒体命令与路径／producer 条件集中在 [运行依赖说明](RUNTIME_DEPENDENCIES.md)，机器可读记录在 [runtime-dependencies-current.json](../environments/runtime-dependencies-current.json)。2026-10-13 的环境部分仅查小型定义与存在性，不能沿用之前 CPU smoke 结果作为今天的完整运行验收。本次检查沿明确引用和少量候选展开，没有全盘遍历 Data，不能宣称已找出所有历史孤立文件。

## 2026-10-13：冻结数据与求解接入分别记录

最新已整理的节目源清单为 [specific_inner_review_v6/fit_dataset.json](../events/ankara_2026/programmes/okada_mei_sp/source_datasets/current_review_20261013/specific_inner_review_v6/fit_dataset.json)，交付事实见 [delivery_summary.json](../events/ankara_2026/programmes/okada_mei_sp/source_datasets/current_review_20261013/specific_inner_review_v6/delivery_summary.json)。冻结清单登记如下：

| 项目 | 冻结记录 | 解释 |
|---|---:|---|
| FIT | 1,282,102 | 拟合候选点，不是全部点已获独立准确性验收 |
| HELD | 2,368 | 留出身份与实际独立性仍按对应源事实判断 |
| 待核支持／待核归属 | 63,085／5,549 | 不进入 FIT，保留来源；不能自动当作已确认倒影或训练负例 |
| 源包／镜头 | 31／9 | 原片覆盖身份按本节目清单解释 |
| 展示帧／独立曝光 | 10,123／8,503 | 不能把展示重复帧算成新的独立证据 |
| 有 FIT／当前无 FIT 的曝光 | 6,626／1,877 | 缺失曝光计入覆盖情况，不用其他时刻的点补齐 |

当前源筛选遵守用户更新的场地范围：只允许原片核实属于冰面或围挡本体的静止实体与材料；看台、人物、字幕、倒影等范围外点及归属未确认点排除，不能以未知深度继续场地拟合。源清单、资格政策、完整状态和视觉准确性是不同记录。

所选 [strict_v6_source_fit_20261013.json](../events/ankara_2026/programmes/okada_mei_sp/strict_v6_source_fit_20261013.json) 显式保留 `ice/board`、未知／未审阅默认排除及原完整续算身份；静态核对时种子仍是 q32。隔壁“推进冈田芽衣网格拟合”正在独立接入这份数据。冻结交付记录中“完整图未接入／求解未启动”是当时状态，不应改写为后续运行状态；同样不能把后续接入称作已经产生通过视觉验收的新模型。实时进展和有效结果继续从 START_HERE 查找。

## 如何核对核心文件

[audit_repository_assets.py](../scripts/audit_repository_assets.py) 可检查正式源码清单、保全资产散列和 profile 声明输入，但会重读登记文件内容，本轮没有运行。要求轻量整理时用 [audit_public_repository.py](../scripts/audit_public_repository.py) 的元数据模式，仅读 Git 路径／属性及文件大小、存在性：

```powershell
& D:\Python\Python311\python.exe -B scripts\audit_public_repository.py --metadata-only `
  --data-root D:\CodexData\RinkSpace_Data `
  --output D:\CodexData\RinkSpace_Data\maintenance\repository_preparation_20261011\public_repository_metadata_20261013.json
```

元数据模式不扫描凭证、历史对象、blob 或文件散列，不能沿用 2026-10-11 的旧核验结论。完整资产报告中的 `missing_declared_profile_inputs` 也只核对直接声明地址，不能代替 JSON／NPZ／pickle 的完整依赖审计。活动 profile 不会因此改写；读取期间另一对话仍可能修改文件，静态报告与实时状态分开解释，不通过自动改清单掩盖差异。

## 与完整公开复现的距离

源码保全、依赖可重建、外部数据可获取和真实准确性验收是不同状态。目前仍有这些实证缺口：

- 视觉和人体路径追加或调用旁项目环境；GVHMR、HMR2、SMPL-X、YOLO 等上游代码和模型需各自登记版本、来源与许可，主项目 `.venv` 不能单独提供它们。
- 部分 native 构建脚本固定本机编译器和 vendor 路径；Brown4 wrapper 还固定核验历史 DLL SHA，重新编译不能预设可通过。已观察 recipe 需要独立环境的实际兼容核验。
- 活动适配器、旧序列化状态、源材料和 profile 仍含本机绝对地址。此次没有改变这些地址；后续路径切换需在独立输出中验证完整状态和源身份。
- 赛事 `manifest.json` 是 2026-10-06 的注册快照，后续状态在节目文件和主入口中继续更新。不能把该快照当作当前拟合身份，也不能把此次保全清单作为最新模型选择器。
- 工作区已有大量未跟踪的核心源码，历史编译产物也有已跟踪项。`.gitignore` 和 LFS 属性不能自动将已有内容变成适合公开的提交；本轮未暂存、提交、改写历史或推送。
- 仓库尚无项目许可证文件；本项目许可证与第三方资产的许可需分别明确。源视频和人体模型不会因存于本机就自动成为公开仓库资产。

可选路径解析与独立 CPU 重建已完成上述准备范围；活动运行地址仍未切换，完整节目与人体／GPU 运行仍需各自核验。当前文档提供目标、接口、结构、来源与实际缺口，尚不宣称公开仓库能够单独复现完整节目。
