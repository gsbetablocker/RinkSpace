# 维护与开发

公开阅读从 [README](README.md) 开始。本页维护工作入口、本机工具和路径解析；存储边界与 Windows 路径限制见 [AGENTS.md](AGENTS.md)。

<a id="work-entry"></a>
## 工作要求与入口

| 内容 | 正文位置 |
|---|---|
| 用户明确的项目长期协作、存储与证据边界 | [AGENTS.md](AGENTS.md) |
| 原片、机位接入、场地拟合范围、广告几何 | [BUILD_SPEC](docs/rink_model/BUILD_SPEC.md) |
| 赛事实体、取点定义、安装事实、参数 | 所属赛事；[安卡拉场景](events/ankara_2026/scene/README.md) |
| 单节目源时钟、镜头范围、输入与结果记录 | 所属节目；[冈田芽衣短节目](events/ankara_2026/programmes/okada_mei_sp/README.md) |
| 模块职责／数值算法／精确接口／已实现能力 | [ARCHITECTURE](docs/ARCHITECTURE.md)／[SOLVER](docs/SOLVER.md)／[INTERFACES](docs/INTERFACES.md)／[REBUILD](docs/REBUILD.md) |
| 启动、重启与续算 | [运行要求](docs/SOLVER.md#run-contract) |
| 完整镜头、独立证据、尺度与身体／接触验收 | [VALIDATION](docs/VALIDATION.md) |
| 公开资产整理、媒体与离线数据边界 | [GITHUB_REVIEW](docs/GITHUB_REVIEW.md#public-cleanup) |

同一要求只在适用位置维护正文。一次任务的取值、处理顺序和失败留在已有的赛事／节目记录或会话；需要接手时链接现有范围与绑定，不另造任务书。历史记录只供追查，不能用其中的“当前”“必须”“下一步”选择今天的模型或扩大授权。

根 AGENTS.md 提供按任务读取的入口；普通 Markdown 链接不会自动加载全文，子目录 AGENTS.md 也不会因为从根目录启动而全部加载。反复复用的操作流程确有需要时再做 skill，具体事实仍归赛事／节目。此安排参考 OpenAI 的 [Customization](https://learn.chatgpt.com/docs/customization/overview)、[AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md) 与 [Skills](https://learn.chatgpt.com/docs/build-skills)，不是 OpenAI 规定的仓库目录标准。

## 本机工具与存储解析

| 工具 | 用途与边界 |
|---|---|
| [INSTALL_LOCAL.cmd](INSTALL_LOCAL.cmd)、[START_LOCAL.cmd](START_LOCAL.cmd) | 基础安装与本地服务；安装器选择 dev/acquisition，不含 advanced |
| [START_POINT_EDITOR.cmd](START_POINT_EDITOR.cmd)、[参数](START_POINT_EDITOR.ps1) | 指定 Dataset、Media、SourceMedia、Model、Port，编辑有源身份的观测 |
| [refresh_project_entry.py](scripts/refresh_project_entry.py) | 刷新 Data 中的结果面板，保留 CURRENT_TASK 与冻结绑定 |
| [START_FAST_SOLVER.ps1](START_FAST_SOLVER.ps1)、[solver_runtime](calibration/solver_runtime/README.md)、[rinkfit](calibration/rinkfit/README.md) | 历史后端／样本入口；只能按相符的输入、完整状态和 profile 使用，不代表新引擎已支持 solve/resume |

共享映射为 [storage.defaults.json](configs/storage.defaults.json)。新机器可复制 [storage.example.json](configs/storage.example.json) 为受忽略的 `configs/storage.json`，填入绝对 Data 根，或设置 `RINKSPACE_DATA_ROOT`。显式 `AssetStore(root, data_root=...)` 优先于环境变量和本机文件；未配置 Data 时，通用工作台使用受忽略的 outputs，pytest 缓存使用系统临时目录。

`AssetStore.project_entry()` 定位到 `Data/workbench/START_HERE.html`；Web `/` 在该文件存在时显示面板，否则显示通用工作台。审阅页经 source_datasets → results 定位 Data 产物。旧根页面和历史快捷方式仅本机保留；刷新器可用旧页面首次初始化，无冻结绑定时无需生成面板。

场景历史资产与导入／安装中间文件的旧路径、SHA、字节数和 Data 地址见[归档映射](events/ankara_2026/scene/archive.json)。`AssetStore(root).resolve(old_path, must_exist=True)` 查询原件；原件内部相对引用仍以原项目目录为基准，经同一 AssetStore 解析。共享 `legacy_paths` 与本机设置按键合并，本机同名项优先。当前场景求值、prepare/review 和公开测试不要求这些历史原件存在。

旧地址别名见 [path_aliases.json](configs/path_aliases.json)，ResourceRegistry 的适用范围见[资源接口](docs/PORTABLE_RESOURCES.md)。

## 开发与公开检查

```console
python -B scripts/check_onedrive_paths.py
python -B scripts/check_github_readiness.py --data-root <外部Data绝对路径> --output <Data内报告绝对路径> --review-only
```

路径检查包含被忽略的环境和缓存；公开检查报告写 Data。修改功能时选择相关测试，基础范围见 README，依赖见[环境说明](environments/README.md)。公开内容、数据身份和许可分别按 [GITHUB_REVIEW](docs/GITHUB_REVIEW.md)、[LICENSE](LICENSE.md) 与[许可核查](docs/LICENSING_REVIEW.md)核对。
