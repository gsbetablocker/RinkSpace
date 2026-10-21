# 维护与开发

公开阅读从 [README](README.md) 开始。源码、环境/构建定义、模型参数、人工事实和必要审阅归档保存在项目；原片、帧图、大数组、完整检查点、缓存、日志和诊断视频保存在配置的外部 Data。目录分工见[架构](docs/ARCHITECTURE.md)。

## 本机工具

| 工具 | 用途 |
|---|---|
| [INSTALL_LOCAL.cmd](INSTALL_LOCAL.cmd)、[START_LOCAL.cmd](START_LOCAL.cmd) | Windows 基础安装和本地服务；安装器选择 dev/acquisition，不含 advanced |
| [START_POINT_EDITOR.cmd](START_POINT_EDITOR.cmd)、[PowerShell 参数](START_POINT_EDITOR.ps1) | 指定 Dataset、Media、SourceMedia、Model 和 Port，编辑有源身份的逐帧观测 |
| [refresh_project_entry.py](scripts/refresh_project_entry.py) | 刷新 Data 中的本机结果面板，保留 CURRENT_TASK 与冻结结果绑定 |
| [START_FAST_SOLVER.ps1](START_FAST_SOLVER.ps1) | 现有样本配置的便利入口；输入、完整状态和后端须与所选 profile 相符 |
| [solver_runtime](calibration/solver_runtime/README.md)、[rinkfit](calibration/rinkfit/README.md) | 求解与后端接口 |

共享资源映射保存在 [storage.defaults.json](configs/storage.defaults.json)。本机 `configs/storage.json` 被 Git 忽略；新机器可复制 [storage.example.json](configs/storage.example.json) 并填入绝对 Data 根，或设置 `RINKSPACE_DATA_ROOT`。显式 `AssetStore(root, data_root=...)` 优先于环境变量和本机文件；未配置 Data 时，通用工作台只使用受忽略的 outputs，本机 pytest 缓存使用系统临时目录。AssetStore.project_entry() 将面板定位到 Data/workbench/START_HERE.html；Web 服务的 / 路由在该页面存在时显示它，否则使用通用工作台。审阅页按 source_datasets → results 定位到 Data，源观测和人工确认仍留原目录。

旧根页面、镜头专用快捷方式及历史交接材料在本机保留并忽略，不作为公开入口。它们可能仍被已有对话使用。刷新器可从旧根页面首次初始化 Data 面板；本机无冻结绑定时无需生成该面板，基础 CLI 和通用工作台可独立使用。

## 检查

~~~console
python -B scripts/check_onedrive_paths.py
python -B scripts/check_github_readiness.py --data-root <外部Data绝对路径> --output <Data内报告绝对路径> --review-only
~~~

路径检查包含被忽略的环境和缓存。公开检查复用已有大小、LFS、凭证候选及本机生成物检查；检查报告写入 Data。修改功能时选择相关测试，基础范围见 README，高级依赖见[环境说明](environments/README.md)。

旧地址的精确文件别名见[路径映射](configs/path_aliases.json)，可选 ResourceRegistry 的限制见[资源接口](docs/PORTABLE_RESOURCES.md)。公开派生文件与原运行字节的区别见[数据身份](docs/GITHUB_REVIEW.md#数据身份与缺失材料)。Git 数据库继续留在本机 Data；本机维护记录不进入公开阅读路径。

公开前按 [LICENSE](LICENSE.md) 与[许可核查](docs/LICENSING_REVIEW.md)核对实际内容。公开树和归档不含比赛视频、截图或叠加图；来源链接与 PTS 可公开，原始素材只留本机。
