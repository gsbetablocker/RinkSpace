# 项目维护导航

本页集中保存开发和项目管理入口。项目介绍见 [README.md](README.md)；结果唯一入口仍是 [START_HERE.html](START_HERE.html)。本页不另建结果页面，也不把历史任务记录当作当前开工授权。

## 状态、结果与协作

| 需要处理的事项 | 入口 |
|---|---|
| 项目 → 赛事 → 节目的当前状态、有效模型与结果 | [START_HERE.html](START_HERE.html) |
| 长期协作要求与文件操作约束 | [AGENTS.md](AGENTS.md) |
| 文件归属、保全记录与历史结构说明 | [仓库结构](docs/REPOSITORY_GUIDE.md) |
| 当前赛事事实、模型登记与节目归属 | [安卡拉赛事说明](events/ankara_2026/README.md)、[赛事清单](events/ankara_2026/manifest.json) |
| 场地与来源验证要求 | [验证说明](docs/VALIDATION.md)、[三维场地建模](docs/rink_model/BUILD_SPEC.md)、[场地验证](docs/rink_model/VALIDATION.md) |

## 本地开发与运行

| 需要处理的事项 | 入口 |
|---|---|
| 基础安装与本地工作台 | [INSTALL_LOCAL.cmd](INSTALL_LOCAL.cmd)、[START_LOCAL.cmd](START_LOCAL.cmd) |
| 原帧点位编辑 | [START_POINT_EDITOR.cmd](START_POINT_EDITOR.cmd)、[启动实现](START_POINT_EDITOR.ps1) |
| 公共求解器接口 | [solver_runtime](calibration/solver_runtime/README.md) |
| 现有样本的求解启动配置 | [START_FAST_SOLVER.ps1](START_FAST_SOLVER.ps1) |
| 环境保存、版本证据与重建限制 | [environments/README.md](environments/README.md)、[运行依赖](docs/RUNTIME_DEPENDENCIES.md) |
| 更新唯一结果入口 | [refresh_project_entry.py](scripts/refresh_project_entry.py) |

各启动器使用自己的既定配置，不代表完整模型已经验收；尤其样本求解配置不能直接套用到其他赛事。是否继续拟合、训练或生成视频，以当前明确授权和对应源验证为准。

## 单目录与存储

日常只维护当前 OneDrive 项目目录。源码、环境和构建定义、模型参数、人工事实以及必要离线审阅数据在项目内；原片、批量帧图、训练数据、大数组、完整检查点、缓存、日志与诊断视频在现有 Data 中。少量归档通过索引按需读取，不需要把所有逻辑资源展开成文件。

Git 数据库位于本机 Data，项目根仅保留指针。2026-10-19 已将私人原件与旧历史完整保全到本机 Data 的 maintenance/private_20261019/，原目录连接新的公开历史；没有长期维护的脱敏副本或额外同步步骤。日常仍在原目录开发，后续正常提交与推送；提交身份使用GitHub的noreply地址。

路径检查使用 [CHECK_PATHS.cmd](CHECK_PATHS.cmd) 或：

~~~console
python -B scripts/check_onedrive_paths.py
~~~

检查覆盖被 Git 忽略的安装环境与缓存。旧地址解析见[路径别名](configs/path_aliases.json)及[可选资源解析](docs/PORTABLE_RESOURCES.md)。公开派生身份见[privacy_derivatives.json](configs/privacy_derivatives.json)：历史 SHA 保留原件含义，不能把脱敏后的文件声称为原始运行字节；既有拟合与续算保护不会因此绕过。

## 公开与许可检查

| 需要核对的内容 | 入口 |
|---|---|
| GitHub 阅读、必要数据和离线读取 | [GITHUB_REVIEW.md](docs/GITHUB_REVIEW.md) |
| 单目录整理、隐私检查与原件保全 | [PRIVACY_REVIEW.md](docs/PRIVACY_REVIEW.md) |
| Git 候选、体积与 LFS 检查 | [PUBLIC_REPOSITORY_CONTENTS.md](docs/PUBLIC_REPOSITORY_CONTENTS.md)、[check_github_readiness.py](scripts/check_github_readiness.py) |
| 当前文本隐私检查 | [audit_worktree_privacy.py](scripts/audit_worktree_privacy.py) |
| 分项许可声明、第三方许可与未确认授权 | [LICENSE.md](LICENSE.md)、[LICENSING_REVIEW.md](docs/LICENSING_REVIEW.md)、[第三方资产](docs/THIRD_PARTY_ASSETS.md) |

隐私、体积与路径检查通过，不代表第三方代码、权重或赛事素材的公开再分发许可已确认。许可核查目前仍有明确未解决项。公开仓库：[gsbetablocker/RinkSpace](https://github.com/gsbetablocker/RinkSpace)。日常仍在此目录开发，正常提交并推送origin/main；没有第二份项目要同步。
