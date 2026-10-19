# Git 公开内容准备

## 2026-10-19：单目录公开准备完成

当前 OneDrive 工作目录已就地脱敏，私人原件与旧 Git 数据库完整保全于本地 Data；临时 public_snapshot 已移除，不维护第二份项目。公开仓库为[gsbetablocker/RinkSpace](https://github.com/gsbetablocker/RinkSpace)，采用同一工作目录正常提交和推送。.gitignore 排除本机配置、凭证及编译产物；模型 LFS 配置可用。实际数据、身份及检查边界见 [PRIVACY_REVIEW.md](PRIVACY_REVIEW.md)。

## 2026-10-19：整理前的阅读与技术审阅准备记录

当前公开review含2286个逻辑资源、2266个唯一成员，25张归档比赛图片已排除；两张本地机位JPG由ignore排除，测量记录与统计数据保留。以下保留整理前盘点；当前以本页首节和隐私检查说明为准。阅读入口为 [GITHUB_REVIEW.md](GITHUB_REVIEW.md)。[节目审阅目录](../events/ankara_2026/programmes/okada_mei_sp/review/README.md) 已把2312个逻辑条目保全为7份ZIP（348,847,718字节），附可读模型、配置、状态、来源/材料映射和四份装配代码，目录共19文件。归档包含当前完整失败态、1622份唯一typed array/W依赖、资格账本、原像素例证和旧基线；当前生产源码复用仓库原目录，不重复复制整个历史。批量提取工具拒绝输出到仓库树。

2026-10-19检查：现有index、HEAD和本地refs可达历史没有超过100MiB的普通blob；Git LFS可用，models/rink_vision.pt受LFS属性覆盖，但未暂存为新指针；当前无remote。没有改写历史、暂存、提交或推送。旧的20个已跟踪编译产物及本机配置仍原位保留，不能因ignore规则存在而声称已取消跟踪。

Data中的关键构建源码已原字节保全到 [tools/build_event_backend.py](../events/ankara_2026/tools/build_event_backend.py)，joint_solver现从仓库定位它。Data原先独有的恢复driver和实际后端版本源码另在code.zip中，关键四份直接可读。完整安装环境、编译器与库仍按环境定义准备；保全不是跨机器完整运行验收。

机器检查报告在本机Data的 maintenance/github_preparation_20261019/，这些诊断JSON不再复制到OneDrive。凭证扫描必须按逐次报告的文本上限/跳过项解释；未完成的全面文本或历史凭证审计没有被标成通过。归档构建对实际读取的4MiB以下文本做了模式扫描，较大的冻结JSON单列跳过。

下文2026-10-13内容保留为当时的盘点记录，不能替代本节快照。


返回结果入口：[START_HERE.html](../START_HERE.html)。目录与资产归属见 [REPOSITORY_GUIDE.md](REPOSITORY_GUIDE.md)，环境现状见 [environments/README.md](../environments/README.md)。本文记录当前内容清单与公开前尚需判断的项目，不改变活动运行路径，也不选择或发布模型。

## 当前快照说明

[audit_public_repository.py](../scripts/audit_public_repository.py) 新增 `--metadata-only` 模式，仅通过 `git ls-files`、`git check-attr` 和文件元数据核对索引路径、当前候选路径、属性和工作区大小。输出报告仅保存在外部 Data；本模式要求显式传入 `--data-root`，因此也不读取 `configs/storage.json` 的内容：

```powershell
& .\.venv\Scripts\python.exe -B scripts\audit_public_repository.py --metadata-only `
  --data-root D:\CodexData\RinkSpace_Data `
  --output D:\CodexData\RinkSpace_Data\maintenance\repository_preparation_20261011\public_repository_metadata_20261013.json
```

本模式不枚举历史 Git 对象，不查询索引 blob 大小，不读取 blob、LFS pointer、权重、数组或媒体内容，不扫描凭证或正文中的绝对路径，不计算候选文件散列，也不读取资产保全 manifest。报告明确记录每项检查未执行，不沿用旧报告的通过或零命中结论。文件分类只来自路径和扩展；人工事实 NPZ 仍是需要按语义审阅的候选，没有被扩展规则排除。

脚本不暂存、提交、推送、改 Git 配置或属性，不改活动源码、profile、状态或原生文件，不运行求解器，不加载 DLL，不解码视频。被忽略的 `configs/local.json` 只记录存在与候选状态，不读取其内容。既有完整模式仍可作正文和 Git 历史大小审计，本轮没有运行；旧 `public_repository_audit.json` 只是 2026-10-11 的历史记录。

2026-10-13 10:29:11.322317（上海时间）的 metadata 快照读取到 620 个索引路径和 1401 个未忽略的未跟踪文件，共 2021 个候选路径。其中 1086 个源码候选尚未跟踪。工作区可获取元数据的候选为 2008 个，另 13 个索引路径在工作区缺失。新增文档和活动会话会继续改变这些数字，最终提交应以重新生成的报告为准。候选集合只是核对范围，不表示其中所有文件都应公开，也不代表被忽略的文件都是可丢弃资产。

| 已核对事项 | 此次 metadata 结果 | 含义 |
|---|---|---|
| 索引 blob 大小 | 未检查 | 仅记录索引路径、对象 ID、stage 和有效属性 |
| 历史对象及大小 | 未检查 | 没有执行 `rev-list` 或 `cat-file`，不作历史大小结论 |
| 工作区超过 50 MiB 的候选 | 1 | `models/rink_vision.pt` 为 113,742,905 字节，超过 100 MiB |
| 已跟踪的原生编译产物 | 20 | 4 个 DLL、8 个 lib、8 个 exp，需逐项判断保留方式 |
| 凭证与正文绝对路径 | 未扫描 | 没有零命中结论，不是公开安全认证 |
| NPZ／NPY 数组候选 | 22 | 仅读文件元数据，未判断人工事实、参数或中间数组角色 |
| 项目根 LICENSE／COPYING | 缺失 | 尚未选择项目许可证；第三方材料许可另行核对 |

大小比较沿用 2026-10-11 所查的 GitHub 普通文件阈值：超过 50 MiB 警告、超过 100 MiB 阻止；1 MiB = 1,048,576 字节。本轮 metadata 模式没有联网复核这些规则。[GitHub 官方说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

## 核心资产候选的判断依据

| 内容 | 当前记录方式 | 公开前还需确认 |
|---|---|---|
| `src/`、`calibration/`、`scripts/` 的公共和必要源码 | 列出精确路径、索引状态、当前大小 | 完整 import／资源依赖、是否仍是所选运行路径、上游来源与许可 |
| 赛事与节目的 JSON 参数、模型、源包、人工事实 | 此次记录路径和元数据，未读保全 manifest 的 roles | 参数身份、来源和准确性状态、引用的外部材料能否提供 |
| 人工笔划和原始支持 NPZ | 保全 manifest 可作为后续不可替代事实的角色依据，本轮未读 | 保留人工确认与源时钟，审阅对应来源；不按 `.npz` 扩展排除 |
| 数组候选 | 均标记为需要语义审阅 | 分清最终参数、人工事实、运动大数组、完整检查点与可重算产物 |
| `models/rink_vision.pt` | 精确大小与当前 LFS 属性分别记录 | 权重身份、来源、许可、实际 LFS pointer 与远端 payload |
| 文档、环境定义和构建 recipe | 跟踪状态及本机路径位置 | 可重建性、上游版本、平台要求及旁项目依赖 |

保全 manifest 的记录数和唯一目标路径数可能不同，同一路径可由多个 profile 引用，不能由两个数字的差异推断丢失。既有完整模式会合并相同目标的 roles；此次 metadata 报告不读取这些清单，不能把 22 个数组候选进一步认定为已核实人工事实或已验收参数。

## LFS 状态仍需实际确认

2026-10-13 10:29 的有效属性查询显示，`models/rink_vision.pt` 工作区属性为 LFS filter／diff／merge，索引属性为 unspecified，且该权重尚未跟踪。因此“工作区配置了 LFS”“Git 索引已存 pointer”“远端 LFS 已有 payload”是三项不同事实。

此次报告同时用 `git check-attr` 与 `git check-attr --cached` 查询工作区和索引的有效属性，没有读取 pointer 或 blob。没有暂存权重，没有读取整个权重做哈希，没有查询或上传远端 LFS。这次不进行历史迁移，也不因属性存在就声称超过普通文件限制的权重已可推送。

## 编译产物、本机路径和历史上下文

已跟踪的 20 个编译产物位于 `calibration/program_tracking/native/` 的 `quadratic_source_edge`、`round6_source_edge`、`time_principal_interval_source_edge`、`time_principal_source_edge` 子目录。现有 `.gitignore` 规则不会自动取消它们的已跟踪状态。DLL／lib 可能仍是旧 wrapper 或已认证状态使用的输入，整理阶段保留其原位与字节，不移动或删除。

此次报告只按文件名和目录为 agent、会话诊断、历史证据、状态与来源清单标记审阅原因；没有读取正文以查找本机绝对路径。这些条目可能是必要来源事实或兼容记录，不自动认定为秘密，也不自动删除。后续需分辨可移植运行地址、不可替代历史地址、个人信息与可公开说明；源时间、人工确认和失败状态应保留。

索引中有部分文件已从工作区删除，报告记录索引身份和工作区缺失状态，没有读取索引版本正文。当前没有恢复、取消跟踪或提交这些删除。完整模式的凭证候选仅输出文件、行号、类别与索引／工作区层，不输出匹配行或值；此次 metadata 模式完全未扫描凭证，不能引用旧报告的无命中作为当前结论。

这份报告用于把公开前的问题变成可审阅的清单。正在运行的会话可在读取期间或之后改变文件；正式公开前需重新核对最终提交、许可证与第三方权利、环境重建及外部数据获取说明，准确性则仍按 [VALIDATION.md](VALIDATION.md) 用真实完整镜头验收。
