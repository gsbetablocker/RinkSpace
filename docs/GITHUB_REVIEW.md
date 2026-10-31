# 公开数据与离线审阅

本页说明仓库内容、原素材来源和复现边界。试用见 [README](../README.md)，能力见 [REBUILD](REBUILD.md)，准确性见 [VALIDATION](VALIDATION.md)。

<a id="public-cleanup"></a>
## 公开资产与存储边界

公开树保留项目源码、当前场景定义、小型参数、来源索引、必要数值观测和选定的冻结审阅包。名义场景、冰面容差版本与资产库存见[场景索引](../events/ankara_2026/scene/README.md)；求解后的数值状态由对应候选绑定，输入和人工确认见[节目入口](../events/ankara_2026/programmes/okada_mei_sp/README.md#有界结果与人工确认)。

比赛视频、截图、叠加画面、完整第三方广告图稿、安装环境和编译库不随公开源码分发。大型运行输入、检查点、日志和诊断产物留在本机 Data；选定离线包的具体范围见下文。环境版本清单不等于附带安装包，公开源码也不保证所有本机实验可仅凭克隆复跑。

[独立 PTZ 检验工具](../tools/ptz_verifier/README_zh.md)公开源码、测试、配置示例和历史试跑摘要；原私人交付的比赛配对图、特征缓存、图像报告及本机交接文件不进入公开树。其试跑结果为证据不足，不能作为主相机参数的认证。

历史模板、导入快照及安装回执通过[归档映射](../events/ankara_2026/scene/archive.json)保留旧路径、原始 SHA、字节数和 Data 地址。当前运行只消费明确登记的资产；历史绝对路径用于追查。存储解析见[维护说明](../MAINTENANCE.md)，来源与分发限制见[许可核查](LICENSING_REVIEW.md)。

公开前核对实际依赖，移出重复资产和可再生中间文件，并同步核对代码与测试引用。原始 UV、整数 PTS、统计 W、人工确认和失败证据保留在对应数值记录或归档中。原件 SHA 与[公开派生身份](../configs/privacy_derivatives.json)分别描述各自字节，不能互换。整理资产不授权提交、推送或改写历史。

## 当前模型与数值来源

当前开发工作区已接通共同 native 几何、时变相机、prepare／review 和连续 GPU LM。当前候选覆盖 S01–S09 九镜头、三个物理机位，已执行真实联合拟合并导出完整回放；候选尚未数值收敛，完整原片视觉验收尚未完成。精确输入与状态见[节目结果绑定](../events/ankara_2026/programmes/okada_mei_sp/fit_joint9/native_joint9_guarded_lines.json)，完整磁盘恢复、全节目自动取点和准确性验证的缺口见 [REBUILD](REBUILD.md)。

广告的小型相机与控制数组见 [ad_cameras](../events/ankara_2026/programmes/okada_mei_sp/ad_cameras/)。Ad3 的[数值行库](../events/ankara_2026/programmes/okada_mei_sp/ad3_sources/rows.npz)／[对应表](../events/ankara_2026/programmes/okada_mei_sp/ad3_sources/correspondence.json)，以及 Ad7 的[数值行库](../events/ankara_2026/programmes/okada_mei_sp/ad7_sources/rows.npz)／[对应表](../events/ankara_2026/programmes/okada_mei_sp/ad7_sources/correspondence.json)保留源 UV、PTS 与身份，不含广播图像缓冲区。来源候选和参考轮廓不因公开、导入或测试通过而自动获得拟合资格。

数值定义、原像素核对、覆盖完整与定位准确分别判断。完整范围见[节目 scope](../events/ankara_2026/programmes/okada_mei_sp/rebuild_scope.json)；图像与视频核对需要本地对应素材。

## 官方视频与源时间

| 用途 | 官方来源 | 时间与身份 |
|---|---|---|
| 当前处理源 | [ISU Skating：Mei OKADA / Ankara 2026](https://www.youtube.com/watch?v=O6j8L8TmfKo) | 1920×1080，50/1 fps，time_base=1/12800 |
| 当前重构处理起点 | [同一单节目视频](https://www.youtube.com/watch?v=O6j8L8TmfKo&t=22s) | 00:22.120，PTS 283136；S01 处理至 00:44.960，共 1143 个曝光 |
| S01 物理镜头与历史取点范围 | [同一单节目视频](https://www.youtube.com/watch?v=O6j8L8TmfKo&t=20s) | 00:20.020–00:44.960，PTS 256256–575488，共 1248 帧 |
| 九镜头冻结失败快照 | [同一单节目视频](https://www.youtube.com/watch?v=O6j8L8TmfKo&t=20s) | 00:20.020–03:08.680，PTS 256256–2415104，共 8434 帧 |
| 历史全场源对齐 | [ISU Skating 全场来源](https://www.youtube.com/watch?v=oxXu1avhrvs&t=14916s) | 04:08:36.283333 的旧帧对应单节目 00:20.020，仅作历史对齐 |

源哈希、频道身份、表征差异与对应关系见[源选择记录](../events/ankara_2026/programmes/okada_mei_sp/primary_source_selection.json)。网页整数秒链接仅帮助定位，实际重建以本地文件原生 PTS／time_base 为准；不能用平均 FPS 或固定秒差替代已核实的帧对应。同一 URL 的不同下载表征不预设字节相同。

## 冻结九镜头失败快照

[节目 review](../events/ankara_2026/programmes/okada_mei_sp/review/README.md)冻结于 2026-10-19，保留九镜头失败状态、模型、配置、材料映射、完整状态、源观测、统计 W、资格召回和必要驱动。该候选未通过完整镜头定位验收，最后工程求解尚未消费原始 W 的统计似然。

归档清单把逻辑路径和历史地址映射到实际 ZIP 成员。模型与关键 JSON 可直接阅读，其余内容按需查询：

~~~console
python -B scripts/read_offline_review.py --check
python -B scripts/read_offline_review.py --get "Data/runs/okada_rec_1019/demo_grid_recovery_v1/candidate/model.json"
~~~

仅需 Python 标准库，无需本机 Data、求解环境或网络。`--check` 核对 ZIP 目录、成员和大小，不重新散列全部内容；`--get` 查询位置与身份。单项提取追加 `--output <新的绝对文件名>`，批量提取使用 `--extract <仓库外的新绝对目录>`；工具拒绝覆盖已有目标或批量展开到项目内，不执行包内代码。

Python 的 `Review.read(原路径或逻辑路径)` 可读取字节。`resource_map.json` 对应旧路径与短文件名；数组 dtype、shape、offset 和行语义按原 packet／manifest 解释。原生文本 checkpoint 即使以 `.json` 结尾，也不能直接当 JSON。

后续 [S02–S04](../events/ankara_2026/programmes/okada_mei_sp/source_datasets/dense234_20261020/result.json)、[S05–S06](../events/ankara_2026/programmes/okada_mei_sp/source_datasets/dense56_20261020/result_v4.json)、[共享源绑定](../events/ankara_2026/programmes/okada_mei_sp/source_datasets/storage_20261021/bindings.json)及[恢复初值](../events/ankara_2026/programmes/okada_mei_sp/fit_joint9/rec1019/artifacts.json)属于独立修订，不是已验证的新物理解。其大型点流和数据库不包含在该冻结 ZIP 中。

<a id="joint6-20261021"></a>
## 历史六镜头试验

2026-10-21 的六镜头运行已停止，精确输入和状态见 [status.json](../events/ankara_2026/programmes/okada_mei_sp/fit_joint6/status.json)及 [scope.json](../events/ankara_2026/programmes/okada_mei_sp/fit_joint6/scope.json)。历史 GPU 确实执行，候选未收敛或验收；observed-row 粗初始化没有通过严格滚动快门验收，保存检查点不代表具备严格恢复入口。

撤回的资格数不等于错点率，其余 active 也不表示全部准确。公开树保留代码与状态，大型数值输入、日志、检查点及原片证据在对应 Data 补充包中。它与当前九镜头候选分开解释。

## 复现限制

数值观测、模型身份与资格记录可用于技术审阅；原像素、颜色剖面和完整镜头视觉验收需要对应素材。场地图、分类掩膜与广播画面按各自来源登记。完整拟合、GPU 与人体能力还需[相应环境](../environments/README.md)；`models/rink_vision.pt` 使用 Git LFS，指针存在不等于 payload 已取得，技术审阅 ZIP 则是普通 Git 文件。
