# 公开数据与离线审阅

本页说明仓库内容、原素材来源和复现边界。试用见 [README](../README.md)，能力见 [REBUILD](REBUILD.md)，准确性见 [VALIDATION](VALIDATION.md)。

<a id="public-cleanup"></a>
## 公开资产与存储边界

公开树保留源码、场景定义、小型参数、来源索引、必要数值观测和选定的冻结审阅包。[赛事入口](../events/ankara_2026/README.md)登记场景与两套节目结果。默认分支 `rebuild` 对应当前开发版，`main` 为 2026-10-24 旧版。各版本以对应 Git 提交收录的绑定为准。

比赛视频、截图、叠加画面、完整第三方广告图稿、安装环境和编译库留在本机 Data。运行输入、检查点、逐轮日志及诊断产物由结果绑定定位。下文的冻结离线包另含指定的工程失败证据。

[独立 PTZ 检验工具](../tools/ptz_verifier/README_zh.md)公开源码、测试、配置示例和历史试跑摘要；原私人交付的比赛配对图、特征缓存、图像报告及本机交接文件不进入公开树。其试跑结果为证据不足，不能作为主相机参数的认证。

历史模板、导入快照及安装回执通过[归档映射](../events/ankara_2026/scene/archive.json)保留旧路径、原始 SHA、字节数和 Data 地址。当前运行只消费明确登记的资产；历史绝对路径用于追查。存储解析见[维护说明](../MAINTENANCE.md)，来源与分发限制见[许可核查](LICENSING_REVIEW.md)。

公开前核对实际依赖，移出重复资产和可再生中间文件，并同步核对代码与测试引用。原始 UV、整数 PTS、统计 W、人工确认和失败证据保留在对应数值记录或归档中。原件 SHA 与[公开派生身份](../configs/privacy_derivatives.json)分别描述各自字节，不能互换。整理资产不授权提交、推送或改写历史。

人体公开内容包括实现源码、测试、播放器配置、小型控制参数与来源索引。参数范围涵盖全身、足鞋、开头、动作初值、根参考和旋转对照，具体字段、SHA、dtype 与 shape 见[参数索引](../events/ankara_2026/programmes/okada_mei_sp/body_parameters_1103_fields.json)及[许可核查](LICENSING_REVIEW.md#body-parameters)。当前候选、失败与人工反馈由[节目结果导航](../events/ankara_2026/programmes/okada_mei_sp/README.md#人体与足鞋结果)维护；公开参数不表示准确性验收通过。

完整拟合输出、SMPL-X 模型与完整表面、网络原始输出、原图、叠图和播放器缓存保留在本机 Data。历史绝对路径登记当时的运行身份；重放需要配置 Data 根并取得相符模型和输入。运行日志、调试报告及文档原文按已有运行目录与归档追查。

## 当前模型与数值来源

2026-11-13 的[赛事登记](../events/ankara_2026/README.md#暂定最终版本)收录共同冰场 `shared_joint_review_1113`、金书贤 24 镜头 `stability_review_1113` 与冈田芽依 9 镜头 `joint_review_1113`。两套节目各自绑定源 PTS、相机、数值状态与本地回放，独立物理标定和整片视觉验收仍待完成。

[共同冰场与相机复核源码](../events/ankara_2026/tools/joint_review_1113/README.md)和[冈田芽依切镜与首旋源码](../events/ankara_2026/programmes/okada_mei_sp/tools/body_review_1113/README.md)记录该轮的实现与来源。金书贤较早的相机试验可通过[历史归档索引](../events/ankara_2026/programmes/jin_shuxian_sp/camera_trial_archive_1113.json)在 Data 核对，各节目当前控制参数随版本登记提供。

广告的小型相机与控制数组见 [ad_cameras](../events/ankara_2026/programmes/okada_mei_sp/ad_cameras/)。Ad3 的[数值行库](../events/ankara_2026/programmes/okada_mei_sp/ad3_sources/rows.npz)／[对应表](../events/ankara_2026/programmes/okada_mei_sp/ad3_sources/correspondence.json)，以及 Ad7 的[数值行库](../events/ankara_2026/programmes/okada_mei_sp/ad7_sources/rows.npz)／[对应表](../events/ankara_2026/programmes/okada_mei_sp/ad7_sources/correspondence.json)保留源 UV、PTS 与身份，不含广播图像缓冲区。来源候选和参考轮廓不因公开、导入或测试通过而自动获得拟合资格。

源范围见[冈田芽依范围](../events/ankara_2026/programmes/okada_mei_sp/rebuild_scope.json)与[金书贤范围](../events/ankara_2026/programmes/jin_shuxian_sp/rebuild_scope.json)。原像素与整片贴合按[验证说明](VALIDATION.md)用对应素材核对。

## 官方视频与源时间

| 节目 | 官方单节目来源 | 拟合处理区间（原片首末帧） |
|---|---|---|
| 冈田芽依短节目 | [ISU Skating：Mei OKADA](https://www.youtube.com/watch?v=O6j8L8TmfKo) · [源选择记录](../events/ankara_2026/programmes/okada_mei_sp/primary_source_selection.json) | 00:22.120–03:08.680，PTS 283136–2415104，8,329 个曝光 |
| 金书贤短节目 | [ISU Skating：Shuxian JIN](https://www.youtube.com/watch?v=SW1IewLFUyI) · [源选择记录](../events/ankara_2026/programmes/jin_shuxian_sp/primary_source_selection.json) | 00:32.300–03:20.720，PTS 413440–2569216，8,422 个原生帧 |

两个单节目源均为 1920×1080、50 fps，时基为 `1/12800`。全场回放的帧对应、源哈希与下载表征由各节目源选择记录维护。冈田 2026-10-19 冻结失败快照使用历史范围 00:20.020–03:08.680，共 8,434 帧。


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

撤回的资格数不等于错点率，其余 active 也不表示全部准确。公开树保留代码与状态，大型数值输入、日志、检查点及原片证据在对应 Data 补充包中。六镜头试验的覆盖和失败结论仅适用于该次冻结输入。

## 复现限制

数值观测、模型身份与资格记录可用于技术审阅；原像素、颜色剖面和完整镜头视觉验收需要对应素材。场地图、分类掩膜与广播画面按各自来源登记。完整拟合、GPU 与人体能力还需[相应环境](../environments/README.md)；`models/rink_vision.pt` 使用 Git LFS，指针存在不等于 payload 已取得，技术审阅 ZIP 则是普通 Git 文件。
