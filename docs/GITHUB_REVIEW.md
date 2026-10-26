# 公开数据与离线审阅

本页说明可取得的数据、原素材来源和复现边界。项目试用见 [README](../README.md)，方法见[架构](ARCHITECTURE.md)与[算法](ALGORITHMS.md)，准确性要求见[验证说明](VALIDATION.md)。

<a id="public-cleanup"></a>
## 提交与推送前的资产整理

提交、推送前按当前公开策略核对实际依赖，清理重复资产版本、导入快照和可再生中间文件；公开树只保留当前资产及必要证据。历史原件归 Data 并保留路径、SHA 和引用映射，不误删原 UV、PTS、W、人工确认或失败记录；清理时同步核对代码与测试引用。本文的整理步骤不构成提交、推送或重写历史的授权。

本机文件分工与 Git 数据库位置见[维护说明](../MAINTENANCE.md)。以下公开快照和检查结论均保留其适用范围，不代替本次核对。

## 2026-10-26 重构进度的公开范围

该范围保留 native/Python 源码、共同参数状态、准备与回放入口、合同及相关测试。此次补入 GPU 连续求解、当前几何路径编译、取点模块的局部接线与已确认缺口、项目自有 GPU 环境的版本及验证清单。安装环境、cuDSS 等第三方运行库二进制及编译 DLL 仍只在本机；版本清单不等于随仓库附带安装包。唯一当前数值入口是[场地模型](../events/ankara_2026/scene/field/model.json)，当前模板/实例版本及未定位项由[场景索引](../events/ankara_2026/scene/README.md)维护，不在公开说明复制库存。

历史参考模板、导入快照及旧回执归本机 Data；[归档映射](../events/ankara_2026/scene/archive.json)保存旧路径、SHA、字节数及 Data 地址。当前运行只消费场景明确登记的资产，不要求历史原件存在。

Ad1/Ad14 的两份相机定义，以及新增广告使用的 [S01/S02/S04/S05 相机定义和小型控制数组](../events/ankara_2026/programmes/okada_mei_sp/ad_cameras/)，均随仓库提供并使用相对控制数组引用。Ad3 的 [7009 行数值源库](../events/ankara_2026/programmes/okada_mei_sp/ad3_sources/rows.npz)及[对应表](../events/ankara_2026/programmes/okada_mei_sp/ad3_sources/correspondence.json)、Ad7 的 [2192 行数值源库](../events/ankara_2026/programmes/okada_mei_sp/ad7_sources/rows.npz)及[对应表](../events/ankara_2026/programmes/okada_mei_sp/ad7_sources/correspondence.json)保留原 UV、PTS 和来源身份，不含广播图像缓冲区。其余已选轮廓读数保留在各自资产或来源索引中。它们是已有证据与候选，不是完整统计求解输入，不因公开或工程测试通过而自动激活拟合。

比赛原片、截图、叠加图、诊断视频、完整检查点和日志不进入本次提交。完整第三方广告图稿也不随本次分发，保留来源 URL、SHA 与本机历史地址；当前数值模型不依赖重新载入原图稿。历史绝对路径用于追查。当前模型和小型参数随公开源码提供；历史参考模板按归档映射在已配置的 Data 中查询，不再重复放在公开树，也不会据历史地址自动重放生产链。

观测人工确认的精确范围见[节目记录](../events/ankara_2026/programmes/okada_mei_sp/README.md#有界结果与人工确认)，结构、尺寸与 D3 待审状态见场景索引；公开、导入或工程测试不扩大这些确认。

公开包可检查数值定义及 native 前向，原像素和视频核对仍需要本地素材。求解实现边界见 [REBUILD](REBUILD.md)，完整节目范围见[节目 scope](../events/ankara_2026/programmes/okada_mei_sp/rebuild_scope.json)。公开完整性、覆盖完整与拟合准确分别判断。

## 2026-10-27 重构进度的公开范围

本次沿用上述媒体与安装环境边界，补入 native 相机／场景联合导数、局部参数图、物理域和逐点像素目标的求解修正，以及来源读取、身份关联、原始支持回读与相关测试。历史 solver_runtime 清单保留，新公开源码身份及等价性记录另行登记，不用当前 SHA 冒充原件身份。跨进程完整 checkpoint/resume 与完整节目自动取点仍未交付。

当前场景继续只由 `field/model.json` 及其引用确定，13 个模板采用完整印刷图案版式；S01 激活的内部坐标为 20。各次旧模板、迁移快照、完整点输入、候选状态、诊断和源画面仍留 Data；公开树保留当前模型、必要的逐点身份修正记录和原始 UV／整数 PTS／来源身份，不分发这些运行产物。已完成的 S04 广告包只公开脚本和小型来源索引，完整包及检查视频按索引留本机。

这些公开内容记录工程进度及已知失败，不扩大人工确认，也不表示相机物理、整镜头画面或 S01–S09 已通过验收。精确能力见 [REBUILD](REBUILD.md)，当前输入及历史修正见[节目记录](../events/ankara_2026/programmes/okada_mei_sp/README.md)。

## 官方视频与源时间

项目不分发 ISU 比赛视频、截图、叠加图或内嵌画面。可依据以下官方 YouTube 来源和源时钟自行重建视觉线索：

| 用途 | 官方来源 | 时间与身份 |
|---|---|---|
| 冈田芽衣短节目的现行处理源 | [ISU Skating：Mei OKADA / Ankara 2026](https://www.youtube.com/watch?v=O6j8L8TmfKo)、[从 00:20.020 附近观看](https://www.youtube.com/watch?v=O6j8L8TmfKo&t=20s) | 50/1 fps，time_base=1/12800；首帧 PTS 256256 = 00:20.020 |
| 首镜头取点范围 | [同一单节目视频](https://www.youtube.com/watch?v=O6j8L8TmfKo&t=20s) | 1248 帧，首/末帧 00:20.020–00:44.960，PTS 256256–575488 |
| 九镜头离线失败快照范围 | [同一单节目视频](https://www.youtube.com/watch?v=O6j8L8TmfKo&t=20s) | 8434 帧，首/末帧 00:20.020–03:08.680，PTS 256256–2415104 |
| 历史全场源对齐事实 | [ISU Skating 全场来源](https://www.youtube.com/watch?v=oxXu1avhrvs&t=14916s) | 04:08:36.283333 的旧帧对应单节目 00:20.020；仅保留历史对齐，不作新的节目处理源 |

官方频道身份及更精确的源哈希、PTS、表征差异和缺口保存在[源选择记录](../events/ankara_2026/programmes/okada_mei_sp/primary_source_selection.json)。YouTube 链接的整数秒仅帮助定位，实际重建以本地文件的原生 PTS/time_base 为准；不能用固定秒差或平均 FPS 取代已核实的帧对应。相同 URL 的不同下载表征不预设解码字节相同。

## 选定的失败快照

[节目 review](../events/ankara_2026/programmes/okada_mei_sp/review/README.md)冻结于 2026-10-19。它保留真实九镜头失败状态、模型、配置、材料映射、完整状态、源观测、统计 W、资格召回和必要驱动。该候选没有通过完整镜头定位验收，最后工程求解也尚未消费原始 W 的统计似然。

归档清单把逻辑路径及历史地址映射到实际 ZIP 成员。模型和关键 JSON 可直接阅读，其余数据按需读取，完整 ZIP 仍随仓库提供：

~~~console
python -B scripts/read_offline_review.py --check
python -B scripts/read_offline_review.py --get "Data/runs/okada_rec_1019/demo_grid_recovery_v1/candidate/model.json"
~~~

仅需 Python 标准库，无需本机 Data、求解环境或网络。--check 核对 ZIP 目录、成员和大小，不批量展开或重新散列全部内容；--get 查询实际位置及身份。要取出单项，追加 --output 和一个新的绝对文件名；批量提取使用 --extract 和仓库外的新绝对目录。工具拒绝批量展开到项目目录或覆盖已有目标，不执行包内代码。

Python 也可导入 Review，使用 read(原路径或逻辑路径) 取得字节。提取的 resource_map.json 对应原路径与短文件名；数组 dtype、shape、offset 和行语义须按原 packet/manifest 解释。原生文本 checkpoint 即使以 .json 结尾，也不能直接当 JSON。

后续源数据与该快照分开：[S02–S04](../events/ankara_2026/programmes/okada_mei_sp/source_datasets/dense234_20261020/result.json)、[S05–S06](../events/ankara_2026/programmes/okada_mei_sp/source_datasets/dense56_20261020/result_v4.json)、[共享源记录绑定](../events/ankara_2026/programmes/okada_mei_sp/source_datasets/storage_20261021/bindings.json)。这些修订和[恢复初值](../events/ankara_2026/programmes/okada_mei_sp/fit_joint9/rec1019/artifacts.json)不等于通过验证的新物理解；大规模源点流和数据库未包含在 10 月 19 日的 ZIP 内。

<a id="joint6-20261021"></a>

## 第1–6镜头：停止状态与系统审核（2026-10-21）

该次六镜头运行已按当时用户要求停止，状态和精确输入见[六镜头状态](../events/ankara_2026/programmes/okada_mei_sp/fit_joint6/status.json)及[源时钟范围](../events/ankara_2026/programmes/okada_mei_sp/fit_joint6/scope.json)。它是完整九镜头的子集，S01 源点确认不等于 S01 模型通过。

历史 GPU 确实执行，候选尚未收敛或验收。撤回的观测资格数不是错点率，其余 active 也不等于全部准确；成本下降与局部改善不能消除已有实体退化。源资格、工程支持边界、广告模板真值假设及停止策略的问题按该记录保留，不由此断言全部新点错误或共同模型无解。

该候选采用 observed-row 粗初始化，真实 RS 验收未通过；保存 observed 检查点不代表存在严格续跑入口。待验证匹配不因文件存在变为拟合输入。历史停止和当时的后续建议不授权今天重启或安排阶段。

公开树提供该次代码与状态；完整源数值、检查点、日志、原片证据和运动记录在本机私有补充包，按 status 的代际身份查询。历史 SHA/地址为追查键；不声称仅克隆公开树即可复跑该次六镜头。

## 数据身份与缺失材料

[manifest.json](../events/ankara_2026/programmes/okada_mei_sp/review/manifest.json)的 public_withheld 明确列出未分发的比赛原图和叠加图，保留其源地址、SHA、PTS 与来源事实。原片、批量帧图、可重建页面、安装环境、编译产物和大量运行缓存不随公开仓库提供。

测量 JSON/GZ/NPZ、原始 UV、统计 W、资格记录和模型身份保留；文件齐全不代表观测已确认或模型准确。仅凭这些数值不能复核原像素、颜色剖面或完成独立整镜头视觉验收。场地图和二值分类掩膜与原始广播画面分别登记，权利边界见[许可核查](LICENSING_REVIEW.md)。

[公开派生身份表](../configs/privacy_derivatives.json)记录隐私整理后的文件。历史 SHA 继续描述原件，公开 SHA 描述当前派生字节，两者不能互相冒充；拟合、续算和 producer 的原身份检查仍有效。JSON 中的历史本机绝对地址通常是查询键或 provenance，不要求读者拥有该盘符，也不表示完整运行路径已可移植。

完整拟合、GPU、人体和原生后端仍须按[环境说明](../environments/README.md)准备。models/rink_vision.pt 使用 Git LFS；指针存在不等于 payload 已取得，技术审阅 ZIP 则是普通 Git 文件。
