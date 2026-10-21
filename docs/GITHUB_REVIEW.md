# 公开数据与离线审阅

本页说明可取得的数据、原素材来源和复现边界。项目试用见 [README](../README.md)，方法见[架构](ARCHITECTURE.md)与[算法](ALGORITHMS.md)，准确性要求见[验证说明](VALIDATION.md)。

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

本轮已按用户要求停止，实际状态与精确输入见[六镜头状态](../events/ankara_2026/programmes/okada_mei_sp/fit_joint6/status.json)、[原生时钟范围](../events/ankara_2026/programmes/okada_mei_sp/fit_joint6/scope.json)。一套scene、三个rig、4568帧覆盖源20.020–111.380秒（末端不含）。S01经过人工审核的源点不等于S01模型已全部正确；当前完整投影依赖的冻结只是初始化阶段，未完成最终共享参数联合验收。

GPU确实执行。最新 `latent_joint_gpu_v1` 使用当前q7源代，完整1210861行、active1168579行；新增源955801行中21069行因实例／有限支持缺失等转为unknown，原UV保留。这是资格撤回数，不是错点率，剩余active也不等于全部准确。最后两次非零接受步的成本约3.847e9→1.360e9，但由首步旧实体退化触发停止回调，尚未数值收敛，也未被安装或验收。

已查实的系统问题包括：旧消费者把像素profile可重读升级为实体已确认；工程覆盖帽混作物理端点；曾把待估广告纹理坐标固定为世界真值；早期迭代的退化被直接作为停机条件。S03部分广告角点第二步恢复并优于基线，说明首步退化不能证明永远无法恢复；S02旧实体RMS仍6.196→8.051→8.113px，不能因此放松最终验收。未证实全部新取点错误、GPU失败或共同物理模型根本无解。

当前仍是observed-row粗初始化目标，真实RS验收未通过。S05–S06已有498204对真实静止围挡匹配，当前只用于验证，尚未安装成拟合目标；observed完整检查点也没有已实现的严格续跑入口。未知UV与广告chart规范、固定模板和信息权重之间的冲突须整体审核后决定下一阶段，不能默认重启最后候选。

公开仓库保存本轮代码与状态；完整源数值、检查点、运行日志、少量原片证据和真实运动记录放在本机私有PRO补充包，详见status中的代际身份。包内原始SHA与历史路径是审阅查询键；公开源码不附比赛画面，不声称仅克隆仓库就能复跑当前六镜头。

## 数据身份与缺失材料

[manifest.json](../events/ankara_2026/programmes/okada_mei_sp/review/manifest.json)的 public_withheld 明确列出未分发的比赛原图和叠加图，保留其源地址、SHA、PTS 与来源事实。原片、批量帧图、可重建页面、安装环境、编译产物和大量运行缓存不随公开仓库提供。

测量 JSON/GZ/NPZ、原始 UV、统计 W、资格记录和模型身份保留；文件齐全不代表观测已确认或模型准确。仅凭这些数值不能复核原像素、颜色剖面或完成独立整镜头视觉验收。场地图和二值分类掩膜与原始广播画面分别登记，权利边界见[许可核查](LICENSING_REVIEW.md)。

[公开派生身份表](../configs/privacy_derivatives.json)记录隐私整理后的文件。历史 SHA 继续描述原件，公开 SHA 描述当前派生字节，两者不能互相冒充；拟合、续算和 producer 的原身份检查仍有效。JSON 中的历史本机绝对地址通常是查询键或 provenance，不要求读者拥有该盘符，也不表示完整运行路径已可移植。

完整拟合、GPU、人体和原生后端仍须按[环境说明](../environments/README.md)准备。models/rink_vision.pt 使用 Git LFS；指针存在不等于 payload 已取得，技术审阅 ZIP 则是普通 Git 文件。
