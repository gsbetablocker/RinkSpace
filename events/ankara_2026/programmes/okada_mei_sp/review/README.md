# 冈田芽衣全九镜头：GitHub 离线技术审阅

冻结日期：2026-10-19。此快照记录当时停止后的全九镜头失败状态，尚未完成拟合与视觉验收；它不描述后续开发的实时状态。

先读 [CURRENT_STATUS.json](CURRENT_STATUS.json) 与[项目阅读说明](../../../../../docs/GITHUB_REVIEW.md)。当前模型直接见 [model.json](model.json)，执行配置见 [config.json](config.json)，材料/来源定义见 [sources.json](sources.json)、[material_map.json](material_map.json)，资格边界见 [source_qualification.json](source_qualification.json)。这些 JSON 内的原本机地址保留为来源记录；实际数据可通过本目录清单离线读取。

## 当前事实

- [ISU Skating 官方单节目](https://www.youtube.com/watch?v=O6j8L8TmfKo&t=20s)，50fps、time_base=1/12800，PTS 256256–2415104，8434帧（首/末帧 00:20.020–03:08.680）。
- 最后完整失败态为 demo_grid_recovery_v1。模型、完整CP、外层状态、lambda/nu、显式station/row/alpha/path及原始重放数组都在归档中。
- engineering_checkpoint.json 实际以 RINKFIT_AL_CHECKPOINT_V1 开头，是原生文本格式，不能当 JSON 解析或因后缀而删掉。
- current g9_ice 为907603行。当前候选的249446活跃行不等于249446条已获人工确认、独立同分布的证据。
- 当前packet的18个数组与原始 W 依赖共1622个唯一物理文件已纳入。它们的dtype、shape、offset、分组和源行身份仍按原packet解释；字节齐全不表示最后工程优化已消费原W统计似然。
- 最新状态仍有约束债务、RS/Query与首镜头保持问题，没有新接受的完整网格视频；召回和未知资格不因保全而恢复。

## 少量归档，包含关键实物

[manifest.json](manifest.json)逐项登记逻辑路径、原地址、SHA、角色、实际ZIP和成员。2286个逻辑条目按内容身份复用为2266个归档成员；没有在OneDrive批量展开文件，也没有放入完整对话、原视频、环境、批量帧或性能缓存。

| 归档 | 压缩大小 | 实际成员 |
|---|---:|---:|
| [baseline.zip](baseline.zip) | 93.65 MB | 250 |
| [code.zip](code.zip) | 0.35 MB | 179 |
| [evidence.zip](evidence.zip) | 52.31 MB | 141 |
| [observations.zip](observations.zip) | 17.58 MB | 10 |
| [source_arrays.zip](source_arrays.zip) | 31.72 MB | 18 |
| [state.zip](state.zip) | 54.00 MB | 64 |
| [weights.zip](weights.zip) | 79.40 MB | 1604 |

baseline保留v98原始输入失败态；state保留最后工程候选、其前后比较和实际初始状态；observations/source_arrays/weights分别保留源packet、18数组、原W与所引用数组；evidence保留源资格账本、首镜头与Demo参考、已知失败记录；code保留Data原先独有的恢复driver及实际后端版本源码/recipe。

原有公共源码在仓库 calibration/ 与 src/ 直接审阅，不重复拷入归档。此处另外外露四份核心装配源码供GitHub直接阅读：

- [materialize_direct_screen_grid.py](materialize_direct_screen_grid.py)：屏幕Demo数据的原始装配；屏幕对应不是物理相机初值。
- [run_demo_grid_recovery_v1.py](run_demo_grid_recovery_v1.py)：最后失败运行的真实启动装配。
- [append_g9_ice.py](append_g9_ice.py)：最近源追加的方法。
- [build_v16.py](build_v16.py)：该运行后端构建方法。完整历史driver在code.zip，不能将保全副本视为已切换运行路径。

在仓库根运行 python -B scripts/read_offline_review.py --check；按原路径或逻辑路径查询用 --get。只需一项时加 --output 新绝对文件名；需要批量提取时用 --extract 仓库外的新绝对目录。工具不访问原Data路径，不自动运行包内代码，批量提取拒绝仓库目录，避免OneDrive生成大批文件。

离线审阅可检查实现、模型/观测/统计定义与失败原因。完整真实镜头准确性验收仍需官方原视频；本次没有恢复求解、训练、统计或视频生成。

公开包不分发比赛视频、截图或叠加图，已排除25张原图及叠加图；旧路径和SHA登记于manifest的public_withheld；Data保留原归档。测量JSON、NPZ、GZ、18个观测数组、原W与模型未删改。原片视觉核验仍需本地原素材。
