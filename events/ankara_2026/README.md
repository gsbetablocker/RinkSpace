# 安卡拉 2026：赛事资产说明

项目整体结构见[架构说明](../../docs/ARCHITECTURE.md)，公开结果见[节目审阅快照](programmes/okada_mei_sp/review/README.md)。本目录只保存本赛事的数值、来源与适用范围，不能把这些数值直接当作其他冰场的标定。

| 目录或文件 | 含义 |
|---|---|
| [manifest.json](manifest.json) | 2026-10-06 注册的模型和人工事实快照；原完整检查点位置作为来源保留 |
| [asset_preservation_manifest.json](asset_preservation_manifest.json) | 本次 105 项参数／人工事实保全登记及嵌套外部依赖，保全副本不自动成为当前输入 |
| [asset_preservation_manifest_20261013.json](asset_preservation_manifest_20261013.json) | strict v6／q32 引用链的增量保全：新增 25 个 JSON、复用 89 项；与首轮清单有重合 |
| `models/registered/` | 已注册模型、拟合信息与必要输入；注册不等于准确性验收 |
| `models/working/` | 已保全工作模型导出；检查点及源 profile 仍需对应 Data 材料 |
| `models/initializations/` | 从已声明输入保全的初始化参数（如存在）；不冒称最终模型 |
| `model_constraints/` | 用户确认的场地、广告、机位安装等事实与范围 |
| `observations/source_packets/` | 本次补充保全的源包及人工事实，含来源／散列与仍外部依赖清单 |
| `tools/` | 赛事特有的源装配或诊断辅助代码，公共数值内核仍复用 calibration |
| [strict_source_authorities/gray_review_20261013](tools/strict_source_authorities/gray_review_20261013/README.md) | 三份冻结来源方法源码及 receipt 身份保全；未切换运行路径，不自动认证源点准入 |
| `programmes/okada_mei_sp/` | 节目运行 profile、相机与模型绑定、审阅状态和节目人工约束 |
| `programmes/okada_mei_sp/source_datasets/` | 冻结源清单与交付摘要；source pack、支持索引和过滤大数组保留在 Data |

2026-10-13 已整理的 [v6 源清单](programmes/okada_mei_sp/source_datasets/current_review_20261013/specific_inner_review_v6/fit_dataset.json) 登记 1,282,102 个 FIT 候选、2,368 个 HELD；待核支持与待核归属不进入 FIT。镜头、曝光和缺失分母见 [交付摘要](programmes/okada_mei_sp/source_datasets/current_review_20261013/specific_inner_review_v6/delivery_summary.json)。该冻结版本的源数据交付、后续求解接入、优化器接受状态及整镜头准确性验收分别追踪；本页不改写冻结记录或选择新模型。

场地拟合只允许原片核实属于冰面或围挡本体的静止实体与材料。人物、看台、字幕、倒影等范围外点及归属未确认点排除，不能借未知深度参与。待核点保留来源，不自动视作已确认的负例。

原片在 Data 的 `events/ankara_2026/media/`；大规模求解运行、数组及完整检查点在对应 `calibration/runs/`；节目视频和运行历史在 Data 的 `programmes/okada_mei_sp/`。Data 原件保持原位，旧文件名或 `history` 标签不代表可以删除。

增量副本保留原字节及内部地址，不切换活动输入。完整相机种子状态、原 source pack／support index 和运动大数组仍外置；环境、原生库与路径／producer 的剩余条件见 [环境说明](../../environments/README.md)。

本赛事当前仍是开发证据。模型、真实源时间、人工事实与失败都须追溯到各自文件；后续 programme 状态会继续变化，本页不写死“最新”模型或数值。精度与完整镜头的验收标准见 [验证说明](../../docs/VALIDATION.md)。
