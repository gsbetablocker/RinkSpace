# 冈田芽衣短节目：范围、结果与历史入口

本页索引节目事实与运行记录，赛事共同几何见[场景说明](../../scene/README.md)。具体工作由用户最新请求及所选输入／状态决定，历史启动命令和阶段锁定不作为当前执行指令。

## 源与范围

| 内容 | 权威记录 |
|---|---|
| ISU 官方 50fps 单节目原片、哈希、表征与历史源对应 | [primary_source_selection.json](primary_source_selection.json) |
| 完整 S01–S09、原生 PTS、处理范围及明确排除尾段 | [rebuild_scope.json](rebuild_scope.json)；源时基 1/12800，处理截止与物理切镜分别保存 |
| 当前共同场景、模板及版本 | [field/model.json](../../scene/field/model.json) |
| 三个独立机位的 Sony PXW-Z190 型号及安装事实 | [用户事实](../../model_constraints/camera_hardware/user_camera_mount_and_height_20261006_v4.json)、[安装配置](../../../../configs/camera_hardware/okada_sp_installations_z190_v1.json)；型号确认不等于现场读出／畸变已标定 |

镜头子集不代表完整节目完成。命令能力见 [REBUILD](../../../../docs/REBUILD.md)／[INTERFACES](../../../../docs/INTERFACES.md)，启动与续算见 [运行要求](../../../../docs/SOLVER.md#run-contract)，完整镜头判定见 [VALIDATION](../../../../docs/VALIDATION.md)。

## 有界结果与人工确认

| 记录 | 已做及未做 |
|---|---|
| [R1 单曝光](r1/exposure.json) | PTS 320000（25.000s）的完整读出域相机快照；保留原控制点，不是新几何拟合 |
| [R2 窗口摘要](r2/result.json)、[窗口](r2/window.json) | PTS 307200–332800（24–26s），101 个原生曝光；记录的最大投影差约 3.87×10⁻¹⁰ px，合法集合一致；验证前向等价，未通过完整 S01 准确性验收 |
| [R3 几何对照](r3.json) | 冻结 R2 相机，比较 24/25/26 秒的六条 R3/R4 既有轮廓；记录的是几何投影位移，非实测源残差，未确认改善或整镜头准确性；人工状态仍为 not_reviewed |
| [G1 GPU 原始残差](g1.json) | R2 同一 24–26 秒、101 帧；R3/R4 六条轮廓 5,920 条真实观测在 GPU 评估。CPU/GPU 最大差约 1.14×10⁻¹² px；原片对照与实际冷热时间／内存位于 Data/runs/g1_20261025。未更新几何、相机或 station，未执行白化／Jacobian／拟合；其它因子 unsupported，不改变人工验收。 |
| [G2 GPU 局部导数与块系统](g2.json) | G1 同一 101 帧、5,920 条观测；原相关组完整白化，连通 station 块累计及 CPU／GPU／有限差分检查通过。仅开放三个相机常量方向与原 station；192 个局部块秩亏已记录。原片小扰动、实际时间／内存在 Data/runs/g2_20261025。无阻尼、消元、参数更新或拟合改善声明。 |
| [G3 一次 GPU LM 候选](g3.json)、[保存状态](g3/state.json) | 沿用 G1/G2 同一 101 帧、5,920 条观测；阻尼、连通局部消元、cuDSS、回代和同目标接受／拒绝已贯通。唯一候选数值接受，目标 49,565.44 → 19,348.20，原像素残差中位数 53.23 → 33.86 px，75 帧下降、末段 26 帧上升，仍未完成拟合或人工验收。当前几何与源数据不变；新相机／station 单独保存。首次闭环 45.15 ms，全部 101 帧对照在 Data/runs/g3_20261025；报告字段失败后复用已保存候选完成交付，没有第二次真实候选。 |
| [当前 GPU 连续求解](../../../../native/README.md#当前模型的-gpu-求解入口) | 全部 624 条可定位 native 路径已进入共同残差与连续求解；工程输入验证 24 个相机控制坐标，几何参数保持当前值。旧 G4 六条靶圈真实连续尝试有分解失败及用户停止，没有最终收敛状态；日志保留 Data/runs/g4_20261025。当前取点数据集接入、真实节目拟合与跨进程完整恢复未交付；工程测试不改变 S01–S09 的验收状态。 |
| [Ad5 当前矩形对照](ad5_rectangle_review.json) | S05、PTS 1273856（99.52s），用户意见为“看不出新旧变化”；只确认该曝光的新旧矩形对照，不扩展到源定位精度、其他 PTS 或 Ad18，已有明确接受范围保留。 |
| [Ad2 O 边界确认](ad2_o_review.json) | 仅 PTS 320000 的两个既有记录的身份与读数；原 UV 未变，`active_for_fit=false`；不扩展到模板或相机拟合 |
| [既有取点流程恢复审计](source_reader_audit.json) | **未完成，按用户要求暂停，等待下一步指令。** 新材料候选到正式读数及下一帧引导的路径仍有实现断层，完整自动取点能力尚未恢复，并非只欠准确性验收。独立审阅核对本次恢复目录 21 份结果、89 帧次，仍只有 24.10–24.28 秒的同一 10 帧；入口首帧带人工引导、每帧带已保存遮挡掩膜。32.50–32.52 秒、44.80–44.82 秒无当前帧人工点／掩膜的启动检查各有形状候选，四帧正式冰面读数及精读查询均为 0；此检查不等于整段连续评测。整段 S01（1248 帧）的可见材料漏读分母、定位及身份错误尚未建立。此前局部修复、204 项测试、原有 10 组冰面保留及 55 个几何文件一致的证据保留，不作为完整恢复证明。当前 native、原 UV、整数 PTS、相关组与人工确认范围保持，切镜恢复后置；本次只记缺口，未修代码或启动运行。 |
| [O0.1 原片取点对照（已撤销）](o01/result.json) | 用户于 2026-10-25 撤销整个 O 工作单；保留原卡与失败，不再请求逐点或整体重复验收。已有人工标注和批准范围不变。 |

R1／R2 的本机画面、行状态和诊断在 `D:\CodexData\RinkSpace_Data\runs\r1_20261025`、`runs\r2_20261025`。R2 记录已查看 101 帧顺序联系表，未声称播放器连续播放；两项都没有启动拟合。数值来自保存的结果记录，本页不是新一次重算回执。

<a id="legacy-maintenance"></a>
## 历史索引

| 历史对象 | 追查入口与边界 |
|---|---|
| d78 原版、完整状态、恢复与后续视觉拒绝 | [s01_original/index.json](s01_original/index.json)；原模型及失败候选保留，不自动续跑 |
| 已停用控制台及旧 solve2 绑定 | [solve2/input.json](solve2/input.json)；旧 8787 服务说明不是启用指令 |
| 六镜头试验与停止状态 | [fit_joint6/status.json](fit_joint6/status.json)、[scope.json](fit_joint6/scope.json)；不是完整九镜头验收 |
| 2026-10-19 公开失败快照 | [review/README.md](review/README.md)；离线读取方法见 [GITHUB_REVIEW](../../../../docs/GITHUB_REVIEW.md) |

2026-10-25 文档审计前的完整说明保存在 `D:\CodexData\RinkSpace_Data\maintenance\agents_audit_20261025\supporting_before.json`，键为本 README 的项目相对路径；原根维护说明另在同目录 `before.json` 的 `MAINTENANCE.md` 键。原路径、SHA 与处置理由见同目录 `supporting_audit.json`。其中旧 S01／六镜头限制、旧 C0 外缘恢复、全局预算 b 与启动顺序只描述历史任务，不能覆盖当前取点定义、范围和数值方案；原 UV、W、人工确认及失败产物本身未改动。
