# 冈田芽依全节目鞋刀复核

[节目绑定](../../events/ankara_2026/programmes/okada_mei_sp/skate_video_1112/record.json)对应原片 22.120–188.680 秒的九个镜头、8329 个曝光。鞋体和冰刀作为一个刚性组件追踪。轮廓使用源画幅 UV，第二轴为运动员左、右脚，整数 PTS 使用 `1/12800` 时基。没有可用轮廓时 `status` 为零。`extent`、遮挡和身份标志决定轮廓能否进入全鞋目标。三维状态在这些曝光仍可求值。

## 源证据和连续拟合

`RUN.ps1` 重放早期候选。后续整鞋轮廓由 `follow_whole_skates.py`、`relink_whole_skates.py` 和 `repair_skate_visibility.py` 维护。最终轮廓包及人工源锚点在节目目录。`body_skate_fit.py` 在原片相机上共同拟合人体、相对鞋姿和共享鞋刀尺寸。`refine_body_skate_source.py` 使用少量重新审阅的原片外轮廓。`foot_attachment.py` 比较开放鞋口与旧脚鞋绑定，`foot_socket.py` 在保留刚性组件世界姿态的情况下校准脚在鞋内的装配。

`deliver_body_skate_fit.py` 从连续控制导出同一人体状态的全节目、质心、冰面投影和鞋刀，再生成原片／旧候选／新候选对照。节目绑定给出 Data 下的 `START_REFITTED_3D.cmd`、对照视频、结果和失败导航。两份公开控制包和整鞋轮廓保持实际运行的原字节。

## 1113 开场与方向实验

[1113 候选](../../events/ankara_2026/programmes/okada_mei_sp/skate_video_1112/audit_followup_1113.json)登记同一人体、整鞋、质心与冰面投影的控制及回投。`programme_audit_repair` 负责由 1112 回放导入共同状态。`motion_review.source_audit` 以原生曝光计算双鞋的源投影和运动差分，`motion_review.scoped_fit`、`direction_fit`、`cut_position_fit` 和 `spin_entry_fit` 使用局部源锚点，`motion_review.refit_audit` 核对修订后的连续曲线。`programme_motion_followup` 从同一组已选控制生成回放和原片对照。全节目 ExMR 比较属于历史试验。

使用 Data 中的 prepared、evidence 和节目绑定指明的控制，从项目根在 `body_cuda` 环境用 `python -B -m experiments.skate_video_v1.<模块> --help` 查看各命令。开场入口还需要上一镜站姿参数和原片数值锚点，方向入口使用各段源锚点及父控制。当前核对使用开发证据，核对范围和失败事件均保存在绑定中。

`htd_review.prepare_pva` 与 `infer_pva` 使用外部作者实现和权重。`audit_fs_pva` 计算独立多机位参照误差，`audit_programme_pva` 与 `native_ablation` 对比节目动力学，`entry_pose_ablation --flight --variant-grid --kinematic` 对比 S05 入旋肩肘。官方权重和 PVA 输入输出保存在 Data。入旋二维误差有局部改善，肘角越界增多，速度场也低估绝对滑行速度。官方来源与版本见[资产记录](../../docs/THIRD_PARTY_ASSETS.md)。

## 原片和原生重放核对

需要登记的原片、相机、恢复绑定、人体观察、已安装的本地模型和原生运行库。设置 `RINKSPACE_SKATE_RECOVERY_BINDING` 指向本机恢复绑定，并在项目根用 `vision_cuda` 环境执行。`$visual` 指包含 `whole_follow_evidence.npz` 的目录，可使用公开节目绑定中的数值包。`$controls` 指 `whole_skate_socket_1112.npz`。收据路径在 Data 下。

```powershell
python -B -m experiments.skate_video_v1.verify_review clock $visual $controls $clockReceipt
python -B -m experiments.skate_video_v1.verify_review recovery $binding $previousProZip $recoveryReceipt
```

在 `results3d` 环境执行 `python -B -m experiments.skate_video_v1.verify_review viewer $viewerConfig $viewerReceipt`。该检查启动实际 3D 播放器，核对全序列质心及冰面投影和两版切换。`recovery` 比对旧离线归档和本机恢复结果的全序列数组及原人体观测身份。

## 审阅限制

`whole_skate_socket_1112` 的 [1112 复核](../../events/ankara_2026/programmes/okada_mei_sp/skate_video_1112/record.json)记录了 845 条留出整鞋对照，轮廓中位误差由 8.32 降至 7.11 px。同期人体关键点中位误差由约 4.62 升至 4.82 px，最大鞋壁冲突中位值由 7.04 升至 11.47 mm。指标仅适用于这个绑定。

[1113 开场与方向绑定](../../events/ankara_2026/programmes/okada_mei_sp/skate_video_1112/audit_followup_1113.json)保存该实验自己的指标与失败位置。使用 11 月 13 日登记相机和共同冰场的[切镜与首旋候选](../../events/ankara_2026/programmes/okada_mei_sp/README.md#人体与足鞋结果)在节目页登记，仍待独立三维与接冰验收。
