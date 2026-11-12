# 冈田芽衣全节目鞋刀复核

[节目绑定](../../events/ankara_2026/programmes/okada_mei_sp/skate_video_1112/record.json)对应原片 22.120–188.680 秒的九个镜头、8329 个曝光。鞋体和冰刀作为一个刚性组件追踪。轮廓使用源画幅 UV，第二轴为运动员左、右脚，整数 PTS 使用 `1/12800` 时基。没有可用轮廓时 `status` 为零。`extent`、遮挡和身份标志决定轮廓能否进入全鞋目标。三维状态在这些曝光仍可求值。

## 源证据和连续拟合

`RUN.ps1` 重放早期候选。后续整鞋轮廓由 `follow_whole_skates.py`、`relink_whole_skates.py` 和 `repair_skate_visibility.py` 维护。最终轮廓包及人工源锚点在节目目录。`body_skate_fit.py` 在原片相机上共同拟合人体、相对鞋姿和共享鞋刀尺寸。`refine_body_skate_source.py` 使用少量重新审阅的原片外轮廓。`foot_attachment.py` 比较开放鞋口与旧脚鞋绑定，`foot_socket.py` 在保留刚性组件世界姿态的情况下校准脚在鞋内的装配。

`deliver_body_skate_fit.py` 从连续控制导出同一人体状态的全节目、质心、冰面投影和鞋刀，再生成原片／旧候选／新候选对照。节目绑定给出 Data 下的 `START_REFITTED_3D.cmd`、对照视频、结果和失败导航。两份公开控制包和整鞋轮廓保持实际运行的原字节。

## 原片和原生重放核对

需要登记的原片、相机、恢复绑定、人体观察、已安装的本地模型和原生运行库。设置 `RINKSPACE_SKATE_RECOVERY_BINDING` 指向本机恢复绑定，并在项目根用 `vision_cuda` 环境执行。`$visual` 指包含 `whole_follow_evidence.npz` 的目录，可使用公开节目绑定中的数值包。`$controls` 指 `whole_skate_socket_1112.npz`。收据路径在 Data 下。

```powershell
python -B -m experiments.skate_video_v1.verify_review clock $visual $controls $clockReceipt
python -B -m experiments.skate_video_v1.verify_review recovery $binding $previousProZip $recoveryReceipt
```

在 `results3d` 环境执行 `python -B -m experiments.skate_video_v1.verify_review viewer $viewerConfig $viewerReceipt`。该检查启动实际 3D 播放器，核对全序列质心及冰面投影和两版切换。`recovery` 比对旧离线归档和本机恢复结果的全序列数组及原人体观测身份。

## 审阅限制

最新候选在 845 条留出整鞋对照中将轮廓中位误差从 8.32 px 降到 7.11 px。39 条历史冰刃描线的中位误差从 11.81 px 降到 6.70 px。人体关键点中位误差从 4.61 px 升到 4.82 px，开放鞋口模型的最大鞋壁冲突中位值从 7.04 mm 升到 11.47 mm。原片跟踪仍有 1331 个单脚曝光未跟住、4009 个单脚曝光范围待确认，另有 19 个可见单脚曝光身份未解决。90.120 秒左鞋仍有明显偏差。像素对照、条件尺度和实机回放均不能单独验证世界位姿或接冰准确性。
