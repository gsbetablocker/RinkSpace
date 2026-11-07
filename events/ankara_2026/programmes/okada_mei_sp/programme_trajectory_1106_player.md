# 全节目轨迹开发候选

新旧两版各8329曝光的完整人体缓存已准备。运行：

```text
D:/OneDrive/codex_projs/RinkSpace/launchers/START_3D_PROGRAMME_TRAJECTORY.cmd
```

此入口独立于既有S01、S05及旧默认。九镜头22.12–188.68秒，完整S05为74.14–101.64秒、1376曝光。配置为同目录 viewer3d_programme_trajectory_1106.json，正式参数为 programme_trajectory_1106.npz，结果绑定见 programme_trajectory_1106.json。

新版本 programme_trajectory 使用最终连续人体—鞋刀状态，旧对照 programme_before_trajectory 是本轮v4保存状态。两者各自有同帧人体和完整钢刃回投。可切版本、自由三维、俯视及脚下视角；8次摄影切镜导航按精确源PTS跳转。

紫色人体是模型预测，深粉/浅粉为模型钢刃下/上缘，蓝/橙轴不是源轮廓。源结构描线另有开关，未标注曝光保留unknown。P是骨盆，G是竖直投影，COM是独立质量模型近似，不等同支撑刀或实测质心。

尚未通过准确性验收：101.66秒人体明显偏位，81.62秒自由鞋有偏差，88–92秒仍有残余漂移。新分支也改变S01，但没有替换用户满意S01。证据在Data的 runs/body_s01_20261029/trajectory_first_1106/result_status.json 与 CHANGELOG_TRAJECTORY.md。

两版缓存、PTS、保存P和独立回投已无GUI检查。没有打开播放器、没有连续人工验收，不把缓存存在当作视觉通过。
