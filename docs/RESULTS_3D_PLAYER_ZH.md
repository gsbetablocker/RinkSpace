# RinkSpace 三维结果播放器

播放器使用对应的本机 Data、人体模型和显示环境。下表给出仓库启动脚本与 Data 中的已登记启动地址。人体空间位置与接冰的验收状态见[节目结果](../events/ankara_2026/programmes/okada_mei_sp/README.md#人体与足鞋结果)。

## 选择入口与结果

| 用途 | 启动入口 | 版本和范围 |
|---|---|---|
| 冈田芽衣当前全节目人体与鞋刀 | [最新结果登记](../events/ankara_2026/programmes/okada_mei_sp/fit_v11/body_spin_pair_1119.json)的 `launcher.path` | 双击 `START_REFITTED_3D.cmd`，从同一 8,329 个曝光查看或逐帧比较两段旋转的修订 |
| 人体与鞋刀联合拟合对照 | [上一版回放及参数](../events/ankara_2026/programmes/okada_mei_sp/fit_v11/body_joint_1119.json) | 供比较原片回投和三维质心、刀刃轨迹 |
| 全节目首旋修订 | [首旋结果绑定](../events/ankara_2026/programmes/okada_mei_sp/fit_v11/body_first_spin_1113.json)的 `data_root_artifacts.viewer.path` | Data 内 `START_REFITTED_3D.cmd`，覆盖 22.120–188.680 秒。修订窗口为 81.140–93.320 秒 |
| 全节目切镜对照 | [切镜结果绑定](../events/ankara_2026/programmes/okada_mei_sp/fit_v11/body_cut_continuity_1113.json)的 `data_root_artifacts.viewer.path` | Data 内 `START_REFITTED_3D.cmd`，在同一登记相机下检查八处切镜 |
| S01 全身与鞋刀 | [START_3D_COMPARISON.cmd](../launchers/START_3D_COMPARISON.cmd) | `skate_s01_1103`，从原片 22.120 秒开始 |
| S05 源结构窗口 | [START_3D_SEQUENCE_BLADE.cmd](../launchers/START_3D_SEQUENCE_BLADE.cmd) | 无参数选择 `sequence_source_structure`，起点 87.640 秒。窗口覆盖 76.000–95.000 秒 |
| S05 软支撑对照 | [START_3D_SOURCE_FUSION_SUPPORT.cmd](../launchers/START_3D_SOURCE_FUSION_SUPPORT.cmd) | 无参数选择 `support_grounded`，窗口覆盖 76.000–95.000 秒。局部鞋刀与接冰偏差仍在 |
| 历史 S01 自由查看 | [START_3D_RESULTS.cmd](../launchers/START_3D_RESULTS.cmd) | 配置初始为 `j1`、原片 36.020 秒。普通启动恢复此前保存的帧、版本和视角 |

上表的 11 月 13 日全节目候选共享登记的冈田相机与安卡拉冰场。完整 S05 为 74.140–101.640 秒、1,376 个曝光，表中两个 S05 入口是其中 951 个曝光的历史比较。版本切换保留源 PTS。通用播放器接受 `-Variant` 和 `-Fresh`，分别用于选择版本和从配置起点打开。

<a id="historical-player-entries"></a>
### 历史入口

| 结果族 | 入口与可用性 |
|---|---|
| 1106 全节目两版 | [START_3D_PROGRAMME_TRAJECTORY.cmd](../launchers/START_3D_PROGRAMME_TRAJECTORY.cmd) 保留早期双版对照。结果与限制见[1106 绑定](../events/ankara_2026/programmes/okada_mei_sp/programme_trajectory_1106.json) |
| 1107 轨迹和动作 | [轨迹配置](../events/ankara_2026/programmes/okada_mei_sp/viewer3d_programme_trajectory_1107.json)登记七版，本机缺该入口所需输入及部分结果。[动作配置](../events/ankara_2026/programmes/okada_mei_sp/viewer3d_programme_action_1107.json)登记八版，配置默认 `programme_action_physics_1107`，本机缺审阅图层及部分结果 |
| 1108 结构与相机，1109 鞋刀与九镜头复核 | 本机缺配置指定的部分结果或审阅依赖，历史参数与失败证据见[节目索引](../events/ankara_2026/programmes/okada_mei_sp/README.md#历史人体候选导航) |
| 足鞋穿戴、入旋及局部动作 | [足鞋检查](../launchers/START_BOOT_INSPECTION.cmd)、[片段选择](../launchers/START_3D_INTEGRATION.cmd)与[历史候选索引](../events/ankara_2026/programmes/okada_mei_sp/README.md#历史人体候选导航) |


## 怎么操作

| 要做的事 | 操作 |
|---|---|
| 播放／暂停 | 空格或左侧按钮 |
| 前后逐帧 | 逗号／句号，或上一帧／下一帧按钮 |
| 定位曝光 | 拖动帧号滑条，或输入当前片段内帧号 |
| 旋转观察 | 按住鼠标左键拖动；暂停时也可使用 |
| 平移、放大 | 鼠标右键拖动、滚轮 |
| 标准视角 | 侧视 XZ、侧视 YZ、俯视、全冰场按钮 |
| 找回选手 | “自由／回到人体”按钮或 X |
| 看冰面以下或鞋内 | 调整人体／冰面／鞋壳透明度，使用剖面及显示层开关 |
| 放慢动作 | 0.25 倍或 0.5 倍播放 |
| 比较保存版本 | 使用结果下拉框；源 PTS 和视角保持 |

“播放时跟随运动员”只平移观察相机，保持你选择的角度和距离。关闭后可固定观察选手经过；全冰场按钮会关闭跟随。观察相机不改变实验里的冰场相机、模型或保存结果。

源支撑和模型对照入口按整数 PTS 显示原片，自由三维与脚下俯视／侧视保持同帧。fit／review 区域只在实际观测 PTS 显示；同源平面足迹依赖冻结相机和名义冰面，不能作为额外观测或定位真值。

## 画面含义与尺度

人体表面、骨架和鞋刀由所选结果在同一曝光中点的状态展开。滚动快门逐行投影单独处理，不把不同时间的三维采样点拼成一副身体。播放器读取已保存状态，不重新推理或拟合。

橙色和青色区分模型中的两只脚，不能据此确定真实左右脚身份。粉色 P 是骨盆中心，黄色 G 是 P 向冰面的竖直投影；G 与轨迹线不是支撑刀位置或经过验证的质心。鞋刀穿冰、悬空和朝向异常按保存结果显示，未知接触不会被显示端自动修正。

尺度随配置绑定，场景与人体结果共同解释：

| 配置范围 | 条件换算（米／场景单位） | 对应绑定 |
|---|---:|---|
| 普通历史 S01 入口 `viewer3d_s01.json` | 60.00104512261907 | [历史尺度文件](../events/ankara_2026/programmes/okada_mei_sp/body_s01_metric_binding_20261030.json)，对应较早冻结状态 |
| 历史全身、足鞋、动作与旋转配置 | 59.96169706678219 | `Data/runs/body_s01_20261029/fullbody_1102/metric_binding.json`，冻结于早期相机场景 |

尺度是随结果保存的条件换算。当前全节目候选的相机、场景和条件米制由各自[候选绑定](../events/ankara_2026/programmes/okada_mei_sp/README.md#人体与足鞋结果)解释，独立米制仍待验证。

## 文件与环境

普通配置见 [viewer3d_s01.json](../events/ankara_2026/programmes/okada_mei_sp/viewer3d_s01.json)，主比较配置见 [viewer3d_fullbody_1102.json](../events/ankara_2026/programmes/okada_mei_sp/viewer3d_fullbody_1102.json)，其余配置由相应启动入口选择。Data 根通过[共享存储配置](../MAINTENANCE.md#本机工具与存储解析)解析。

独立环境位于 `Data/runtime_tools/results3d`；缺少环境时运行[安装脚本](../environments/install_viewer3d.ps1)，使用已登记的 SMPL-X 模型。显示缓存位于 `Data/cache/results3d`，窗口状态和日志位于 `Data/workbench/results3d`；缺少缓存时只重建显示几何。默认窗口后端为 Pyglet，依赖版本和许可见[播放器许可核查](LICENSING_REVIEW.md#viewer3d)。

从项目根目录选择片段或核对配置，可使用：

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\launchers\START_3D_INTEGRATION.ps1 -Clip jump -Variant A_root -Fresh
powershell -NoProfile -ExecutionPolicy Bypass -File .\launchers\START_3D_INTEGRATION.ps1 -Check
powershell -NoProfile -ExecutionPolicy Bypass -File .\launchers\START_3D_SUPPORT.ps1 -Check
powershell -NoProfile -ExecutionPolicy Bypass -File .\launchers\START_3D_SPIN_MODELS.ps1 -Check
~~~

`-Check` 核对配置和对应文件，不证明人体、轨迹或接触准确。求解与证据生成接口见[实现状态](REBUILD.md#人体与鞋刀拟合执行层)及[复核源码索引](../events/ankara_2026/programmes/okada_mei_sp/body_verification_sources_1103.json)。

## 本地视频

在版本登记中读取 Data 下的回放地址，复制到 PotPlayer。表中时间为对应原片区间，回放自身从零起算。

| 回放 | 原片首末帧 | 地址字段 |
|---|---|---|
| 冈田芽衣当前人体与鞋刀回投 | 22.120–188.680 秒 | [人体结果登记](../events/ankara_2026/programmes/okada_mei_sp/fit_v11/body_spin_pair_1119.json)的 `reprojection.path` |
| 安卡拉四节目当前冰场回放 | 见各节目原片首末帧 | [赛事入口](../events/ankara_2026/README.md#暂定最终版本)中各节目的 `full_review.video.path` |
