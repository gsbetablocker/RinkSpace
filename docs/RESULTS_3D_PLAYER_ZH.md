# RinkSpace 三维结果播放器

双击 [launchers/](../launchers/) 中的相应启动文件，打开独立三维窗口。结果需要对应的本机 Data、模型和显示环境；公开仓库不附比赛视频、完整人体模型或显示缓存。当前人体结果均未通过独立三维准确性或接冰验收，候选状态见[节目入口](../events/ankara_2026/programmes/okada_mei_sp/README.md#人体与足鞋结果)。

## 选择入口与结果

| 入口 | 默认结果与起点 | 适用范围 |
|---|---|---|
| [START_3D_RESULTS.cmd](../launchers/START_3D_RESULTS.cmd) | 原配置默认 `j1`（J1_s2），首次停在原片 36.020 秒；后续恢复保存的帧、版本和视角 | 历史 S01 人体与鞋刀参考 |
| [START_3D_COMPARISON.cmd](../launchers/START_3D_COMPARISON.cmd) | `skate_s01_1103`，Fresh 从原片 22.120 秒、side 全身视角开始 | 处理 S01：22.120–44.960 秒，1143 个曝光；保留较早全身、开头和鞋刀候选 |
| [START_3D_SEQUENCE_BLADE.cmd](../launchers/START_3D_SEQUENCE_BLADE.cmd) | `sequence_source_structure`，无参数启动使用 Fresh，原片 87.640 秒 | 当前 S05 完整源结构与条件支撑位置候选，76.000–95.000 秒；保留旧 14 步、39 步及原 A 对照 |
| [START_BOOT_INSPECTION.cmd](../launchers/START_BOOT_INSPECTION.cmd) | `opening1102`，Fresh 停在原片 36.020 秒 | 四项足鞋／开头历史对照；默认项是旧开头候选 |
| [START_3D_INTEGRATION.cmd](../launchers/START_3D_INTEGRATION.cmd) | 先选择片段，回车选 S01；S01 默认 `skate_s01_1103`，其他三段默认 `A_root` | 跳跃 44.980–52.000 秒、步法 25.000–35.000 秒、旋转 76.000–95.000 秒 |
| [START_3D_SUPPORT.cmd](../launchers/START_3D_SUPPORT.cmd) | `support_new`，默认原片 88.000 秒 | 87.500–92.500 秒上下文，核心为 88.000–92.000 秒 |
| [START_3D_SPIN_MODELS.cmd](../launchers/START_3D_SPIN_MODELS.cmd) | `exmr_prefoot`，默认原片 88.120 秒 | 76.000–95.000 秒 GVHMR／ExMR 对照，951 个曝光 |
| [START_3D_SPIN_REFINED.cmd](../launchers/START_3D_SPIN_REFINED.cmd) | `local_281`，首次原片 81.580 秒 | 历史局部关联修复；整段续算 `continuous_refined` 在同一配置中保留 |
| [START_3D_SPIN_CONTINUOUS.cmd](../launchers/START_3D_SPIN_CONTINUOUS.cmd) | `continuous_source`，首次原片 81.300 秒 | 历史绝对连续 A/B，76.000–95.000 秒 |
| [START_3D_SPIN_ENTRY.cmd](../launchers/START_3D_SPIN_ENTRY.cmd) | `entry_motion`，首次原片 81.300 秒 | 历史入旋／质心重力候选，76.000–95.000 秒 |

版本切换保持原片 PTS 和观察角度。表中列出配置初始值；Fresh 不恢复上次会话。S01 比较、足鞋检查和无参数的 S05 源结构入口按配置重新定位；普通入口可恢复保存的帧、结果和视角。使用 -Variant 选择保留版本，使用 -Fresh 从配置起点检查。

足鞋入口的四项为“旧封闭刚性鞋壳”“相同顶点的开放鞋口”“共享穿合与有限上帮形变”及 `opening1102`。最后一项保留固定姿态／近静止条件的失败历史。鞋壳显示连接改变和实际拟合改变分别解释，未观察到的刀齿或接点不会补造。

动作入口的三段保留 `A`、`B`、`A_root`、`B_root`；旋转的 `B_root` 复用与 B 数值等价的 A 根修正。源支撑入口比较原 A、旧 root、首轮保留状态和最终候选。模型入口比较固定 GVHMR 初值、GVHMR 一步候选、ExMR 足细化前／后初值；足细化后存在结构 unknown，没有合格最终拟合。具体失败与结果身份由[节目结果导航](../events/ankara_2026/programmes/okada_mei_sp/README.md#人体与足鞋结果)定位。

S05 源结构入口可逐帧查看完整源轮廓，用“上一／下一源标注”定位实际描线曝光；未标注帧不继承邻帧描线。配置中保留的旧结果按版本单独解释，源像素范围与条件冰面位置均不代表真实接触已验收。

本轮完整76–95秒、951曝光的新旧比较使用 [START_3D_SOURCE_FUSION_SUPPORT.cmd](../launchers/START_3D_SOURCE_FUSION_SUPPORT.cmd)，显式选择 `support_grounded`，保留旧版本并预备完整缓存。此候选仅有有限改善，91.20秒刀具俯仰及局部鞋刀退化仍未解决，界面标记未通过视觉验收；[结果身份](../events/ankara_2026/programmes/okada_mei_sp/spin_support_correction_1106_audit.json)不将工程加载通过解释为画面准确。

全节目开发对照使用 [START_3D_PROGRAMME_TRAJECTORY_1107.cmd](../launchers/START_3D_PROGRAMME_TRAJECTORY_1107.cmd)：22.120–188.680 秒、8329 原生曝光，新旧和四个尺度／深度诊断及足部关联失败共 7 版，均已有缓存。镜头切换导航使用相邻真实 PTS；人体形状和尺度按每个候选自己的保存值显示。紫色骨架和粉色钢刃是模型回投，源轮廓与条件支撑另行标识，局部鞋底不等同完整钢刃或接冰真值。该入口不改变旧默认；[结果和限制](../events/ankara_2026/programmes/okada_mei_sp/README.md#whole-programme-body)明确保留未通过世界轨迹与接冰验收的状态。

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
| 全身／足鞋／动作／旋转对照配置 | 59.96169706678219 | Data 中 `runs/body_s01_20261029/fullbody_1102/metric_binding.json`，对应 `camera_solver_refined_1102` |

这些是统一单位换算，不是实测场地尺度、体型或三维精度。其他场景更新不会自动改变已冻结结果；具体相机、场景、参数和源时间以所选配置为准。

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

| 视频（Data 根内相对路径，供 PotPlayer） | 对应原片时间 |
|---|---|
| `runs/body_s01_20261029/support_spin_1104/source_A_root_new_88_92.mp4` | 88.000–92.000 秒 |
| `runs/body_s01_20261029/exmr_spin_compare_1104/native_protocol/source_GVHMR_ExMR_76_95.mp4` | 76.000–95.000 秒 |

视频、源图、完整拟合与逐轮诊断保留在 Data，由[节目结果绑定](../events/ankara_2026/programmes/okada_mei_sp/README.md#人体与足鞋结果)定位；界面操作和播放检查不能代替这些结果的准确性验收。
