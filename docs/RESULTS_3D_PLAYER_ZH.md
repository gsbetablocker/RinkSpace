# RinkSpace 三维结果播放器

双击项目根目录的 [START_3D_RESULTS.cmd](../START_3D_RESULTS.cmd)。无需输入命令；打开的是独立三维窗口。

第一次默认停在原片 **36.020 秒**，以后记住退出时的帧、结果版本和观察角度。窗口左边的下拉框可切换已保存的 S01 人体结果，切换时保持帧号和视角。

## 2026-11-03 全S01连续候选：当前人工审核入口

[START_3D_COMPARISON.cmd](../START_3D_COMPARISON.cmd)以 `Fresh` 进入 `skate_s01_1103`，从原片 **22.12秒、side全身视角**开始，按50fps连续播放至44.96秒。整段1143曝光共用一条全69自由度的连续修正曲线，没有内部冻结窗口。`full1102`、`hand1103`及已封存的`skate1103`可同PTS比较。普通入口及用户会话保留；本轮后台检查不打开或控制桌面窗口。

[本轮候选绑定](../events/ankara_2026/programmes/okada_mei_sp/skate_s01_1103.json)、[紧凑参数](../events/ankara_2026/programmes/okada_mei_sp/skate_s01_1103.npz)保存完整旧base和新的整段修正。实际计算于907.009秒预算出口停止，3个接受步，未加罚；随后只导出、复核并生成缓存。源50Hz最大非穿透违例89.549→7.705mm，鞋口单截面超出40.504→3.996mm，但26.22秒支持候选仍高于冰面32.113mm，43.86秒12.077mm。刀线P95 26.571→35.542px、开场静稳关节变化4.371°→12.460°也保留为失败。开场语义因子只继承了base状态，未在本次继续约束；零姿态成本初值下相对10%出口不适用。**候选没有通过PRO物理及开场保持验收。**

`Data/runs/body_s01_20261029/skate_s01_1103/physical_overview.png`总览整段非穿透、明确支持曝光、generic foot内腔和鞋口截面；`postfit_audit.json`保留姿态回归。无窗口播放器388项检查通过，并逐一播放全部1143曝光，PTS无跳过或重复；这些只验证结果接线与回放。物理S01的20.02–22.12秒不在当前身体输入中，左右脚／刃身份和未审支撑曝光仍unknown。支持3mm上界及1mm余量仍未标定。

## 较早鞋刀候选（保留对照）

在同一入口选择 `skate1103`，可回看原片35.20秒的旧候选。左侧“同侧视角”“足鞋正视”和“鞋壳剖面”可检查足鞋；剖面可隐藏朝向观察相机的一侧。切换维持源PTS与观察角度。

[候选绑定](../events/ankara_2026/programmes/okada_mei_sp/skate_contact_1103.json)、[紧凑参数](../events/ankara_2026/programmes/okada_mei_sp/skate_contact_1103.npz)连接已封存的第2轮外迭代、累计24个接受步；源检查点SHA256为 `08a414b1a2ca9874ca5c964235b3c74f1f4c77274b8c58228cb9966b250a6e4b`。旧求解的身体窗口为28.12–43.80秒，新共享鞋刀回放全部1143个源曝光。输入22.12–44.96秒不包含物理镜头20.02秒起的完整范围。

此候选**已转人工审核，不再优化，未通过准确性验收**。核心2.5ms密检最大非穿透违例1.188411mm，全修正窗口最大1.658310mm；35.20秒支持候选高度4.142277mm，核心generic foot内腔最大越界0.894770mm。全S01源曝光最大非穿透约束违例89.548823mm（25.04秒），鞋口单截面最大超出40.504mm（25.54秒）；单截面不能认证三维穿戴。26.22／39.78／43.86秒支持候选高度为66.609／−2.165／37.076mm，物理左右脚仍未知。核心同源刀线中位数3.188→9.130px、P95 7.892→12.994px的退化已触发审核出口。3mm和1mm仍是未标定调试量。

`Data/runs/body_s01_20261029/skate_contact_1103/`中的`validation.json`、`fit/candidate.json`、`parameter_replay.json`及`collar_gap.json`分别保存同源比较、停止候选、参数回放与鞋口限制。播放器实际side检查240项、81曝光50fps连续播放、内腔检查89项和正视检查通过；相关80项测试与2个子测试通过，均不替代物理或原片审核。

整段曲线的实际运行和限制见上节。[预算策略](../events/ankara_2026/programmes/okada_mei_sp/skate_run_policy_1103.json)为900秒／8个接受步／1轮外迭代，禁止自动加罚；它不追溯应用于这份累计24个接受步的旧候选。相对质量出口只有在初值为正时可比较。

## 2026-11-03 开头时序修复

在[同一比较播放器](../START_3D_COMPARISON.cmd)切换 `hand1103`，可定位原片22.120秒检查“开场扶腰、22.90秒后下探与展开”。用户已明确纠正：基本到22.90秒只有手部动作，然后才下探。前一动态候选的下探过早和开头约一秒1.11米平移错误，已标作失败对照保留。新的静稳条件只覆盖22.12–22.90秒的躯干和腿，手臂和手腕继续动态；22.90秒后开放下探、起身和转动。最终右臂局部精修保持这份身体时序逐数组不变，手指仍为模型默认形状。

[时序参数与证据](../events/ankara_2026/programmes/okada_mei_sp/opening_timing_1103.json)连接完整结果；`Data/runs/body_s01_20261029/opening_timing_1103/timing_report.json`单独报告22.12–22.90秒的平移、高度和角度变化，并与两份失败候选使用相同源PTS比较。源图与三维检查图只辅助定位错误，屏外下肢、深度和接触仍按证据范围解释。

[最终右臂与时序绑定](../events/ankara_2026/programmes/okada_mei_sp/opening_hand_1103.json)保存 `hand1103`，运行产物在上述时序目录的`hand_polish/`。静稳段XY跨度6.07cm、高度下降1.26cm，下探最低点23.26秒；残留腕点偏差、穿冰和离冰见节目记录。该版本会同时加载时序紧凑参数和右臂紧凑参数。

复用已保存的时序拟合，导出、审阅并更新同一播放器：

```powershell
& 'D:\OneDrive\codex_projs\RinkSpace\.venv\Scripts\python.exe' -I -B -X utf8 'D:\OneDrive\codex_projs\RinkSpace\scripts\run_opening_timing.py' --root 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029' --output 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029\opening_timing_1103' --stage export
& 'D:\OneDrive\codex_projs\RinkSpace\.venv\Scripts\python.exe' -I -B -X utf8 'D:\OneDrive\codex_projs\RinkSpace\scripts\review_opening_timing.py' --root 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029' --output 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029\opening_timing_1103'
& 'D:\OneDrive\codex_projs\RinkSpace\.venv\Scripts\python.exe' -I -B -X utf8 'D:\OneDrive\codex_projs\RinkSpace\scripts\deliver_opening_timing.py' --root 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029' --output 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029\opening_timing_1103'
& 'D:\OneDrive\codex_projs\RinkSpace\.venv\Scripts\python.exe' -I -B -X utf8 'D:\OneDrive\codex_projs\RinkSpace\scripts\run_opening_hand.py' --root 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029' --output 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029\opening_timing_1103\hand_polish' --stage export
& 'D:\OneDrive\codex_projs\RinkSpace\.venv\Scripts\python.exe' -I -B -X utf8 'D:\OneDrive\codex_projs\RinkSpace\scripts\review_opening_timing.py' --root 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029' --output 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029\opening_timing_1103\hand_polish'
& 'D:\OneDrive\codex_projs\RinkSpace\.venv\Scripts\python.exe' -I -B -X utf8 'D:\OneDrive\codex_projs\RinkSpace\scripts\deliver_opening_hand.py' --root 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029' --output 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029\opening_timing_1103\hand_polish'
```

主日志是该目录`fit_progress.jsonl`。静态源图入口仍用下节的`review_opening_motion_visuals.py`，增加`--source-dir <opening_motion_1103目录>`复用既有源图，避免复制或重新解码素材。

## 2026-11-03 动态修复失败历史

该历史结果现标为“失败候选：下探过早与开头1.11米平移”，与“旧版：固定姿态与近静止条件（失败对照）”均可在同一播放器切换。历史窗口22.12–25.00秒，共145个曝光；骨盆实际平移和竖直速度自由，右手扶腰关系在23.10–23.40秒释放，25秒处连续接回足鞋父版。

结果、完整源图对照与剩余失败见[本轮绑定](../events/ankara_2026/programmes/okada_mei_sp/opening_motion_1103.json)。Data中的 `motion_report.json`、`hand_waist_report.json`、`source_motion_comparison.png` 和 `motion_curves.png` 分别保留同一观测分母、手腰模型距离、源图投影和逐曝光动作变化。屏外腿部及手指仍有推断；显示程序不宣告接触或准确性通过。

复用已有拟合结果导出，再核验OneDrive参数并更新同一播放器缓存：

```powershell
& 'D:\OneDrive\codex_projs\RinkSpace\.venv\Scripts\python.exe' -I -B -X utf8 'D:\OneDrive\codex_projs\RinkSpace\scripts\run_opening_motion.py' --root 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029' --output 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029\opening_motion_1103' --stage export
& 'D:\OneDrive\codex_projs\RinkSpace\.venv\Scripts\python.exe' -I -B -X utf8 'D:\OneDrive\codex_projs\RinkSpace\scripts\deliver_opening_motion.py' --root 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029' --output 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029\opening_motion_1103'
```

`--stage run`用于缺失拟合时的实际求解；主日志是该目录 `fit_progress.jsonl`。静态图入口 `scripts/review_opening_motion_visuals.py --root <同上> --output <同上>`使用已有源图片和保存结果，在现有 `Data/runtime_tools/vision_cuda` 绘图环境运行。它不解码或生成视频。

## 2026-11-02 开头裁切修复历史

该历史开头结果现标为“旧版：固定姿态与近静止条件（失败对照）”，可在同一播放器切换并保持源PTS与观察角度。[START_3D_COMPARISON.cmd](../START_3D_COMPARISON.cmd)当前以 `Fresh` 从22.12秒进入全S01候选；开头时序和右臂版本保留，普通播放器入口仍可恢复原会话。

修复覆盖 **22.12–25.00 秒、145 个曝光**；前 **22.12–24.44 秒、117 个曝光**使用近原地转身条件，末段连续衔接父版。原片 **19.960 秒**准备画面作为辅助曝光：复用 `secondary_long_side` 的冻结安装和完整光学模型，只求这一帧的 pan、tilt、log(f)，把实际 native 几何投回源图。MEIWA 蓝绿条参与该帧相机拟合，两条冰面线另行检查；B2 与 F2–R2 的交点和可见交叉腿姿势用于开头初始化及软约束。身体、下鞋和刀具继续由同一状态求值，原始屏外下肢观测没有补造。

参数和证据见 [opening_repair_1102.json](../events/ankara_2026/programmes/okada_mei_sp/opening_repair_1102.json)。自动比较保存在 `Data/runs/body_s01_20261029/opening_repair_1102/results.txt`，包含平移跨度、躯干倾斜、刀刃高度、共同源像素误差与衔接检查；`preparation_geometry_overlay.png` 是准备画面的实际几何叠图，`source_opening.png` 是三个源时刻的原图／修复前／修复后对照。准备画面的物理左右脚仍有两种候选，屏外腿部仍标为推断。

复用已有结果导出并更新缓存（不会重新推理或重新拟合）：

```powershell
& 'D:\OneDrive\codex_projs\RinkSpace\.venv\Scripts\python.exe' -B -X utf8 'D:\OneDrive\codex_projs\RinkSpace\scripts\run_opening_repair.py' --root 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029' --output 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029\opening_repair_1102' --stage export
```

主日志：`Data/runs/body_s01_20261029/opening_repair_1102/fit_progress.jsonl`。两次早期候选保留在该目录的 `history/`，未覆盖原全身或足鞋结果。

开头和足鞋的独立证据生成代码已保存到仓库，可对现有结果执行，不启动拟合：

| 入口（从源码根运行 `python -B -m` 加下列模块名） | 参数与用途 |
|---|---|
| `calibration.program_tracking.body_tracking.boot_invariants` | `--root <人体运行根> --output <足鞋目录> --report <新JSON路径> --final`；检查共享状态、刚体关系和尺度不变量 |
| `calibration.program_tracking.body_tracking.opening_verification` | `--root <人体运行根> --output <开头目录> --report <新JSON路径>`；从 OneDrive 紧凑参数重放全部1143帧，对比保存的身体和鞋刀 |
| `calibration.program_tracking.body_tracking.opening_metrics` | `--root <人体运行根> --output <开头目录> --report-dir <新的Data报告目录>`；按117个原生PTS生成平移、躯干角和最低刀刃高度曲线 |

前两项使用完成该结果时的项目 `.venv`（NumPy 2.4.6／SciPy 1.17.1），独立绘图命令使用已含 Matplotlib 的 `Data/runtime_tools/vision_cuda` 环境。图和报告写入指定 Data 位置，既有结果不改写。三模块保留原 Data 脚本的来源说明；固定原路径与哈希见[代码保全索引](../events/ankara_2026/programmes/okada_mei_sp/body_verification_sources_1103.json)。

两份比较配置中的 `metric_binding_base="data"` 使尺度文件跟随本机 Data 根；普通旧配置默认按项目根解析，绝对路径仍可用于历史记录。

## 2026-11-02 足鞋穿戴检查

双击 [START_BOOT_INSPECTION.cmd](../START_BOOT_INSPECTION.cmd)，进入同一个播放器的足部近看与透明鞋模式，停在原片 **36.020 秒**。可切换以下三项：

| 名称 | 具体内容 |
|---|---|
| 对照：旧封闭刚性鞋壳 | 上一轮全身结果和原有凸包封闭鞋壳 |
| 显示修复：22 面开放鞋口，顶点不变 | 与上一项完全相同的身体、鞋顶点和刀具，只更换表面连接，去掉鞋口上盖 |
| 候选：共享穿合与有限上帮形变 | 全 S01 共用足鞋配准；加入足床、近似内腔、跟杯／前掌锚点，以及小腿间隙和接触驱动的鞋口／鞋舌形变 |

本轮人体姿态新增优化区间是 **35.20–36.80 秒、40.00–41.50 秒，共 157 帧**；共享鞋配准应用于 **22.12–44.96 秒的 1,143 帧**。其余身体动作继承上一轮全身候选。下鞋仍挂在足后段 7/8；刀具刚性安装在下鞋上，鞋口形变不会改变刀刃形状。足部体型、骨长未缩放，未加入逐帧滑鞋参数。

“穿戴检查”保留完整人体表面，默认显示透明鞋壳与标色的鞋口边界。“剖切侧面”可去掉鞋壳一侧，也可自动去掉朝向观察相机的一侧。足床／锚点、内腔和坐标轴各有独立开关，默认关闭辅助线以免遮住足部。界面分别显示小腿相对足段旋转、刀具世界倾角、上部形变量、内腔超出量和锚点距离。这些数值与自动报告共用保存的状态，不靠观察相机角度判断。

实际前后数值在 `Data/runs/body_s01_20261029/boot_integration_1102/results.txt`，来源、参数与播放器绑定见 [boot_integration_1102.json](../events/ankara_2026/programmes/okada_mei_sp/boot_integration_1102.json)。报告分别统计显示拓扑修复和几何拟合，并列出鞋部源像素变差的帧。内腔和足床是明确保存的几何先验；现有数据没有已识别刀齿观测，因此不会把刀刃端点标成点冰刀齿。

以下一条命令复用匹配的拟合结果，完成导出、报告和三项播放器缓存；缺失的拟合阶段才会运行，不启动窗口：

```powershell
& 'D:\OneDrive\codex_projs\RinkSpace\.venv\Scripts\python.exe' -B -X utf8 'D:\OneDrive\codex_projs\RinkSpace\scripts\run_boot_integration.py' --root 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029' --output 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029\boot_integration_1102' --stage run
```

主日志：`Data/runs/body_s01_20261029/boot_integration_1102/fit_progress.jsonl`。`inputs.json` 核对父版动作、体型、相机、观测与配准先验；不匹配时保留已有结果并报告差异。

## 2026-11-02 全身短段对照

双击 [START_3D_COMPARISON.cmd](../START_3D_COMPARISON.cmd)，使用同一个播放器加载[本轮绑定](../events/ankara_2026/programmes/okada_mei_sp/viewer3d_fullbody_1102.json)；当前默认从22.12秒进入全S01连续候选，可切回全身短段并定位36.020秒。下拉框保留原 J1、修订观测后的旧根／腿自由度、全身与根朝向候选，以及历史 B5／J2 参考；切换保持源 PTS 和视角。

本轮三组共同冻结 `camera_solver_refined_1102` 的 native 冰场与相机，条件尺度为 59.96169706678219 米／场景单位。人体表面、骨架和鞋刀共用唯一全身状态；曝光中点几何与滚动快门逐行投影分别保存。场景尺度换算不是拟合改善。

实际优化只覆盖原片 24.50–27.00、32.20–33.80、35.20–36.80、37.40–39.00、40.00–41.50 秒，共 445 帧；S01 其余 698 帧仍是原 J1。界面标明短段内外，完整节目的人体跟踪尚未完成。接冰、真实左右刀身份与独立三维精度仍未知，鞋刀悬空和穿冰照实显示。

参数和证据索引见 [body_fullbody_1102.json](../events/ankara_2026/programmes/okada_mei_sp/body_fullbody_1102.json)。运行产物位于 `Data/runs/body_s01_20261029/fullbody_1102/`：`results.txt`、`metrics.json`、源图对照与 `player_check/checks.json`。原始观测、9 点局部修正、29 点待确认排除及修正前的失败均保留；本轮全部材料用于开发，不是独立验收集。

当前判定是**工程可运行，三维准确性尚不能验收，模型仍有穿冰**。报告开头提供全部445帧的自动穿冰统计和两份拟合的三维位置差异；播放、旋转和静态画面仅帮助查找异常，不承担证明深度或真实接触正确的作用。

在 PowerShell 中一条命令核对输入签名、完成缺失窗口、导出、生成报告与播放器缓存（已有匹配结果直接复用；不运行神经网络、不打开窗口）：

```powershell
& 'D:\OneDrive\codex_projs\RinkSpace\.venv\Scripts\python.exe' -B -X utf8 'D:\OneDrive\codex_projs\RinkSpace\scripts\run_fullbody_round.py' run --root 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029' --output 'D:\CodexData\RinkSpace_Data\runs\body_s01_20261029\fullbody_1102'
```

主日志是输出目录的 `run_progress.jsonl`，求解明细是 `fit_progress.jsonl`。旧输入签名不符时保留结果并报告差异，不覆盖或把旧窗口误认作新优化。

## 怎么操作

| 要做的事 | 操作 |
|---|---|
| 播放 / 暂停 | 空格，或左侧按钮 |
| 前后逐帧 | 逗号 / 句号，或上一帧 / 下一帧按钮 |
| 找到一帧 | 拖动帧号滑条，或输入 S01 内的帧号 |
| 绕到人体侧面、背面 | 在画面中按住左键拖动；暂停时也可以 |
| 平移、放大 | 右键拖动、滚轮 |
| 标准视角 | 侧视 XZ、侧视 YZ、俯视、全冰场按钮 |
| 找回运动员 | “自由 / 回到人体”按钮，或 X |
| 看骨架、冰面以下的部分 | 调低人体 / 冰面透明度，或关闭相应显示层 |
| 放慢看动作 | 0.25 倍或 0.5 倍播放 |
| 看旧结果的差别 | 切换结果下拉框；帧与视角保持不变 |

“播放时跟随运动员”只平移观察相机，保持你选的角度和距离。关闭后可以固定在某个位置看运动员经过。全冰场按钮会关闭跟随。

## 画面里的东西代表什么

- **人体表面与绿色骨架**：用所选结果保存的体型、姿态和局部修正展开成完整 SMPL-X 表面；没有重新跑人体识别或拟合。
- **橙色和青色鞋刀**：原结果中的两只鞋刀候选。颜色区分模型中的两只脚，不能据此认定画面里的真实左右脚身份已经确认。
- **粉色 P**：骨盆中心。**黄色 G** 是 P 向冰面作竖直投影的位置；黄色整段曲线连接每个曝光时刻的 G。它不是支撑刀的位置，也不是经过验证的质心。
- **鞋刀穿冰、悬空、朝向不对**：原样保留供检查。播放器不会把鞋刀自动按到冰面上，接冰状态仍是未知。
- **旧结果缺鞋壳**：旧 A / B / C / B5 只保存过简化刀刃代理，播放器不会给它们补上新鞋壳，界面会提示。

所有人体、骨架和鞋刀在同一个原片帧的曝光中点求值。原求解里为处理滚动快门保留的逐行采样点，不会被拼成一副不同时间的三维身体。

## 打开的究竟是哪份结果

普通 `START_3D_RESULTS.cmd` 的原配置默认结果为 **J1_s2：根位置与腿姿态联合**，另有 J2_s2（同时调整鞋尺寸）、J0_s2_footinit（根固定只修腿）、旧 B5、旧 B、旧 C、旧 A。比较入口当前默认是 `skate_s01_1103`，并保留 `skate1103`、`hand1103`。准确性均未验收；播放器提供可检查的三维解释，不提高结果本身的精度。

冰场读取所选人体结果配套的冻结 review_scene。其他场景或拟合状态的更新不会自动改变这份结果；切换到新结果需要在节目绑定中登记对应的参数、场景和源时间。

冰场、人体和鞋刀采用绑定中的统一条件尺度换算：当前为 **60.00104512261907 米 / 场景单位**。旧人体文件的坐标按同一比例整体换算后显示，原文件不改写。原实验按近似身高选定的人体比例仍沿用，不能当成实测体型。

## 文件和环境

- 节目绑定：[viewer3d_s01.json](../events/ankara_2026/programmes/okada_mei_sp/viewer3d_s01.json)。
- 源码：src/rinkspace/viewer3d/；根目录的 cmd / ps1 是启动入口。
- 独立运行环境：Data/runtime_tools/results3d，CPU Torch，不改现有推理或求解环境。
- 完整人体显示缓存：Data/cache/results3d/s01。按输入文件指纹区分，原结果不变；首次缺缓存时只重建显示几何。
- 窗口状态、日志、操作检查及三维检查图：Data/workbench/results3d。
- 重建独立环境：运行[安装脚本](../environments/install_viewer3d.ps1)；使用已存在的 SMPL-X 模型，不另下载人体权重。

默认窗口后端为 Pyglet。安装版本、许可原文与 Qt 依赖边界见[播放器许可核查](LICENSING_REVIEW.md#viewer3d)；显示缓存和 SMPL-X 模型不随公开源码分发。

基础三维交互由 [AITViewer](https://github.com/eth-ait/aitviewer) 提供，本项目补上源时间、统一单位、结果版本、冻结场景和鞋刀显示。窗口的观察相机仅控制你的视角，不会改动实验里的冰场相机。
