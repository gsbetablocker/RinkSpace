# RinkSpace 求解器重构设计：唯一数值核心与 GPU 执行路径

版本：2026-10-24，针对公开仓库 `3e866a835dac26c461bb4ae436ae289814a1ae57`。本文规定待实现的完整数值架构；本轮首切片只实现统一类型、部分场景几何和投影前向计算，不能把这些接口说明当成已完成 GPU 求解器。

## 1. 决策与上一版必须纠正的地方

生产求解器只保留一条路径：**C++ 控制的阻尼 Gauss–Newton / LM；CUDA 计算同一几何、相机、残差、导数和块累计；cuDSS 解共享稀疏线性系统**。Python 只准备数据、启动任务、保存结果、组织审核。

Ceres 留作开发期的小问题数值对照，不是第二个生产求解入口，不负责真实节目主循环。已经可靠的 CUDA 数学可以迁入新内核；不能继续从 Python 回调到 Ceres，在每个评估点跨语言拼装逐观测 Jacobian。决定采用小范围 LM 的理由是本问题需要 GPU 上按相关观测组消元与累计，Ceres 的公开 CostFunction 接口以残差和 Jacobian 为边界，不能直接消费本设计已在 GPU 累计好的正规方程。自行维护的范围严格限制在已有标准 LM 算法、块布局和任务生命周期，不自造稀疏分解器、通用约束框架或自动求解器选择器。

官方 Ceres 已支持 CUDA + cuDSS 的稀疏线性求解。因此“Ceres 做不了 GPU”是不成立的。当前仓库的 GPU bridge 仍把 sparse_normal / sparse_schur 配成 EIGEN_SPARSE，是当前集成选择。上面的决策是项目架构选择，不是 Ceres 能力结论。

必须撤回或修正上一轮这些做法：

- 不再以 G0 审计作为重构启动前置。已知缺陷归入相应模块的实际替换工作。
- 撤回 S01 每轮 2 秒、旧六镜头子集每轮 5 秒的无实测预期。完整冈田芽衣短节目是 S01–S09 九镜头，六镜头不是完成范围。当前只定义计时方法与资源上限，最终数值由 5070 Ti 上的真实数据给出。
- “静态背景跟踪”限定为已确认属于冰面或围挡材料的点。看台、人物、字幕、倒影和未知深度点不得进入当前场地求解。
- 不把全部 legacy_import 设为不可用；已确认旧观测允许语义、时钟、坐标等价迁移。不要求每行新机器观测都有人类复核签名。
- 不把固定光心当作默认物理事实。保留非零杆臂及姿态导致光心变化的能力；旧模型已有的轴间距、入口瞳项先准确读取，不能在导入时静默归零。
- 不无条件把每个像素的预测行和 Query 参数升级为全局未知量；滚动快门按来源明确的模式进入同一前向核。
- 用户目视接受是唯一画面质量验收。数值下降、内核一致、源数据完整只是工程正确性条件。

## 2. 对外接口与所有权

数值核心只消费一份已经独立可读的 `PreparedProblem`，不读取旧审计包、旧日志或历史生产链。赛事名字、镜头号、广告品牌不出现在优化器分支中。

| 对象 | 所有权与内容 | 不允许的职责 |
|---|---|---|
| `SceneDefinition` | 实体拓扑、材料测量定义、几何关系、支持区间 | 不保存逐帧相机或像素 |
| `SceneParameters` | 本安装时期共同的场地、标线、围挡、广告参数 | 不出现每镜头一份冰场 |
| `CameraRig` | 真实物理安装、轴、杆臂、镜头模型、读出模式 | 不用一个“相机型号”强迫不同机器共享数值 |
| `ShotTrajectory` | 一镜头的时间范围、云台角度/焦距曲线与活动参数 | 不重定义场景实体 |
| `ObservationTable` | 不可变源 UV、整数 PTS、实体身份、支持、噪声和准入状态 | 不把模型投影写回观测 |
| `FactorPlan` | 参数依赖、局部变量连通组、白化布局、稀疏结构 | 不制造新的人工事实 |
| `State` | 一份共享连续参数向量、稳定 ID 到偏移的映射 | 不保存多个互不对应的“最新模型” |
| `RunState` | LM 阻尼、有效活动集、优化候选、耗时和终止原因 | 不写入用户已接受状态 |

具体 CLI 参数、数组 dtype/shape、状态、错误与输出格式统一由 [INTERFACES.md](INTERFACES.md) 定义。当前已有 API 包括该文档列出的共同几何、静态及完整时变相机前向、prepare/review 和 D1/D3；下面是待实现的唯一聚合求解生命周期，不是已经能运行的命令：

```cpp
CompiledProblem compile_problem(const PreparedProblemView&, const ActiveSet&,
                                const BackendOptions&);
RunResult solve(CompiledProblem&, State&, RunState&,
                const RunOptions&, RunControl&);
ProjectionBatch project(const CompiledModel&, const State&,
                        ArrayView<ProjectionRequest>, ProjectionWorkspace&);
Checkpoint snapshot(const CompiledProblem&, const State&, const RunState&);
void request_stop(RunControl&);
```

准备由 Python `session.prepare.prepare(...)` 一次性产生自包含 PreparedDataset；`session.solve.solve(...)` 与 `session.resume.resume(...)` 组织同一个 native `solve`，后者恢复完整 State/RunState，不另有续算算法。`step` 只在 LM 内部使用，不公开另一套 prepare/step/snapshot/cancel 引擎。`snapshot/request_stop` 是同一生命周期的安全点辅助操作；数值接受步不改变用户通过状态。

完整范围从 `events/ankara_2026/programmes/okada_mei_sp/rebuild_scope.json` 读取，必须覆盖 s01–s09；公共核心不硬编码镜头数。单镜头/多镜头阶段显式保存 selected 与 required，子集不能报告完整节目。原片物理镜头区间和用户要求处理区间分别保留，处理终点不能改名为切镜点。

`project()` 与 residual evaluator 必须调用同一套 `geometry.hpp` / `projection.hpp` 中的数学；渲染器不能维护另一份相机公式。对已导入旧模型的回放采用标记为只读的旧几何常量，但调用同一前向核；这些独立旧几何数值不会自动注册为新求解自由度。

## 3. 共享场景参数：关系在构造时成立

世界坐标：冰面 Z=0，场地几何中心为原点，场地长边为 X。先以单一 L=1 定规；只有有来源的米制标定才增加一个统一比例，不能独立拉伸 XY。场景原点/轴/尺度的规约与现场测量分开保存。

本安卡拉场景的确认关系写成生成器，公共引擎不硬编码 F/R 的数量或标签：

- 四行 `y_i = y0 + i*pitch`；`F_i=(xF,y_i,0)`、`R_i=(xR,y_i,0)`。两列不强制关于中场镜像。
- Lane 中线引用同一 `y_i`，上下边引用同一个共同宽度。已确认的右边线共用一条纵向 station；各左侧源支持范围不同，不恢复已撤销的等长关系，不生成左封口。
- 八组同规格大本营引用共同外半径与同心半径比。未观测实体可以在模型库存中标为推断/待观测，不伪造源约束。
- C0 由场地中心生成；蓝线关于中心对称；共用冰球坐标架。保留共同 paint angle 的能力须以场地关系记录允许为准，不能让每条线独立旋转。
- 围挡表面 `B(s,z)` 使用一个连续封闭边界和共同高度方向。当前真实超椭圆弯角必须支持，不能为了实现方便换成四个圆弧再称等价。
- 广告在展开面 `(s,z)` 中有实例位姿及模板尺寸，三维位置由 `B` 生成。可跨直边与弯角。相同模板/品牌中已确认等尺寸的实例共享尺寸，位置和离冰高度独立。

正尺寸用 log 参数化；半径排序用正增量构造。有限轮廓、超椭圆参数域和板面几何必须在构造时检查。未知图元在编译阶段报出 ID 和类型，不能跳过后宣称“全部已检查”。

### 编译输入只允许有限类型，不能发展成任意表达式语言

`SceneDefinition` 使用封闭的 `PrimitiveKind` / `RelationKind` 枚举、稳定实体 ID、参数引用和有限数组。首版类型是点、有限直线、圆/弧、围挡表面路径、模板路径、两实体交点以及下列布局生成器；不接受字符串公式、Python eval、任意 JSON 运算图或事件专用脚本注入。新增真正不同的原语必须在公共核心增加有类型实现与导数，不在赛事配置里藏代码。

| 场景聚合类型 | 主要字段 | 生成的关系 |
|---|---|---|
| `PairedRows` | `row_count, x_f, x_r, y0, pitch, outer_radius, ring_ratios[], lane_width, lane_right_x, left_source_support[], centreline_support[]` | 圈心共列/等距/配对、共同半径、Lane中线过圈心、等宽；support不冒充实际端点 |
| `HockeyLayout` | `common_radius, blue_offset, goal_station[2], faceoff_inward[2], faceoff_half_span[2], crease_shape[2], trapezoid_goal_span, trapezoid_board_span, paint_frame` | C0中心由场地生成、B对称、同端争球圈关系；仅来源明确的跨端参数引用同一个block |
| `Board` | `length_gauge, width, height, corner_rx, corner_ry, corner_power, boundary_family` | 连续闭合围挡及展开表面；当前superellipse族不得被circle族冒名替换 |
| `PrintTemplate` | `template_id, normalized_paths[], size_class_id, aspect_policy, measurement_definition` | 共同图案轮廓、已确认同尺寸或同宽高比关系 |
| `PrintInstance` | `instance_id, template_id, station_s, height_z, rotation_sz, size_class_ref` | 展开面安装与跨直/弯边映射；不自由优化每个采样顶点 |

这些是目标聚合数据字段，不代表本轮首层已实现全部图元。不存在的实际场地实体不因类型支持就自动加入。

### 固定点不是最近线上的一个自由位置

`Intersection {parent_a, parent_b, branch, measurement_definition}` 的世界位置由共享几何求交。branch 表示已确认哪一交点/哪一侧，不能每次迭代取离像素最近的分支。相交必须落在实体的真实支持范围内；“延长线数学交点”可以用于分析，未在原片建立来源时不能作为观测。

对于固定交点，所有帧共用这个实体，绝无逐帧 station。对于普通曲线采样，局部 station 仅在所属有限实体范围内优化。对于跨帧匹配到同一个真实材料斑点的追踪，station 是共享材料点的位置，不是每帧各自一个自由点。不能将椭圆拟合中心当作圆心的投影。

## 4. 相机模型：安装、运动、镜头与广播坐标分开

世界到相机为 `Xc=Rcw(t)*(Xw-Cw(t))`。基础安装族采用源模型的 pan/tilt 轴及非零相机局部杆臂，`Cw(t)` 从安装支点、转动和杆臂推导，不能为每帧直接自由拟合 XYZ。

| 参数块 | 共享范围 | 活动策略 |
|---|---|---|
| 安装支点 P、轴方向、参考姿态、杆臂 l | 同一物理安装时期 | 保留旧值与来源；有跨姿态证据时估计，不把初值认作测量 |
| 云台角与 log(f) 样条 | 单镜头 | 首先活动的主参数；切镜处分开 |
| 主点、像素比例、skew、畸变基系数 | 真实相机/镜头状态 | 按证据逐族开放，不全部逐帧自由 |
| 变焦相关畸变/主点/入口瞳项 | 同一机组及覆盖的焦距范围 | 共享低维曲线；保留已存在族的准确回放，不宣称某个 log(f) 规律是厂家定律 |
| 广播 crop/resize/像素域变换 | 特定源视频/已知处理阶段 | 从媒体登记读取；未知时标为条件性，不混进焦距解释 |
| 读出时间、方向、参考时刻 | 源成像模式/机组 | 已有值准确继承；缺乏证据不能为了模型完整强行解锁 |

初期生产曲线使用有物理含义的 pan/tilt 与 log(f) 低维样条，不默认每一曝光都有三个独立自由控制点。主线的安装参考姿态是完整 SO(3)，包含固定安装 roll；不能只保存两个欧拉角。旧模型完整三维姿态、已有 roll、轴间距/入口瞳项全部按旧 family 准确回放。若旧模型含时间变化 roll，导入状态保留它，不能以“新模型只有 pan/tilt”为由丢失；是否将其解释为固定安装或保留滚转曲线，是有独立对照的模型变更。一个旧字段尚未支持时，导入明确失败，不静默简化。

新生产拟合初始使用每 0.25 秒一个结点、含两端所需支撑的三次样条；这是可配置的数值初值，不是相机运动硬事实。d78 密集曲线的准确回放仍保存原结点；投影误差受控的压缩转换另外记录误差，不能伪称完全等价。

样条增结只在外层 refinement 中执行，不在一次 LM trial 中改变图：

1. 当前活动参数的优化达到数值停止或稳定后，检查是否仍有同方向、随时间变化的运动型残差。一个候选区间至少要覆盖三个不同原生 PTS、两个几何方向不同且身份已确认的材料实体；优先同时使用冰面与围挡。仅一个边缘/广告失败、未知身份或遮挡簇，不触发增结。
2. 用新增结点对应的相机运动导数列评估其可观测性。只有新列在已有列之外提供有效方向，才提出该区间中点；近零/线性相关的新列不得注册。
3. 采用精确 knot insertion 保持当前曲线不变，再开放新增自由度。这个操作在数值误差内不得改变一帧投影。随后更新 FactorPlan、RS 支撑并集和 cuDSS 符号结构。
4. 对候选进行相同目标的优化，用预先划分的调参选择组选择增结。首版默认要求该组白化误差平方和至少下降 1%，且各已接受镜头在该组上的误差均不增加，才保留这次增结；该 1% 是 RunOptions 中明确的数值选择规则，不是视觉通过标准。调参选择组已经参与模型选择，不能再称为最终未参与调参证据。选择证据不足时不自动增结，报告不足而非自动加满每帧控制点。
5. 按预期收益最高的有效区间逐次执行；已拒绝且证据未变化的区间不重复尝试。不设置“每帧自动加三个控制点”的退路。控制密度始终不改变源 50fps、原生 PTS 或观测准入。

上述计数和阈值是首版可调数值配置，不升级为项目长期规则或新的用户审核关卡。完整镜头最终仍由用户目视验收。

参考角为零等规约仅消除参数化冗余，不能偷偷固定实际轴朝向。对于单平面、狭小视野、仅共线点等欠约束情况，不开放无法辨识的畸变/主点/几何组合。报告相关的参数列与证据覆盖；不足的自由度维持显式的条件假设，而不是让算法通过极端参数假装确定。

必要的 `Rig` 与 `Shot` 字段必须精确登记：

- `Rig`：`rig_id, installation_id, coordinate_frame_id, mount_family, reference_rotation, pivot_world, pan_axis, tilt_axis, lever_camera, axis_spacing, pupil_model, sensor_geometry, lens_family, lens_coefficients, zoom_reference, readout_mode`。`sensor_geometry` 包含源像素尺寸和像素轴约定；`readout_mode` 内含 tau、方向和参考时刻定义。family不认识时显式失败，不能忽略旧wide扩展或入口瞳项。
- `Shot`：`shot_id, rig_id, source_id, first_pts, last_pts, reference_pts, native_time_base, spline_knots, pan_controls, tilt_controls, log_f_controls, optional_roll_controls, readout_padding, active_block_ids`。原片帧身份始终是 source+整数PTS；`time_base` 必须与源一致，不能用显示帧号或 `frame/fps` 代替。源像素域的 crop/resize 由 source_id 引用同一媒体记录。

### 滚动快门的实施边界

源已声明全局快门/等效单时刻时，直接用帧时刻。源明确使用滚动快门时，行曝光时刻为 `t(v)=t_ref + tau * row_fraction(v)`，参考时刻与读出方向必须来自同一源定义。

生产前向投影求满足 `v = pi_y(X,t(v))` 的行根，并使用隐函数导数。数值求根属于局部前向算子，不把每个 row 变成长期全局变量。仅在明确的源似然模型中，允许用观测行近似作初值；不能让拟合用观测行、渲染用预测行而不记录差异。根不唯一、支撑时间越界或导数奇异时返回明确状态，拒绝该 trial；不能丢掉失败行后继续降低目标。

读出时间改变会跨越样条结点。编译依赖取声明读出域内的支持并集；越过该域就更新布局并重新符号分析，不把漏掉的 Jacobian 项当作零。

## 5. 观测准入与目标函数

准入由源与实体身份、测量定义、独立源质量决定，不能由当前拟合残差大小决定。人工已确认观测可等价导入；属于已确认实体、通过来源/跟踪/模糊/遮挡检查的机器观测自动加入。未知实体、被用户否决、合成观测不能进入真实拟合。低置信疑义按实体与时间簇提交审核，避免再次要求用户逐帧标注。

固定观测的像素误差：`r_i(theta)=project(X_i(theta_scene),theta_camera,t_i)-uv_i`。曲线观测为 `r_i(theta,u_i)`；真实边界/中心线的意义由实体测量定义决定。首版生产目标是明确几何观测的似然：原始光度 profile 用于准备阶段估计中心/边界的UV、噪声和相关关系，原始像素样本仍保留可追溯。已经明确的旧UV与W等价迁移；只能按旧Query模型解释、尚不能等价转换的源组单独标为需要适配，不能借Query参数改变像素身份或仅改标签冒充等价。源图搜索alpha不进入全局求解未知量；不为此另建第二套光度求解器。

总目标固定为：

`F(theta,u) = 1/2 sum_g rho_rank(g)( || A_g r_g(theta,u) ||^2 )
              + 1/2 ||r_motion(theta)||^2
              + 1/2 ||r_measured_prior(theta)||^2`。

`A_g` 是源相关组的白化算子，保持其行顺序、坐标单位与有效秩。不把人工像素容许误差当作噪声标准差。首版材料观测统一使用按有效秩归一的 Huber：`rho_d(s)=d*huber(s/d,delta)`，`delta=2.5`（白化后的无量纲 RMS）；`huber(a,delta)=a` 当 `a<=delta^2`，否则为 `2*delta*sqrt(a)-delta^2`。阈值归 RunOptions，不由某次拟合难易偷偷修改；已测先验和运动项为二次项。不能任意逐分量 robustify 一个可任意旋转的白化基并声称统计意义不变。

- 已确认恒等几何关系已由构造实现，不再加入重复罚项。
- 工程像素预算成为始终显示的验收指标；删除“先仅最小化全局 b，再设 b=1”的双重目标状态机。
- 已知安装/尺寸测量按其真实不确定度形成先验；标准尺寸只作工程参考，不伪装成现场测量。
- 运动项采用 pan/tilt 和 log(f) 二阶时间导数的连续积分（按样条区间积分或一致求积），不直接把控制点二阶差分不加时间尺度地相加；增结不能无意改变先验强度。它是明确记录的运动先验，强度用调参选择组比较确定；冻结的最终检查组只在候选定稿后报告表现，不反复反馈选择先验强度。不能在渲染阶段再平滑网格。
- 同一原始线条证据生成曲线与交点时，保留相关关系或选择明确的非重复表示。新交点只替换同源证据，不删除同曝光的其他独立观测。
- 一次求解冻结的是观测准入、源测量协方差、损失函数及其配置。IRLS 的 `rho_prime` 权重在每个已接受状态的新线性化处更新，在同一线性化的拒绝 trial 重试中保持不变；每个 trial 都重新计算真实 robust 目标。因源质量修改需要新数据版本及变化列表；数值优化不能自行删掉最难拟合的东西。

### 三类证据分开使用，不能把调参数据伪装成最终检查

准备时建立三份固定用途集合：训练组进入拟合目标；调参选择组选择样条增结、运动先验强度和已声明的模型选项；冻结最终检查组只在候选定稿后用于报告未参与调参的表现。最终检查组不进入自动增结、参数解锁或先验选择循环。调参选择组虽然不进入该次连续参数拟合，其结果已影响模型选择，因此绝不能称为最终独立证据。

划分以原始相关组、同一材料track以及连续时间块为单位；同一源线条衍生的点/交点、同一相关白化组、相邻帧高度相关的跟踪不得分散到训练和检查两侧。必要时合并其关联闭包再分配，并记录时段边界及用途。若真实数据不足以形成独立最终检查组，明确报告这一限制，不按行随机切分制造独立性。最终检查结果一旦被用来修正模型，该组对后续版本就成为已参与开发的证据，后续报告如实改变其身份；不能仍标为未参与调参，也不把重新人工标注大量帧作为默认补救。用户仍连续审核完整镜头，数据分组不限制其视觉验收范围。

### 唯一因子目录与参数依赖

所有因子登记在同一个 `FactorKind` 表中，以下“局部维度”指消元候选，不代表已经独立可消元；共享材料点或相关噪声会把它们连接起来。

| 因子 | 原始残差维度 | 依赖参数块 | 局部维度 | 失效规则 |
|---|---:|---|---:|---|
| 固定点/已确认交点投影 | 2 | 点的父级scene参数、Rig、该PTS支撑的Shot控制 | 0 | 未知身份、错误交点分支/支持、投影深度或RS域失败 |
| 有限冰面曲线采样 | 2 | 曲线scene参数、Rig、Shot支持、station | 1；同一真实材料点跨帧共享 | station越真实支持、误把外边界当中线、原语退化 |
| 围挡/广告固定角点 | 2 | Board、PrintTemplate/Instance、Rig、Shot支持 | 0 | 展开面定义/模板身份错误、未支持原语、投影域失败 |
| 围挡/广告轮廓采样 | 2 | 上述块及所属轮廓station | 1 | 跨错轮廓或错误有限支持；不能独立移动轮廓顶点 |
| 相机运动积分 | 按求积节点与活动运动通道 | 同一Shot局部支撑的运动控制 | 0 | 跨切镜连接、错误时间单位/积分权重 |
| 真实测量先验 | 测量的有效秩 | 被测的Scene/Rig/Lens/Scale块 | 0 | 单位/来源/安装时期不一致；标准尺寸不能冒充实测 |

相关组的白化不是另一套几何因子：它对相应原始残差及其依赖做一致变换，保留有效秩。对表内因子统一使用同一 registry、同一目标、同一LM；没有 advertisement_solver、s01_solver 或 joint6_solver。物理有效性检查是 trial domain 检查，不靠无限大伪残差或把失效行删除来通过。

## 6. 初始化与活动变量：通过配置逐步扩大同一个问题

初始化属于同一模型的状态构造，不另建 S01 专用优化器。

1. 导入已经存在的完整 d78 / 多镜头状态，保留精确相机、源时间、旧几何和观测身份。不可读取的字段明确报错；不默认归零。
2. 构造符合确认关系的新共享场景。旧几何与新几何的差异单独可见，不能在第一次求解前被“兼容转换”隐藏。
3. 首先利用空间分散的固定交点、真实广告/围挡角点及普通曲线来调整主要相机曲线。已有 d78 相机是初值，不能宣称移植到新场景后还保持原投影。
4. 再开放可观测的共享场景参数和安装参数。广告模板/位置与相机使用相同目标，有活动块阶段，无另起一套广告 solver。
5. 在多镜头证据覆盖足够后开放共享镜头与变焦相关项。每次只开放有明确残差解释和数据覆盖的一族，记录为何需要；不是用更多参数吞掉错误标注。

新的赛事如果没有旧相机，可用平面对应/已知板面点得到姿态和焦距初值；这只是初始化。平面单应性不能单独证明完整三维和镜头内参。求解前的可观测性报告应指出共线点、纯平面、单一变焦、实体覆盖不足等具体缺口。

首个 S01 运行使用以下明确的 ActiveSet，内部阶段无需另开用户审批：

| 数值阶段 | 活动参数 | 保留初值而不改动的参数 |
|---|---|---|
| 相机初始调整 | 该镜头 pan/tilt、log(f) 曲线；已存在且不能等价去除的动态 roll | 当前结构化场景、完整安装、镜头/RS 参数、广告模板 |
| 共享场景调整 | 上述运动 + 有来源支持的共同场地/标线参数 + 广告安装参数；已确认等式始终成立 | 镜头高阶/变焦项、尚不可辨识的安装轴/杆臂、RS |
| 安装调整 | 上述参数 + 本安装时期支点、杆臂、轴图中具独立证据的列 | 主点/畸变/读出项默认仍不自动解锁 |
| 有证据的镜头调整 | 已声明的一族镜头参数；数据覆盖需区分视场半径/变焦与实体身份错误 | 其他未经证据支持的镜头族和自由度 |

这是同一求解器的活动块配置，不是四套目标。如果残差明确呈现随图像半径/变焦变化的结构，应检查镜头族，不能让场景无限变形去吸收。安装或镜头列缺乏独立证据时保留条件性初值并指出可辨识性不足，不宣称这些值已经被现场测量证实。

新增镜头默认只开放该镜头的运动曲线；不自动解锁旧镜头镜头模型、既有场地或全部杆臂。之后按本轮已授权的共享调整阶段开放相应块，并同时显示旧镜头对照。活动参数集合是一份 `ActiveSet`，不同阶段仍编译相同实体、目标与核；严禁 `if shot_id == "S01"` 的数学分支。新增镜头是新增 ShotTrajectory 和观测引用，不复制优化器。

## 7. 一次迭代的精确定义

在已定规和缩放的切空间中，逐相关组先应用白化及 `sqrt(rho_prime(s))` 稳健权重，再形成 `H=J^T J`、`g=J^T r`。这是明确的 Gauss–Newton / IRLS 近似，不把稳健损失的负曲率二阶项放入要求 SPD 的线性系统。trial 接受仍用未近似的实际总目标。参数单位缩放是数值变换，不是新的物理先验。首版在已声明的缩放切空间中采用 D=I，不再叠加另一套隐藏的自动单位缩放。相机角采用局部旋转/轴图；正尺寸采用 log；有限 station 使用有界活动集。

把局部变量记为 l、保留变量记为 q。**LM 阻尼要在局部消元之前加入**：

`A = H_qq + lambda*D_q`

`C = H_ll + lambda*D_l`

`S = A - H_ql * C^-1 * H_lq`

`rhs = -g_q + H_ql * C^-1 * g_l`

cuDSS 求 `S*delta_q=rhs`，然后局部回代。固定交点没有 station，不产生 l。共享材料点的 station 不属于逐曝光独立变量；若跨帧/相关组连接，按真实连通关系处理。

相关白化会耦合局部变量，不能逐行独立 Schur。依据局部变量图的连通分量分组；小连通分量 GPU 消元。若某组太大或其填充不适合局部消元，就保留在同一个全局系统中，而不是拆开相关性。若预计全局分解仍超过预算，停止并报告具体结构和内存，不能无声切换另一种优化问题。

有限 station 的活动集在边界处检查下降方向；朝域外的活动分量固定，其他维度求解。trial 使用可行步长，并以实际应用的增量重新计算预测下降。不能把分量裁剪后还用原未裁剪增量计算信赖域比率。

```cpp
while (!stop_requested && !converged) {
    // 准入、实体身份、源白化W与loss配置冻结；
    // IRLS权重在每次已接受状态的新线性化处更新，拒绝trial内不变。
    auto lin = gpu.linearize_and_reduce(problem, accepted_state);
    for (;;) {
        auto damped = damp_before_elimination(lin, lambda, active_bounds);
        auto step = cudss.factor_and_solve(damped.reduced_system);
        recover_local_step(damped, step);
        auto trial = retract_feasibly(accepted_state, step);
        auto actual_step = difference(trial, accepted_state);
        double predicted = quadratic_decrease(lin, actual_step);
        auto value = gpu.evaluate_objective(problem, trial);
        if (value.domain_ok && predicted > 0 && value.cost < lin.cost
            && gain_ratio(value, lin, predicted) > acceptance_ratio) {
            accepted_state = trial;
            update_damping_after_acceptance();
            publish_iteration_metrics();
            break;
        }
        increase_damping();
        if (pause_or_numerical_failure()) return checkpointed_status();
    }
}
```

首版数值规则固定为 Nielsen 式 LM 更新，所有常量集中在 RunOptions：`lambda0 = 1e-3 * max(diag(H0))`；若最大对角为零，返回无可观测下降方向。`eta=1e-4`，只有实际目标下降且 `rho>eta` 才接受；接受时 `lambda *= max(1/3,1-(2*rho-1)^3)` 并令 `nu=2`；拒绝时 `lambda*=nu`、`nu*=2`。检查有限值与线性解残差，不能溢出后继续执行。

数值停止条件：缩放切空间 `||g||_infinity <= 1e-6`；或者一次接受步同时满足 `||delta||_infinity <= 1e-7` 且相对目标下降 `<=1e-8`。RunOptions 明确允许调整这些实现默认值；默认最大 100 次新线性化作为单次运行数值预算，用户授权的任务可根据报告继续同一阶段。暂停、用户停止、预算用尽、无可行步、不可辨识、数值失败、达到数值停止分别返回不同原因；不得把预算用尽写成收敛。以上均不是画面质量阈值，不构成新的项目审批。

上述 `accepted_state` 仅表示优化器接受的一步，不表示用户验收。非有限数、深度错误、实体分支消失、RS 不可解、非正定或线性残差过大，都记录具体首批实体/参数 ID。错误 trial 不覆盖最后可用状态。

拒绝 trial 后重用同一线性化，但局部消元必须随阻尼重新数值更新。不能因为旧 Schur 已算好就只给消元结果加一个对角项冒充同一 LM。

## 8. GPU 数据布局与内存管理

### 常驻数据

采用 SoA 连续数组：整数 PTS、shot index、entity index、UV、来源/相关组索引、有限支持、噪声因子和只读的实体拓扑。稳定字符串 ID 只保留在主机映射表和结果文件中，CUDA 热路径不遍历 Python 对象、JSON 或字符串。

`FactorPlan` 在准备/布局改变时计算：各因子依赖参数块、块偏移、局部变量组、H 的 CSR/BSR 目标地址、分组归约顺序。正常迭代只更新值。完整 J 不作为主机常驻对象，也不在每次迭代回传。

导数沿同一数学依赖链计算：每个几何原语输出 XYZ 及对少量场景参数的导数；相机核输出对 XYZ、时间、云台角、焦距及安装/镜头参数的导数；样条按局部基函数链到其支撑控制点。现有解析相机公式可迁入；复杂几何可用局部小维度前向自动微分。公共数学头支持主机/设备与标量模板，禁止为 GPU 复制另一套公式，也禁止构建长度等于全局未知量数的巨大 Dual 向量。逐参数整批有限差分只供开发对照，不进入生产迭代。

### 计算顺序

1. 同一核计算场景点、运动、镜头、投影及必要局部导数。
2. 按相关组应用白化和稳健权重，生成小组局部 J / 累计块；组超过 tile 大小时分块运算，但保留所有相关跨项。
3. 对参数块对做分段归约，形成全局稀疏 H/g 及局部 Schur 资料。
4. GPU 完成局部消元、写回约定 CSR 值；cuDSS 在相同 stream / 明确 event 依赖下分解求解。
5. GPU 求 trial 目标。通常仅把标量目标、状态码、时序和少量诊断计数回传。
6. 需要检查点或显示时才拷回状态向量和所请求投影，不回传所有逐点导数。

首版一个主 compute stream 加一个受 event 保护的传输 stream 即可。cuDSS API 有部分同步限制，不能假定所有调用都异步。CUDA Graph 仅在稳定布局的测量证明 kernel launch 成为主要成本后引入，不作为第一版必需层。

### 精度与容量

首先用 FP64 参考小数据建立数值正确性。生产候选可采用 FP32 几何/局部导数、FP64 块累计与线性求解，但必须测量真实条件数下的投影、导数方向与步长一致性；不默认 FP16/TF32。消费级卡上的 FP64 速度和稀疏填充必须实测，不能凭 TFLOPS 广告推断迭代耗时。

对 16GB VRAM / 32GB RAM 的机器，启动时报告可用容量与用户可见预算；可以以约 12GiB 设备、24GiB 主机作为明确的初始工程配置，不把它描述为已经测到的占用。真正容量由图结构决定：观测常驻数组通常线性于行数，块矩阵与因子分解可能产生填充。cuDSS 符号分析后的内存估计要参与判断。

仅用 N×参数数估计 Jacobian 内存是不够的。必须输出观测数、活动参数数、局部组最大维度、矩阵非零数和预计分解内存。观测 tile 可以减少工作缓冲，不能解决全局分解填充超过显存的问题。

## 9. cuDSS 的使用边界

- 首次构建 CSR 结构后运行 analysis；数值值变化时复用符号结构并重新 factorization/solve。
- 新增镜头、增加样条结点、改变相关组或活动参数会改变稀疏结构，重新 analysis。
- 仅改变几何参数、相机值、LM 阻尼或同结构的噪声数值，不重建全套输入。
- 选择与实际矩阵属性一致的 SPD 模式；不能只因为期望 H 正定就无视求解器返回的非正定/数值错误。
- 记录分解状态与 `||S delta - rhs||` 等内部数值诊断；这些是求解正确性检查，不是视觉拟合评分。
- 首版不引入 GPU PCG、第二套 Schur 体系、多 GPU 或混合主机溢出路径。若真实结构超预算，先修正数据布局/稀疏性或调整明确的求解批次，不静默换算法。

## 10. 文件责任与接口落点

本轮首层实现文件（实际能力以根交付的运行记录为准）：

| 路径 | 当前责任 |
|---|---|
| `native/include/rinkspace/core_types.hpp` | 公共 POD 数值类型、状态码与坐标约定 |
| `native/include/rinkspace/geometry.hpp` | 统一场景几何、固定节点、围挡展开映射 |
| `native/include/rinkspace/projection.hpp` | 同源前向相机投影 |
| `native/include/rinkspace/core_api.h` | 薄 C ABI |
| `native/src/core_api.cpp` | CPU 编译及 C ABI 实现 |
| `src/rinkspace/domain/contracts.py` | 类型、来源身份、时间与实体引用 |
| `src/rinkspace/core/native.py` | 薄绑定；不重写几何公式 |
| `src/rinkspace/session/core_preview.py` | 首层模型预览，用同一核心生成可见几何 |

待实现的生产求解部分：

| 路径 | 唯一责任 |
|---|---|
| `native/include/rinkspace/problem.hpp` | Prepared/CompiledProblem、参数块、观测与 FactorPlan |
| `native/include/rinkspace/solver.hpp` | RunOptions、RunResult、暂停与回调 API |
| `native/src/problem.cpp` | 实体引用解析、活动参数、相关组与稀疏依赖 |
| `native/gpu/evaluate.cu` | 几何→相机→残差及局部导数；调用公共数学头 |
| `native/gpu/reduce.cu` | 白化、稳健权重、块累计 |
| `native/gpu/local_eliminate.cu` | 相关局部变量分组消元/回代 |
| `native/src/cudss_system.cpp` | 稀疏分析、分解、求解、状态及内存估计 |
| `native/src/solver.cpp` | 唯一 LM 状态机；无赛事、广告、S01 特判 |
| `native/src/checkpoint.cpp` | 原子候选保存和数值续算状态 |
| `src/rinkspace/session/prepare.py` | 当前输入的一次性编译，变更仅重建受影响部分 |
| `src/rinkspace/session/solve.py` | 当前输入加载、工作目录、进程生命周期 |
| `src/rinkspace/session/resume.py` | 同输入/布局的数值续算，不重跑准备 |
| `src/rinkspace/review/render.py` | 调用同一核心的完整镜头审核导出 |
| `src/rinkspace/review/compare.py` | 同 PTS、同可见库存的基线与候选比较 |

新 review 包在对应批次一次性替换旧 `review.py`，不允许同名模块与包并存。C ABI 只有 `core_api.h/core_api.cpp`，不另建空 bindings 框架。

文件存在不等于功能完成。未实现能力必须明确报 unavailable；不能让空的 `solve()` 返回成功或把核函数占位注册为有效 backend。

### 最早的 GPU 实跑切片

第二个实际代码批次就必须贯通常驻 GPU 的完整一次 LM 迭代：使用等价迁移的当前真实固定点/有限曲线输入，执行残差和J、H/g累计、阻尼、局部消元、cuDSS、trial目标、接受/拒绝和checkpoint。它先支持已声明的因子子集，再逐类接入完整目标。不能把只有 CUDA project() 或单位矩阵线性求解的演示称为求解器已启动。

这个工程切片应在检测器全面重写之前完成，直接使用已有可靠观测。每次增加因子沿同一内核与接口扩展，没有第二个临时求解器；尚未接入的因子明确列出，并且该子集结果不能被标成完整S01通过。随后完成全库存审核与整镜头拟合。

## 11. Windows / 5070 Ti 构建与运行

使用单一 CMake 项目与少量明确 preset：CPU reference、Windows CUDA production。复用已有可用工具链；不因为“重构”就重装 Python、CUDA 或 Ceres。先记录现有 NVCC、MSVC、driver 与 cuDSS 版本，选与该 CUDA 主版本匹配的 cuDSS。

5070 Ti 的计算能力是 12.0。编译必须包含适配该卡的 cubin/PTX，并在本机实际加载核；不要误用数据中心 Blackwell 的 sm_100 作为等价目标。CMake 的 CUDA 架构字段必须由支持它的实际工具链验证，不能硬写一个版本字符串就算支持。

cuDSS 在 Windows 运行时 DLL 位于其 `bin`，`lib` 是链接库。新版依赖项随 cuDSS 版本变化，应在一次性构建 recipe 中固定并验证 DLL 解析。运行求解不重新下载或重新编译依赖；只有相应源码/构建配置变化才触发增量构建。安装/构建一次，prepare一次，可以反复solve与resume。仅改参数值不重编结构；改某个观测组只重建其相关组及必要稀疏依赖；改拓扑/活动集才更新对应结构和符号分析。

公开源码、配置、最终参数留项目；编译缓存、原片、观测大数组、运行日志、完整检查点和视频在 `D:\CodexData\RinkSpace_Data`。不改变用户的 Git 数据库位置。

## 12. 运行、暂停、恢复与审核产物

启动必须立即给出本次主日志绝对路径。日志逐次记录目标分项、块数、设备时序、域失败、阻尼、是否接受数值步及当前可见产物的位置，不能长时间只有一个“running”。

数据准备与求解分离。恢复读取当前数据包、模型与 checkpoint 的必要一致性，直接恢复；不因重编译或无关源码修改重新执行历史资格链。输入改变仅检查受影响对象及其引用闭包。

暂停在当前 GPU 操作的安全边界响应，保存最后可用状态。崩溃不覆盖已有 checkpoint；同目录临时文件完成后原子替换。优化候选与用户接受版本分别存储，用户接受只通过明确的审核动作改变。

每个实际代码批次产生与改动相关的可见结果：模型层产生正确关联的场地与节点图；投影层产生对应真实源帧的全模型叠加；求解层产生完整 S01 和残差定位；新增镜头产生新增镜头及旧镜头回归对照。不能拿“编译成功”或一套合成测试代替画面交付。

性能测量必须固定源输入、活动参数、目标、精度和结果质量，并分开报告冷启动、暖启动、线性化、累计消元、分解、trial 评估、输出时间与峰值内存。少量实测迭代足以定位瓶颈后应继续实际拟合，不做无穷 benchmark。没有视觉改善时不反复编码完整节目。

## 13. 官方能力依据与仓库证据

以下官方链接仅证明相应接口/支持情况，不证明本项目实现或速度：

1. Ceres CUDA / cuDSS 支持与构建： https://ceres-solver.readthedocs.io/latest/installation.html
2. Ceres CostFunction / EvaluationCallback： https://ceres-solver.readthedocs.io/latest/nnls_modeling.html
3. cuDSS 数据类型、分阶段、内存估计、Windows 支持： https://docs.nvidia.com/cuda/cudss/latest/index.html
4. cuDSS 分析/分解/求解与 Windows 运行库： https://docs.nvidia.com/cuda/cudss/latest/getting_started.html
5. RTX 5070 Ti 计算能力： https://developer.nvidia.com/cuda/gpus
6. Blackwell cubin/PTX 兼容： https://docs.nvidia.com/cuda/blackwell-compatibility-guide/
7. 标准LM与Nielsen阻尼更新（第2.4、3.2节）： https://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf 。本文有界步、robust IRLS及默认阈值是项目实现选择，不能当作原文全部算法已适用于本项目的证明。

仓库固定证据：`calibration/solve2/rinksolve/{model.py,parameterization.py,optimizer.py,projector.py,whitening.py}`、`calibration/solve2/native/gpu_batch_ceres.cpp`、`events/ankara_2026/model_constraints/geometry_relations_current.json`、`AGENTS.md`。所有当前行为判断绑定上述提交；未来代码变更需要更新对应事实，不自动沿用旧缺陷结论。
