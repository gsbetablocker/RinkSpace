# 求解方法与运行要求

本文维护共同数值模型、GPU 执行流程与运行要求。当前可调用能力见 [REBUILD](REBUILD.md)，精确接口见 [INTERFACES](INTERFACES.md)。驻留 GPU 连续 LM、安全停止与同 handle 继续已实现；聚合生命周期、增量准备缓存和完整磁盘恢复仍按各节所标目标理解。

## 1. 数值架构选择

生产求解器只保留一条路径：**C++ 控制的阻尼 Gauss–Newton / LM；CUDA 计算同一几何、相机、残差、导数和块累计；cuDSS 解共享稀疏线性系统**。Python 只准备数据、启动任务、保存结果、组织审核。

Ceres 留作开发期的小问题数值对照，不是第二个生产求解入口，不负责真实节目主循环。已经可靠的 CUDA 数学可以迁入新内核；不能继续从 Python 回调到 Ceres，在每个评估点跨语言拼装逐观测 Jacobian。决定采用小范围 LM 的理由是本问题需要 GPU 上按相关观测组消元与累计，Ceres 的公开 CostFunction 接口以残差和 Jacobian 为边界，不能直接消费本设计已在 GPU 累计好的正规方程。自行维护的范围严格限制在已有标准 LM 算法、块布局和任务生命周期，不自造稀疏分解器、通用约束框架或自动求解器选择器。

官方 Ceres 已支持 CUDA + cuDSS 的稀疏线性求解。因此“Ceres 做不了 GPU”是不成立的。当前仓库的 GPU bridge 仍把 sparse_normal / sparse_schur 配成 EIGEN_SPARSE，是当前集成选择。上面的决策是项目架构选择，不是 Ceres 能力结论。

场地材料、旧记录及机器观测的准入统一见 [BUILD_SPEC](rink_model/BUILD_SPEC.md#observation-eligibility)，完整原片与用户目视验收见 [VALIDATION](VALIDATION.md)。旧迭代耗时预期、阶段顺序和运行预算不作为当前要求；性能以相同输入和目标的实测为准。

## 2. 对外接口与所有权

核心消费可独立读取的当前 PreparedProblem，不读历史生产链。Scene、Rig、Shot、Observation 与状态所有权见 [ARCHITECTURE](ARCHITECTURE.md#数据所有权)；CLI、数组和唯一聚合生命周期见 [INTERFACES](INTERFACES.md)。prepare 与驻留 native 连续求解已实现；PreparedProblem 的完整 solve/resume 生命周期仍为目标。

数值状态、优化器接受步与用户验收分别保存。镜头全集、处理与物理区间从所属节目清单读取，不在公共核心硬编码镜头数量。

project 与 residual evaluator 调用同一 geometry/projection 数学，渲染器不维护另一套相机公式。旧模型回放可使用有来源的只读旧几何，但不自动将这些值注册为新求解自由度。

显式相机耦合由 [camera_coupling.py](../src/rinkspace/domain/camera_coupling.py) 将多个机位的标量光学方向连接到同一 native 保留列，不添加罚项。共同曲线的焦距参考和广角基函数按固定成像倍率对齐，初值转移现有投影；倍率仍是明确的工作模式或外部事实，不由独立曲线自动认定。全局相机列数按唯一共享列计数。

## 3. 共享场景参数：关系在构造时成立

世界坐标：冰面 Z=0，场地几何中心为原点，场地长边为 X。先以单一 L=1 定规；只有有来源的米制标定才增加一个统一比例，不能独立拉伸 XY。场景原点/轴/尺度的规约与现场测量分开保存。

已确认几何关系由有类型生成器构造；严格名义版本不使用重复罚项“近似成立”。赛事关系及适用版本由[安卡拉场景](../events/ankara_2026/scene/README.md)和[关系记录](../events/ankara_2026/model_constraints/geometry_relations_current.json)维护，公共引擎不硬编码 F/R 数量或赛事标签。

显式选用冰面容差版本（选项 64／layout146）时，原共享参数生成共同名义几何，107 个偏差坐标各自在 [-0.01,+0.01] 内。尺寸按共同名义尺寸缩放，位置按声明的局部名义长度归一化，方向按漂移／前进长度归一化；不会按迭代重新基准化。native 同时维护局部关系范围、图元类型、共享接点、有限支持和严格顺序，使用同一解析关系与导数进行 trial 可行性检查、有界步和原目标停止锥判断。该选项不改变原像素目标，不加入几何罚项；原选项 0–63 保留原解释。赛事配置与零偏差数值迁移见[当前场景](../events/ankara_2026/scene/README.md)。

围挡 B(s,z) 是连续封闭表面；真实超椭圆不能被圆弧近似冒称等价。广告在展开面上有共享模板/尺寸和独立安装，可跨直段与弯段，三维位置由同一 B 生成。

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

这些是目标聚合数据字段，是否可调用以 REBUILD 和实际 schema 为准。不存在的实际场地实体不因类型支持就自动加入。

### 固定点不是最近线上的一个自由位置

`Intersection {parent_a, parent_b, branch, measurement_definition}` 的世界位置由共享几何求交。branch 表示已确认哪一交点/哪一侧，不能每次迭代取离像素最近的分支。相交必须落在实体的真实支持范围内；“延长线数学交点”可以用于分析，未在原片建立来源时不能作为观测。

对于固定交点，所有帧共用这个实体，绝无逐帧 station。对于普通曲线采样，局部 station 仅在所属有限实体范围内优化。对于跨帧匹配到同一个真实材料斑点的追踪，station 是共享材料点的位置，不是每帧各自一个自由点。不能将椭圆拟合中心当作圆心的投影。

## 4. 相机模型：安装、运动、镜头与广播坐标分开

世界到相机为 `Xc=Rcw(t)*(Xw-Cw(t))`。基础安装族采用源模型的 pan/tilt 轴及非零相机局部杆臂，`Cw(t)` 从安装支点、转动和杆臂推导，不能为每帧直接自由拟合 XYZ。

| 参数块 | 共享范围 | 活动策略 |
|---|---|---|
| 安装支点 P、轴方向、参考姿态、杆臂 l | 同一物理安装时期 | 保留旧值与来源；有跨姿态证据时估计，不把初值认作测量 |
| 云台角与 log(f) 样条 | 单镜头 | 按 ActiveSet 选择；切镜处分开 |
| 主点、像素比例、skew、畸变基系数 | 真实相机/镜头状态 | 按证据逐族开放，不全部逐帧自由 |
| 变焦相关畸变/主点/入口瞳项 | 同一机组及覆盖的焦距范围 | 共享低维曲线；保留已存在族的准确回放，不宣称某个 log(f) 规律是厂家定律 |
| 广播 crop/resize/像素域变换 | 特定源视频/已知处理阶段 | 从媒体登记读取；未知时标为条件性，不混进焦距解释 |
| 读出时间、方向、参考时刻 | 源成像模式/机组 | 已有值准确继承；缺乏证据不能为了模型完整强行解锁 |

初期生产曲线使用有物理含义的 pan/tilt 与 log(f) 低维样条，不默认每一曝光都有三个独立自由控制点。主线的安装参考姿态是完整 SO(3)，包含固定安装 roll；不能只保存两个欧拉角。旧模型完整三维姿态、已有 roll、轴间距/入口瞳项全部按旧 family 准确回放。若旧模型含时间变化 roll，导入状态保留它，不能以“新模型只有 pan/tilt”为由丢失；是否将其解释为固定安装或保留滚转曲线，是有独立对照的模型变更。一个旧字段尚未支持时，导入明确失败，不静默简化。

样条结点和控制密度由所选输入与运行配置记录；既有曲线的准确回放保留原结点。压缩转换另外记录投影差，不伪称等价。

增结属于外层 refinement，不在 LM trial 内改变图。候选区间需要跨源时刻、跨几何方向的真实材料证据与可观测的新导数列；单个错误身份、遮挡簇或广告失败不触发盲目加参。精确 knot insertion 先保持当前曲线，再开放新增自由度、更新 FactorPlan、RS 支撑及稀疏符号结构。

增结与先验强度由预先划分的调参选择证据比较，采用的判据显式保存；最终检查组不参与选择。证据不足报告不足，拒绝且证据未变的候选不重复尝试。控制密度不改变原生 PTS、观测准入或完整镜头验收范围。实际结点、容差与预算以所选配置及实现为准。

参考角为零等规约仅消除参数化冗余，不能偷偷固定实际轴朝向。对于单平面、狭小视野、仅共线点等欠约束情况，不开放无法辨识的畸变/主点/几何组合。报告相关的参数列与证据覆盖；不足的自由度维持显式的条件假设，而不是让算法通过极端参数假装确定。

必要的 `Rig` 与 `Shot` 字段必须精确登记：

- `Rig`：`rig_id, installation_id, coordinate_frame_id, mount_family, reference_rotation, pivot_world, pan_axis, tilt_axis, lever_camera, axis_spacing, pupil_model, sensor_geometry, lens_family, lens_coefficients, zoom_reference, readout_mode`。`sensor_geometry` 包含源像素尺寸和像素轴约定；`readout_mode` 内含 tau、方向和参考时刻定义。family不认识时显式失败，不能忽略旧wide扩展或入口瞳项。
- `Shot`：`shot_id, rig_id, source_id, first_pts, stop_pts_exclusive, reference_pts, native_time_base, spline_knots, pan_controls, tilt_controls, log_f_controls, optional_roll_controls, readout_padding, active_block_ids`。原片帧身份始终是 source+整数PTS；`time_base` 必须与源一致，不能用显示帧号或 `frame/fps` 代替。源像素域的 crop/resize 由 source_id 引用同一媒体记录。

### 滚动快门的实施边界

源已声明全局快门/等效单时刻时，直接用帧时刻。源明确使用滚动快门时，行曝光时刻为 `t(v)=t_ref + tau * row_fraction(v)`，参考时刻与读出方向必须来自同一源定义。

生产前向投影求满足 `v = pi_y(X,t(v))` 的行根，并使用隐函数导数。数值求根属于局部前向算子，不把每个 row 变成长期全局变量。仅在明确的源似然模型中，允许用观测行近似作初值；不能让拟合用观测行、渲染用预测行而不记录差异。根不唯一、支撑时间越界或导数奇异时返回明确状态，拒绝该 trial；不能丢掉失败行后继续降低目标。

读出时间改变会跨越样条结点。编译依赖取声明读出域内的支持并集；越过该域就更新布局并重新符号分析，不把漏掉的 Jacobian 项当作零。

## 5. 观测准入与目标函数

观测准入按 [BUILD_SPEC](rink_model/BUILD_SPEC.md#observation-eligibility)执行；数值残差与导入成功均不改变资格。源 UV、相关 W 和原状态保留。

固定点像素误差为 `r_i(theta)=project(X_i(theta_scene),theta_camera,t_i)-uv_i`，曲线观测为 `r_i(theta,u_i)`。测量语义由实体定义，源 UV 不被模型投影覆盖。当前 native 可显式选择下述逐点像素目标或相关目标；旧查询仅在可保持原似然语义的接口中使用，不能改标签冒充等价。源图搜索 alpha 不作为独立全局未知量。

逐点像素目标为 `F(theta,u) = 1/2 sum_i ||project_i(theta,u_i)-uv_i||^2`。此模式直接使用原始源像素，不接入观测权重、协方差、稳健降权或像素容许范围；数据不覆盖所选 native 参数。数值变量缩放和 LM 阻尼不属于观测加权。普通曲线 station 可沿其实体调整，但每个点各自保留两个原像素残差，不能先压缩后再用多个自由 station 抵消整组误差。

历史相关观测入口另支持以下目标；只有调用者明确选择且确认残差与来源统计模型相容时使用，不由数据存在 W 字段自动启用：

`F(theta,u) = 1/2 sum_g rho_rank(g)( || A_g r_g(theta,u) ||^2 )
              + 1/2 ||r_motion(theta)||^2
              + 1/2 ||r_measured_prior(theta)||^2`。

`A_g` 是源相关组的白化算子，保持其行顺序、坐标单位与有效秩。不把人工像素容许误差当作噪声标准差。显式相关模式使用按有效秩归一的 Huber：`rho_d(s)=d*huber(s/d,delta)`，delta 为 RunOptions 中显式记录的白化后无量纲 RMS 阈值；`huber(a,delta)=a` 当 `a<=delta^2`，否则为 `2*delta*sqrt(a)-delta^2`。阈值归 RunOptions，不由某次拟合难易偷偷修改；已测先验和运动项为二次项。不能任意逐分量 robustify 一个可任意旋转的白化基并声称统计意义不变。

- 已确认恒等几何关系已由构造实现，不再加入重复罚项。
- 工程像素误差持续显示为验收指标，不以全局 b 的两阶段优化替代同一统计目标。
- 已知安装/尺寸测量按其真实不确定度形成先验；标准尺寸只作工程参考，不伪装成现场测量。
- 运动项采用 pan/tilt 和 log(f) 二阶时间导数的连续积分（按样条区间积分或一致求积），不直接把控制点二阶差分不加时间尺度地相加；增结不能无意改变先验强度。它是明确记录的运动先验，强度用调参选择组比较确定；冻结的最终检查组只在候选定稿后报告表现，不反复反馈选择先验强度。不能在渲染阶段再平滑网格。
- 同一原始线条证据生成曲线与交点时，保留相关关系或选择明确的非重复表示。新交点只替换同源证据，不删除同曝光的其他独立观测。
- 一次求解冻结的是观测准入、源测量协方差、损失函数及其配置。IRLS 的 `rho_prime` 权重在每个已接受状态的新线性化处更新，在同一线性化的拒绝 trial 重试中保持不变；每个 trial 都重新计算真实 robust 目标。因源质量修改需要新数据版本及变化列表；数值优化不能自行删掉最难拟合的东西。

### 三类证据分开使用，不能把调参数据伪装成最终检查

拟合组进入目标，调参选择组用于增结/先验/模型选项，最终检查组只报告定稿后的独立表现。相关组、同一材料轨迹及同源衍生证据按关联闭包划分；字段见 INTERFACES，冻结、保护邻域、数据不足及旧留出再利用的判定统一见 [VALIDATION](VALIDATION.md#观测与独立性)。

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

初始化消费所选完整状态，保留相机、几何、源时钟和观测身份；未知字段明确失败。旧几何转为新的共享结构时显式展示差异，不声称旧相机移植后仍保持原投影。没有旧状态时可由平面对应或已知板面点产生初值，但这不证明完整三维与内参正确。

ActiveSet 依据当前任务、数据覆盖和可辨识性指定，不固定 S01 开工顺序。运动、场景、安装和镜头参数始终属于同一目标与同一核；冻结/规约项与未可辨识的条件性初值分别记录。半径/变焦相关残差需检查相机族，不让场景无限变形吸收错误。

增加镜头是增加 ShotTrajectory 和观测引用，不复制优化器，也不自动解锁全部共享参数。修改共享块时同时检查已有镜头；具体选择由实际运行配置保存，不由历史阶段表授权。

## 7. 一次迭代的精确定义

在已定规和缩放的切空间中，逐点模式直接由原像素残差形成 `H=J^T J`、`g=J^T r`；显式相关模式先应用 W 与 `sqrt(rho_prime(s))`，形成 Gauss–Newton／IRLS 近似，不将稳健损失的负曲率项放入 SPD 系统。trial 接受重算所选真实目标。参数单位缩放不是物理先验；阻尼度量由配置显式选择，当前默认 D=I。相机使用局部旋转／轴图，正尺寸用 log，有限 station 使用有界活动集。

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
    // 准入、实体身份与所选目标配置冻结；
    // 相关模式在接受状态更新 IRLS 权重，拒绝 trial 内保持。
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
        if (pause_or_numerical_failure()) return stopped_status(accepted_state);
    }
}
```

采用 Nielsen 式阻尼更新：实际下降且增益比 rho 大于显式接受阈值 eta 时接受，lambda 乘 max(1/3,1-(2*rho-1)^3)，nu 复位为 2；拒绝时 lambda 乘 nu、nu 加倍。初始阻尼结合缩放后的 H 对角设定；无可观测下降方向、非有限数或线性解失败需明确返回。

只有在已接受状态重新线性化后，原目标的投影梯度满足容差，才声明数值收敛。边界截断或大阻尼造成的小步与小下降不能单独证明收敛；接受步与代价同时变小只细分停止原因，仍须通过投影梯度判据。容差和实际采用的运行预算保存在 RunOptions。当前连续控制器分别报告梯度收敛、同时满足梯度与接受步／代价判据、浮点停滞、代价进展停滞及调用者停止。代价进展停滞要求上一接受步的预测下降与实际下降（包含回调中的 station 精修）均低于已声明的相对代价容差；若该步的阻尼能量超过预测下降的一半，且正常 LM 更新确实降低了 lambda，则继续降低阻尼尝试，不能仅凭该微小下降终止。其他代价进展停滞返回非收敛，不设额外次数／时间上限，也不证明局部最小值。实际默认值以当前绑定与运行配置为准。暂停、用户停止、预算耗尽、无可行步、不可辨识、数值失败和数值停止分别报告；预算耗尽不是收敛，也不是画面质量判据。

连续求解可以先用原目标的自由坐标证明尚未收敛：若活动约束 Jacobian 的某列全为零，该坐标的投影梯度等于原梯度，因此其绝对值是完整投影无穷范数的严格可用下界。下界超过原收敛阈值时直接进入 LM，候选仍执行全部原物理约束；下界不足以排除收敛时才计算完整切锥投影。若自由最大值还严格超过受约束部分原梯度的二范数上界，则它就是精确的全局投影无穷范数。缓存与日志区分完整投影、精确标量范数和非收敛下界，后者不填入精确范数字段，不据此声明收敛；显式完整诊断不复用缺少投影向量的缓存。调用者停止后只读取接受状态的原目标，不强制重启完整切锥诊断；未知投影梯度返回空值。

上述 `accepted_state` 仅表示优化器接受的一步，不表示用户验收。非有限数、深度错误、实体分支消失、RS 不可解、非正定或线性残差过大，都记录具体首批实体/参数 ID。错误 trial 不覆盖最后可用状态。

拒绝 trial 后重用同一线性化，但局部消元必须随阻尼重新数值更新。不能因为旧 Schur 已算好就只给消元结果加一个对角项冒充同一 LM。

## 8. GPU 数据布局与内存管理

### 常驻数据

常驻 SoA 数组包括整数 PTS、镜头／实体索引、源 UV、有限支持与只读拓扑；相关模式另包含噪声因子与相关组。稳定字符串 ID 留在主机映射和结果文件，CUDA 热路径不遍历 Python 对象、JSON 或字符串。

`FactorPlan` 在准备/布局改变时计算：各因子依赖参数块、块偏移、局部变量组、H 的 CSR/BSR 目标地址、分组归约顺序。正常迭代只更新值。完整 J 不作为主机常驻对象，也不在每次迭代回传。

导数沿同一数学依赖链计算：每个几何原语输出 XYZ 及对少量场景参数的导数；相机核输出对 XYZ、时间、云台角、焦距及安装/镜头参数的导数；样条按局部基函数链到其支撑控制点。现有解析相机公式可迁入；复杂几何可用局部小维度前向自动微分。公共数学头支持主机/设备与标量模板，禁止为 GPU 复制另一套公式，也禁止构建长度等于全局未知量数的巨大 Dual 向量。逐参数整批有限差分只供开发对照，不进入生产迭代。

### 计算顺序

1. 同一核计算场景点、运动、镜头、投影及必要局部导数。
2. 逐点模式直接累计两个像素残差；相关模式按组应用 W 与稳健权重并保留所有交叉项，超出 tile 时分块计算。
3. 对参数块对做分段归约，形成全局稀疏 H/g 及局部 Schur 资料。
4. GPU 完成局部消元、写回约定 CSR 值；cuDSS 在相同 stream / 明确 event 依赖下分解求解。
5. GPU 求 trial 目标。通常仅把标量目标、状态码、时序和少量诊断计数回传。
6. 需要检查点或显示时才拷回状态向量和所请求投影，不回传所有逐点导数。

首版一个主 compute stream 加一个受 event 保护的传输 stream 即可。cuDSS API 有部分同步限制，不能假定所有调用都异步。CUDA Graph 仅在稳定布局的测量证明 kernel launch 成为主要成本后引入，不作为第一版必需层。

### 精度与容量

当前 native 路径使用 FP64。混合精度属于待验证的优化选择，需先测量真实条件数下的投影、导数与步长一致性，不默认 FP16／TF32。GPU 的 FP64 速度和稀疏填充以实际输入测量，不能凭 TFLOPS 推断迭代耗时。

启动时读取设备与主机可用容量，记录实际运行配置中的资源预算；本文不指定固定显存/内存配额。容量依据观测数组、块矩阵及分解填充估计，cuDSS 符号分析的内存估计参与判断，不能把配置预算说成实测占用。

仅用 N×参数数估计 Jacobian 内存是不够的。必须输出观测数、活动参数数、局部组最大维度、矩阵非零数和预计分解内存。观测 tile 可以减少工作缓冲，不能解决全局分解填充超过显存的问题。

## 9. cuDSS 的使用边界

- 首次构建 CSR 结构后运行 analysis；数值值变化时复用符号结构并重新 factorization/solve。
- 新增镜头、增加样条结点、改变相关组或活动参数会改变稀疏结构，重新 analysis。
- 仅改变几何参数、相机值、LM 阻尼或同结构的噪声数值，不重建全套输入。
- 选择与实际矩阵属性一致的 SPD 模式；不能只因为期望 H 正定就无视求解器返回的非正定/数值错误。
- 记录分解状态与 `||S delta - rhs||` 等内部数值诊断；这些是求解正确性检查，不是视觉拟合评分。
- 首版不引入 GPU PCG、第二套 Schur 体系、多 GPU 或混合主机溢出路径。若真实结构超预算，先修正数据布局/稀疏性或调整明确的求解批次，不静默换算法。

## 10. 文件责任与接口落点

模块职责只在 [ARCHITECTURE](ARCHITECTURE.md)维护，已实现文件与能力见 [REBUILD](REBUILD.md)，C ABI 与聚合接口见 [INTERFACES](INTERFACES.md)。prepare、review 包及 compare 已有实现；其增量缓存或求解扩展不能与已存在模块一起写成“待建文件”。

GPU 求解能力需要实际贯通所声明因子子集的残差/导数、H/g 累计、阻尼与局部消元、线性求解、trial 目标、接受/拒绝及完整状态保存。CUDA 投影演示、单位矩阵求解或空 solve 返回成功都不构成该能力；未接入因子与缺口明确报告。此判定不规定开发批次数、镜头先后或新的审批步骤。

## 11. Windows / 5070 Ti 构建与运行

CPU／CUDA／cuDSS 构建命令见 [native](../native/README.md)，连续 LM 与磁盘恢复的能力边界见 [REBUILD](REBUILD.md)。复用已有 Python、CMake、MSVC、CUDA 与匹配的 cuDSS，按实际构建记录版本。

RTX 5070 Ti 的计算能力为 12.0，目标代码须由支持该卡的工具链编译并实际加载，不能用数据中心 Blackwell 的 sm_100 冒充。Windows cuDSS 的运行 DLL 在 bin，lib 保存链接库；依赖解析随所用版本验证，不能靠改 PATH 掩盖版本不匹配。

源码/构建配置变化触发相应增量构建。目标求解缓存按输入和依赖变化失效：参数数值变更不重建拓扑，相关组或活动集改变时更新相应结构/符号分析。当前 prepare 的实际校验与未实现的缓存边界见 REBUILD，不将设计写成既有性能保证。文件位置统一见 [MAINTENANCE](../MAINTENANCE.md)。

<a id="run-contract"></a>
## 12. 运行、暂停、恢复与审核产物

本节的主日志和当前输入要求适用于实际启动、重启与续算；前文的重构设计不表示新求解器已经实现，可调用能力以[实现状态](REBUILD.md)为准。

每次启动或重启求解，立即向用户提供本次主日志的绝对路径。日志逐次记录目标分项、块数、设备时序、域失败、阻尼、是否接受数值步及当前可见产物的位置，不能长时间只有一个“running”。

正常求解与续算不得重放历史数据生产或资格链，不得把旧包、历史源码或历史审计回执作为启动前置条件。数据准备负责交付可独立消费的当前数据；未改变的数据不重复校验来源，变更只检查受影响部分及其必要关联。求解器只检查当前输入、模型与状态的必要一致性，不因重编译或无关源码修改重新追溯来源；不得把每次重放换名为准备或缓存回退。原片、人工确认和历史失败照常保留供追查。

暂停在当前 GPU 操作的安全边界响应，保存最后可用状态。崩溃不覆盖已有 checkpoint；同目录临时文件完成后原子替换。优化候选与用户接受版本分别存储，用户接受只通过明确的审核动作改变。

可见结果及其验证范围按 [VALIDATION](VALIDATION.md#工程验证与交付)登记：模型关联、真实源投影、拟合残差和受影响镜头对照与实际改动对应。具体目标镜头由当前任务及节目清单决定，不固定为历史 S01；编译或合成测试不代替画面验收。

性能测量必须固定源输入、活动参数、目标、精度和结果质量，并分开报告冷启动、暖启动、线性化、累计消元、分解、trial 评估、输出时间与峰值内存。性能检查服务于当前授权任务，不自行扩成持续 benchmark；视频生成条件见 VALIDATION。

## 13. 官方能力依据与仓库证据

以下官方链接用于核对外部接口/支持情况，不证明本项目实现或速度；本机能力仍须由实际构建与运行记录支持：

1. Ceres CUDA / cuDSS 支持与构建： https://ceres-solver.readthedocs.io/latest/installation.html
2. Ceres CostFunction / EvaluationCallback： https://ceres-solver.readthedocs.io/latest/nnls_modeling.html
3. cuDSS 数据类型、分阶段、内存估计、Windows 支持： https://docs.nvidia.com/cuda/cudss/latest/index.html
4. cuDSS 分析/分解/求解与 Windows 运行库： https://docs.nvidia.com/cuda/cudss/latest/getting_started.html
5. RTX 5070 Ti 计算能力： https://developer.nvidia.com/cuda/gpus
6. Blackwell cubin/PTX 兼容： https://docs.nvidia.com/cuda/blackwell-compatibility-guide/
7. 标准LM与Nielsen阻尼更新（第2.4、3.2节）： https://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf 。本文有界步、robust IRLS及默认阈值是项目实现选择，不能当作原文全部算法已适用于本项目的证明。

仓库实现入口为 [core_api.h](../native/include/rinkspace/core_api.h)、[GPU 薄绑定](../src/rinkspace/core/gpu.py)、[场景适配](../src/rinkspace/session/gpu_geometry.py)与[场景关系](../events/ankara_2026/model_constraints/geometry_relations_current.json)。历史 Ceres 对照的范围由其模块说明和冻结运行记录确定。
