# 本片事件适配：把已实现内核接到真实全源问题

本页描述 rinkfit 历史后端的模块合同，适用范围见[模块入口](../README.md)；当前 rebuild 的 native／GPU 能力见[实现状态](../../../docs/REBUILD.md)。

本文是初始架构到实际 `rinkfit_event_fit` 的具体实施接口。数学权威为 `../spec/design_contract.json` 和 `../spec/SPEC.html`；已经可编译运行的 Ceres 框架接口见 `solver_backend.md`。本文保留 2026-10-17 初始首镜头包的技术合同；当前公开源码的输入/设计文件缺口及入口见[模块 README](../README.md)，后续实现状态按实际代码和节目快照核实。

## 1. 数据已经在包中，原始义务固定

`data/observations.npz` 保留五个原始数组，`allow_pickle=False`：

| 字段 | dtype / shape | 作用 |
|---|---|---|
| source_point_ids | `<U69`, (182017,) | 原始取点 ID；跨帧重复，不能单独去重 |
| source_target_labels | `<U50`, (182017,) | 171 个原始材料标签；不可改名合并 |
| native_pts | `<i8`, (182017,) | 同一帧同一 PTS；time base=1/12800 |
| frame_indices | `<i4`, (182017,) | 原镜头帧 1…1248 |
| source_uv | `<f8`, (182017,2) | 1920×1080 原生整数像素中心坐标系中的已审核 UV |

PTS 首项 256256，步长 256，即 50 fps。镜头相对时间原点设于 frame 1；RS 中间行时间等于该帧名义时间是一项时间规范，不是独立测得的传感器时间。旧 20 ms readout 值只作候选初值。

`data/input_manifest.json` 记录逐数组字节哈希。S0 必须按原始顺序保留全部行。不能把多个帧上的同 ID 当作同一观测，也不能用 (frame, material) 代替逐点身份。

`data/seeds.json` 保存 171 原始标签 → 166 旧模型目标的显式映射，所有旧几何/相机都有“条件初值”角色。围挡多标签共用一个连续世界边界，并不删除这些来源标签。`Ad7.print.component000` 有两种原始描述分支，其中包含多条分离闭合路径；必须读取原始关联范围/分支元数据，不能任意取第一条，也不能把路径间隙连接成线段。

## 2. 直接可用的 Ceres 输入 ABI

`launch.py export-backend` 在运行 Data 目录写 `backend_input/manifest.json` 和以下紧凑 little-endian C-order 文件：

| 数组 | 文件 | 类型 / shape |
|---|---|---|
| native_uv | uv.f64 | float64 [N,2] |
| source_rows | row.u64 | uint64 [N]，严格 0…N−1 |
| frame_index | frame.i32 | int32 [N] |
| native_pts | pts.i64 | int64 [N] |
| material_index | material.u32 | uint32 [N] |
| point_id_index | point_id.u32 | uint32 [N] |
| engineering_budget_px | budget.f64 | float64 [N]，初值全为 2 |

`string_tables.json` 保存两张 UTF-8 字符串表。索引只用于运输，完整主键仍为 `(shot_id, native_pts, source_point_id)`。每个文件有 SHA-256、dtype 和 shape；`check_packet()` 验证文件字节。事件 reader 应核对 ABI、字节数、端序、索引范围、完整主键及源哈希，再一次性构造稳定参数/观测存储。

场景、全帧相机、原始标量统计元数据和 media 映射随 packet 复制；这些文件各自有 hash。配准/图目录是可选初值输入，不能进入最终 likelihood。它们缺失时仍可从别的合法初值建立事件。

## 3. 原始统计和语义接入位置

下列路径在审核快照中存在，应首先在**当前本机仓库**定位并读取，避免运行整个历史求解启动链：

| 来源 | 启动包用途 |
|---|---|
| `events/ankara_2026/programmes/okada_mei_sp/source_datasets/s1_ad5_id/binding.json` | 原始数据集绑定与来源引用 |
| `events/ankara_2026/programmes/okada_mei_sp/source_datasets/shot1_fit_20261014/identity_fix2.json` | 场景关联/有限支持覆盖的版本线索；需以当前绑定为准 |
| `calibration/program_tracking/first_shot_fit.py` 中 `_load_frames`、`FrozenWhitening`、`FirstShotSourceBundle` | 原来源数据和 frozen whitening 的结构；仅提取读取层所需内容 |
| `calibration/program_tracking/first_shot_source_evidence.py` 中 `prepare_frame_evidence` | 原生 source/profile/readout 和来源关系 |
| `calibration/program_tracking/source_evidence_whitening.py` 中 `SourceWhiteningOperator`、`joint_line_corner_operator` | 原始分组统计算子和线/角点共同噪声关系 |
| `data/seeds.json` 各 descriptor 的 association/source 路径及哈希 | 保留材料分支的实际证据和候选有限支持 |

`source_sigma.npz` 不包含全镜头 grouped W；审核包中的少量选帧 W 也不等于完整 W。事件适配应用现有读取层导出一个新的不可变 `semantic_packet`：

- 每个原始 row 的 semantic 类型、物理材料 ID、已知实体/未知 station、合法 branch 与有限区段。
- 只有存在来源的 run 才包含顺序/方向；观察数组顺序不提供连接或单调性。
- 真 profile 的查询点、单位法向、有限 alpha 区间及所有共享来源；没有这些元数据的二维 UV 不能被改成 profile。
- 每个原始统计组的原始 row 索引、W 或其等价稀疏分解、共同模式设计/先验、来源版本和冻结哈希。线/角点衍生关系不得重复计量。

若旧读取层依赖 mutable owner，把读取逻辑隔离成纯导出函数，并在全 182017 行上比对五个不可变数组。不要为复用 W 而继承旧几何、旧残差、旧 outlier 或更新白化的行为。完整 W 接通前可继续 S0–S4 的工程可行性工作；S5 不可宣称使用了未接入的完整统计目标。

## 4. 静态场景图和相机状态布局

使用稳定命名参数块。不要把每帧不同的几何数据重新存成独立场景。

| 块 | 内容 / 规范 | 当前原语 |
|---|---|---|
| world_gauge | L=1；世界中心/朝向固定；z=0 冰面 | seeds.json 已按 L 归一化 |
| rink_shape | 宽长比、合法圆角参数、共享中心线/冰面图形尺寸；连续闭合/相切关系 | `RoundedRink` 已支持 p=2 椭圆角；其余候选须显式实现 |
| board_surface | 连续周长展开坐标 s 与高度 h；广告顶/底/角点共享安装变量 | `BoardSurface`、`BoardSplineMaterial` |
| ice_materials | 有限线、圆、弧、封闭边界；共同几何关系与有限可调印刷偏差 | `FiniteLine`、`CircleArc` |
| shared_prints | 材料静态有限路径、闭合拓扑和安装变量；跨弯道表面不平面化 | `BoardSplineMaterial`；分离路径 union 需接入 |
| fixed_head | 共享 R0、世界 pan 轴 n、相机 tilt 轴 b、固定 pivot P、相机杠杆 d、轴距 a | `FixedHead` / C++ `physical.hpp` |
| optics | f_ref、eta、主点及有限 zoom 斜率、Brown k1/k2/p1/p2、入口瞳低维项 | `ZoomOptics` / C++ `physical.hpp` |
| camera_controls | alpha(t)、beta(t)、log f(t) 原生均匀三次样条；alpha/beta 的函数参考值严格消元 | `UniformCubicBasis`、`GaugeFixedSpline` |
| source_stations | 每条普通曲线记录的合法 station；实体点不变成滑动点；确认 run 有顺序约束 | `ordinary_curve_residual`、`verified_order_inequalities` |
| root_witnesses | 显式 RS row 和真实 profile alpha/查询等式；模型状态与分支完整保存 | `explicit_rs_equality`、`profile_auxiliary` |
| budget_sq | Phase I 中 b≥0；Phase II 精确固定为 1 | Ceres/PHR 框架 |

普通曲线的二维误差必须是到**同一有限材料**上的实际预测误差。线端部仍保留切向偏差，不能投到无限延长线上后只剩法向零误差。用真实 station 变量与 bounds/顺序进入 Ceres；独立 replay 再解合法最近点/真实 query 检查所选分支。分支切换需在同一候选状态、同一物理定义下明确恢复，不能改变原始材料身份。

严格固定机构公式由内核给出：`R=Rot(b,beta) R0 Rot(n,alpha)`，`C=P+Rᵀ[d+ell(z)e3]+a h`。`bᵀR0 n=0`、单位轴和 SO(3) 使用流形或严格消元；不靠近似惩罚冒充等式。参考参数变换要同时变换 R0、轴/杠杆或相关规范，保持实际相机不变。

## 5. S1/S2/S3 到物理初始化的实际衔接

S1 当前使用带来源的**静态条件旧材料形状**做初始配准，探索历史投影、真实实体 DLT（有足够支持时）和独立仿射假设。它不能证明旧形状正确。其采样版目标只是 capture；所有候选以同一原生连续有限支持重新比较。

S2 目前自动绑定了真实 `ad_corner` 边。真实数据验证中只有 `board_y_max` 产生此类足秩边，图仍有大量孤立/弱支持帧。这不是“那些帧没有用”：普通冰线、曲线轮廓和三维围挡的跨帧关系还需要事件适配。可使用同一材料的 frame-to-frame 形状匹配或共同 atlas 变量生成独立原始边，保留歧义和实际秩；不使用重复 point ID 假造逐点运动。

图层已经支持多锚和双向传播、联合精化。`*_charts.json` 显式记录每分量 Q：若参考源为同一条件旧平面 H_ref，则 `Q=H_ref`，因为 `H_component=H_old H_ref^-1`；无度量依据的 projective 分量 Q 保持 unknown。不同分量不能直接共用 I。

`upgrade` 命令当前使用同帧、满秩 S1 候选，在明确旧世界图表上条件设置 Q=I 求共同 P；共享 P 仍只是初值。后续需把多平面/公共图集信息回收到共同低维内参、正深度方向和固定云台。无足够多平面时保留退化诊断，可以直接用全时域物理优化从其他合法初值启动。

若用自由相机姿态桥接查找初值，其逐帧自由位移/任意旋转必须在进入 S4 正式物理目标前删除，投影回一套固定机构并原生回放。最终目标不含 H/flow/P 伪观测；共同 atlas 或共享物理印刷形状只能作为一个静态参数实体。

## 6. Ceres 事件因子和完整 `ProblemBuilder`

以 `cpp/src/synthetic_demo.cpp` 的有限线/实体点例子为编译接口样板。创建事件源文件并 link `rinkfit_ceres`，具体虚函数表已在 `solver_backend.md` 完整列出。以下四个接口决定能否实际验收，不能写固定 true 或零值占位：

1. `replay_native(snapshot)`：从传入深拷贝快照重新构建全物理状态，逐原始顺序恢复真实 RS/query/root/finite support，输出全 N 行和具体失败原因；不读取优化中可变缓冲区。
2. `evaluate_constraints(snapshot)`：按固定 ID 顺序返回真实 `g/h`，前 N 个 g 必须是 q_i−b，后续物理/顺序/query 约束使用自己的冻结尺度；不能使用白化后的 AL 残差替代 g/h。
3. `lagrangian_gradient(snapshot, phase, lambda, nu)`：在当前流形和活跃 bounds 的缩放切空间上重算目标+真约束拉格朗日导数。不能以“Ceres 已收敛”、旧 Jacobian、残差梯度或损失数值代替。
4. `capture_state/restore_state`：完整、深拥有、可原样恢复所有命名变量、station、分支和 root chart；另保存固定的来源/模型/预算/W/尺度/目标指纹。对同一状态的 replay token 必须一致。

PHR 内循环每个目标冻结乘子/rho/统计/材料状态。预算 AL 因子与硬域因子不得加 robust loss；无根/负深度/畸变非法等定义域错误不得由 inactive constraint 分支吞掉。具体一般约束/箱式 bound 的职责见求解器文档。

## 7. RS 和 profile 的求解/验收分离

内核已经给保护区间根搜索、隐式导数、无根/多根/奇异/未认证状态。完整事件 C++ 包装需要将这些语义接到 Jet/解析 Jacobian：`dUV/dtheta = pi_theta + pi_t dt/dtheta`，分母含真正的行时间反馈，不能固定观测行替代真实根。

为可恢复的错误初值，可以用显式行/查询 witness 和等式一起优化，等式尺度与像素 b 分离。最终 replay 必须回到真实根，检查 witness 与原生根的一致性。有限粗采样发现一个根不证明全区间唯一；实施解析或区间速度/分母界，或其他有完整区间依据的根隔离。内核 `uniqueness_uncertified` 状态是明确未认证，不能直接改成 valid。

首尾时域至少覆盖半次最大读出；不能靠截断时间 clamp 掩盖缺失轨迹。对于其他低维光学/输出候选（主点随 zoom、径向随 zoom、非正交轴、输出裁切/去隔行等），按 SPEC 的证据顺序显式增补模型，保留各候选的可辨识性和同支持对比；不同时放开一堆互相抵消的自由流。

## 8. 二进制协议和最终文件

事件二进制名 `rinkfit_event_fit`。`--capabilities-json` 的 stdout 只输出一个 JSON：

```json
{"schema":"rinkfit.event_backend.v1","backend_kind":"physical_event","shot_id":"ankara_2026/okada_mei_sp/shot1","solver":"Ceres","whole_shot_rows":182017,"independent_native_replay":true,"fixed_shared_camera_head":true,"synthetic_demo":false}
```

只在实际实现这些能力后设置字段。主程序参数为 `--packet <manifest.json> --run-dir <Data目录> [--resume <完整checkpoint>]`。进度输出应含明确阶段/误差/约束/停止原因；长期运行不能静默。

S6 至少输出 `final_state`、完整 checkpoint、`final_predictions.npz`、`final_source_residuals.npz`、`verification.json`、按帧/材料汇总。预测 NPZ 字段为 `source_rows/native_pts/source_point_ids/source_target_labels/predicted_native_uv`，逐原始顺序排列。另保存合法 station/branch、root状态、物理域与独立 evaluator 版本。

`launch.py verify-native` 可以独立核对来源身份和普通二维原生像素算术。它故意不能单独接受物理结果；真实 profile、共享几何/固定机构、根和 KKT 等验收还要由独立事件 replay/S6 完成。必须在实际截图/关键帧叠加中检查同一个最终状态；部分平面 0.3 px、合成例 0.1 px 或程序 return code=0 都不代表本片成功。

## 工程预算与统计目标

初始首镜头配置的 E=4I px² 对应 2 px 工程半径，与 sigma/分组 W 的统计含义不同。普通二维 q=eᵀE⁻¹e，b=t²；Phase I 最小化 b 且 q_i≤b，Phase II 固定 b=1 后优化原始统计目标及有依据的项。真实 profile 使用其法向预算和精确 query 等式，有限查询区间不随 b 放宽。该初始配置不是其他赛事的默认精度或长期项目规则。
