# 配准与关系图接口（S1 / S2 启动实现）

本页描述 rinkfit 历史后端的模块合同，适用范围见[模块入口](../README.md)；当前 rebuild 的 native／GPU 能力见[实现状态](../../../docs/REBUILD.md)。

本模块实际执行有限材料平面配准和关系图联合精化，使用 NumPy / SciPy 双精度计算。它是 v1.1 的启动层和局部比较器：不包含最终 Brown / 隐式行时间相机，不执行物理 Phase I / II，不输出全镜头 `t_hat`，也不把局部 H 的损失称为物理模型下界。

`src/rinkfit_start/registration.py` 和 `graph.py` 不读取或修改主 NPZ。调用者向局部比较器提供一个真实平面的显式观测子集；完整原始义务继续留在主账本里。`RegistrationGraph` 接受全部帧 ID，孤立帧也保留。

## 1. 已实现的运行路径

1. 用真实实体对应执行 Hartley 归一化 DLT，检查设计矩阵至少 8 秩；或者接受调用者有来源的 H 初值。
2. 用四个虚拟控制点构造八参数局部 H 图表。这四点只是参数坐标，不增加观测。采用 Frobenius 归一化，允许 `h33=0`。
3. 对同一材料的有限支持运行归一化正权 soft-min。合法站位、分支、正权重在一次 base/trial 比较中保持不变，投影坐标每次重新计算。固定实体点仍只投影自己的固定站位。
4. 默认执行 `1/8, 1/4, 1/2, 1` 倍金字塔，对应原生 `64, 32, 8, 2 px` 的工程捕获核；每次换尺度/重采样都重建目标和 Jacobian。
5. 检查连续曲线的采样填充距离。基态采用 `sigma_native / 8` 的保守上界，trial 必须不超过 `sigma_native / 4`；超限时拒绝该 trial，在外层允许有限次重建。超出固定采样预算会明确记录 `sampling_limited`。
6. 切换到原生连续有限材料残差，使用有限步数的 8 维阻尼 LM。所有步接受都比较同一目标的预测/实际下降；不合法分母、镜像分支改变和采样不足的 trial 被拒绝。
7. 保留每个已接受的完整局部状态，按相同支持上的原生最大距离、再按 RMS 排序；最后计算实际原生残差 Jacobian 的秩和弱方向。
8. 用有来源的原始两帧边建立全帧关系图，从多个锚点正向和反向传播，分别固定各真实连通分量的图集规范。联合稀疏 LM 同时更新该图内所有可初始化 H，保留未定帧。

## 2. 方向、坐标与图表

| API / 字段 | 方向或单位 |
|---|---|
| `H_atlas_to_native` | 一个真实平面的图集坐标 → 该帧原生整数像素中心 |
| `G_i_to_j` | 帧 i 的原生像素 → 帧 j 的原生像素 |
| `pixel_center_transform(s)` | `u_s = s * (u + 0.5) - 0.5`，x/y 同比例 |
| `scale_atlas_to_native(H, s)` | `T_s @ H`，图集不缩放 |
| `scale_frame_i_to_j(G, si, sj)` | `T_sj @ G @ inverse(T_si)` |
| `LocalHomographyChart.matrix(p)` | 8 个无量纲控制点位移 → H |
| `LocalHomographyChart.parameters(H)` | H → 同一个局部图表的 8 个参数 |

`project_points(H, xy)` 只在被调用的支持点上检查分母，不把画外投影删掉。连续材料还检查整个**调用者声明的有限分支**是否穿过 H 的地平线。图表失效时抛出明确错误；当前版本不自动切换到另一组虚拟控制点，调用者可以显式给 `atlas_controls` 后重新建立等价候选。

## 3. 有限材料与观测

```python
from rinkfit_start.registration import MaterialSupport, RegistrationObservations

line = MaterialSupport.segment(
    "ice.line", [0.0, 0.0], [1.0, 0.0],
    provenance="经确认的共享平面线段模板及版本",
)
arc = MaterialSupport.arc(
    "ice.circle", center=[0.5, 0.3], radius=0.1,
    theta0=0.0, theta1=6.283185307179586,
    provenance="共享圆几何候选及来源",
)
point = MaterialSupport.point(
    "ad.corner", [0.2, 0.4], provenance="同一个实体角点的共享坐标候选",
)

observations = RegistrationObservations(
    native_uv=uv,                       # N x 2，原生值的不可变副本
    material_ids=material_labels,       # N，按材料身份分通道
    record_keys=composite_record_keys,  # N，完整 shot/PTS/source ID；必须唯一
)
```

支持 `point`、不连接的 `points`、有限 `segment`、有明确连接依据的 `polyline`、`arc`、`cubic_bezier`。所有连续类型使用归一化站位 `[0,1]`，并可指定 `station_interval=(a,b)` 和 `branch_id`。`polyline` 必须提供 `connectivity_evidence`；无顺序元数据时使用 `points`，不会按数组次序推断折线。

普通曲线观测不提供固定站位。真正物理角点/端点可以使用独立 `point` 材料，或者给 `RegistrationObservations` 传入 `fixed_stations`（未知位置为 `NaN`）和逐条 `fixed_station_evidence`。没有实体依据的有限 `fixed_stations` 会报错。裁剪/遮挡 run 的端点不自动成为固定实体点，重复 `source_point_id` 也不产生共享站位。

### 原生连续最近点

```python
predicted_uv, stations, at_boundary, ambiguous = nearest_on_material(
    H_atlas_to_native, native_uv, material,
)
replay = native_replay(H_atlas_to_native, materials, observations)
```

线段/折线计算投影后的有限线段最近点，并将投影分数精确换回 projective 站位。圆弧和三次 Bézier 把距离驻点化为有限次数多项式，计算所有数值实根和两个合法端点，然后选取最小距离。因此它们求连续材料的双精度数值最近点，不是稀疏样本、栅格 EDT 或无限直线距离。驻点多项式根接近重根时仍有数值风险；`ambiguous_minimum` 和边界标记要保留。

`NativeReplay` 提供 `predicted_native_uv`、`error_native_uv`、`stations`、`at_support_boundary`、`ambiguous_minimum`、`max_error_native_px`、`rms_error_native_px`。距离为原生像素，只对传入的完整局部子集有效。

### 完整轮廓跨过 H 地平线的限制

旧 GS 种子可能让完整 `board.contact` / `board.top` 轮廓跨越地平线，即使本帧原生点只来自合法的前方区段。当前接口不会自行猜测哪个区段可见、删除远处样本或按误差截断材料。

若已有独立支持的合法有限分支，可通过 `station_interval` 和带来源的 `branch_id` 明确传入。否则异常信息会写明 **“Initialization support-domain limitation … not an observation-error finding”**。应保存失败初值和完整源义务，再尝试其他有来源种子（包括独立仿射捕获种子），或交给完整 3D/物理初始化处理。这不是审核 UV 有误的结论。

## 4. 最小真实数据调用

以下示例直接使用包内的完整输入加载器和有来源的条件几何种子。选择第一帧真实 `board_y_max` 材料，创建一个局部比较器；其他平面和所有其他帧仍在 `source` 原始对象中。旧静态模板与旧相机 H 只提供启动候选，不声明其几何准确或通过物理验收。

```python
from pathlib import Path
import numpy as np
from rinkfit_start.data import Observations, SeedCatalog
from rinkfit_start.registration import (
    MaterialSupport, RegistrationObservations, RegistrationOptions, fit_registration,
)

source = Observations.load(Path("data"))
seeds = SeedCatalog(Path("data"))
frame, plane = 1, "board_y_max"
frame_indices = source.indices_for_frame(frame)
selected = np.array([
    i for i in frame_indices
    if seeds.plane_for_material(source.source_target_labels[i]) == plane
], dtype=np.int64)

materials = []
for label in np.unique(source.source_target_labels[selected]):
    variant_ids = seeds.materials[label]["variants"]
    if len(variant_ids) != 1:
        raise ValueError("需要显式的分支候选；不能猜测或平均模板")
    variant = seeds.variants[variant_ids[0]]
    xy = seeds.plane_coordinates(plane, seeds.world[variant["world_array"]])
    provenance = f"seeds.json:{variant_ids[0]}; legacy_static_seed_only_not_world_truth"
    if variant["geometry"]["kind"] == "point":
        material = MaterialSupport.point(label, xy[0], provenance=provenance)
    elif variant["geometry"]["kind"] == "polyline":
        material = MaterialSupport.polyline(
            label, xy, provenance=provenance,
            connectivity_evidence=variant["connectivity_provenance"],
        )
    else:
        raise ValueError("必须按模板的真实有限几何显式构造材料")
    materials.append(material)

local = RegistrationObservations(
    source.source_uv[selected], source.source_target_labels[selected],
    source.record_keys(selected),
    metadata={"source_input": "observations.npz", "scope": "frame 1 board_y_max only"},
)
result = fit_registration(
    materials, local, seeds.historical_H(plane, frame),
    frame_id=frame, plane_id=plane,
    initial_provenance="historical GS/no-distortion seed; conditional initialization only",
    options=RegistrationOptions(
        capture_iterations_per_stage=1, native_iterations=2,
        max_sampling_rebuilds_per_stage=0,
    ),
)
print(result.status, result.best_native.replay.max_error_native_px,
      result.best_native.rank.rank)
candidate = result.best_native.to_dict()  # 包含原生子账本、几何、S、采样、目标版本和来源
```

此短调用是可运行启动诊断，不是生产收敛设置。完整原始账本必须继续包含全部 182017 条记录和 1248 帧。此结果不能输出全源 `t_hat`。

本包制作时已实际运行该示例：第一帧 `board_y_max` 的 81 条原生观测、10 个材料完成调用，完整输入对象仍保留 182017 条记录；初始局部最大距离 `18.4486335345 px`，短调用后的最佳局部最大距离 `17.4139586151 px`、RMS `8.0390217588 px`，实际 Jacobian 秩为 `8/8`，状态为 `iteration_limit`。这说明入口可运行，也直接表明该短诊断远未达到 2 px 工程目标；秩满不等于像素拟合通过。

## 5. `RegistrationOptions` 最终参数表

| 参数 | 默认值 | 用途 |
|---|---|---|
| `pyramid_scales` | `(.125, .25, .5, 1.)` | 等比例像素中心金字塔；非空调度必须以 1 结束 |
| `sigma_native_px` | `(64., 32., 8., 2.)` | 与每层对应的原生捕获核尺度；不是像素预算或测量 sigma |
| `capture_iterations_per_stage` | `12` | 每层每次固定采样模型的 LM 步数上限 |
| `native_iterations` | `25` | 连续原生有限材料精化步数上限 |
| `max_samples_per_material` | `8193` | 单材料正求积采样上限；不足明确记录 |
| `max_sampling_rebuilds_per_stage` | `2` | 当前采样域限制后允许的外层重新采样次数 |
| `lm_initial_damping` | `1e-3` | 小规模阻尼 LM 的初始数值阻尼 |
| `max_parameter_step` | `.15` | 每次 8 维归一化虚拟点位移的范数上限 |
| `scaled_gradient_tolerance` | `1e-6` | 数值缩放后的梯度停止阈值 |
| `relative_cost_tolerance` | `1e-9` | 同目标实际下降的有限步停止阈值 |
| `rank_relative_threshold` | `1e-5` | 实际原生 Jacobian 奇异值相对秩阈值 |
| `engineering_residual_scale_native_px` | `2.` | 原生精化/秩报告的固定工程单位；不称测量 sigma |
| `objective_version` | `finite_same_material_softmin_v1.1/native_continuous_v1` | 写入每个候选快照的目标版本 |

`pyramid_scales=(), sigma_native_px=()` 可以显式跳过捕获，仍执行连续原生精化。`capture_iterations_per_stage=0` / `native_iterations=0` 仅生成并回放该有限诊断状态，不能作为拟合完成标志。

### 状态与不可变候选

`RegistrationResult` 包含 `best_native`、`final`、`snapshots`、`optimization_log`、`status`、`limitations`。每个候选保留原始 UV 和复合键、共享材料几何和来源、H、显式合法站位、边界/歧义状态、原生残差、捕获核及固定求积采样、目标版本。数组由独立 bytes 支持，不能重新开启写标记；嵌套元数据被深冻结。

`rank_diagnostics()` 对实际连续有限材料残差重新计算归一化 Jacobian，排除先验和 LM 阻尼；结果包括全部奇异值、弱方向以及边界/分支是否影响可微性。一个未知站位圆的合成测试显示秩至多 5，代码不会因为重复取很多点就宣布 H 唯一。

这些是**完整局部比较器状态**。完整物理候选还必须由调用方把共享场景、完整 S/分支、相机、光学、轨迹、目标版本和全帧账本一起不可变保存。不能把多个局部优胜帧拼成一个声称已验收的物理状态。

## 6. 原始关系图最小调用

```python
from rinkfit_start.graph import PairwiseEdge, GraphAnchor, RegistrationGraph

# uv_i / uv_j 至少含四个、且足秩的真实同一实体对应。
# 不能用两帧第 k 个曲线取点或重复 source ID 代替实体依据。
edge = PairwiseEdge.from_entity_correspondences(
    i, j, uv_i, uv_j, plane_id="board_y_max",
    entity_evidence="真实 ad_corner 物理身份逐点核对来源",
    provenance="直接从两帧原始角点估计，未由已有 H 合成",
    source_record_keys=keys_i, target_record_keys=keys_j,
)
graph = RegistrationGraph(range(1, 1249), "board_y_max", [edge])
initialized = graph.initialize()  # 每个真实连通分量建立自己的 projective 规范
refined = graph.refine(initialized, max_iterations=10)
full_frame_ledger = refined.frame_ledger
cycles = graph.fundamental_cycles(max_cycles=128)
```

已有共同模板下的多个锚点时：

```python
anchors = [
    GraphAnchor(frame_a, H_a, "shared_template_v1", "独立锚点 A 的来源"),
    GraphAnchor(frame_b, H_b, "shared_template_v1", "独立锚点 B 的来源"),
]
initialized = graph.initialize(anchors, gauge_mode="external_fixed_atlas")
```

同一连通分量中的锚点必须明确属于同一个 `atlas_id`。默认 `component_projective` 模式以覆盖较好的参考锚点作统一右坐标变换，使 `H_ref=I`；其余锚点保留并分别沿实际边正反传播。`external_fixed_atlas` 保持已声明的外部共同模板坐标，并固定一个参考锚点的 H 作为图优化规范。可通过 `frame_information_scores` 输入有依据的锚点信息评分；没有额外评分时使用实际边见证点的二维覆盖和数量，时间顺序只作相同评分的确定性决胜。

直接构造 `PairwiseEdge(...)` 时必须显式设置 `original_evidence=True` 才能进入传播/精化/循环；默认 `False`。`derived_from_H=True` 不能同时声称原始独立边。仅由 H 合成的边不会建立连通性，也不会证明零循环误差。

`check_cycle([(edge_id, forward_bool), ...])` 要求至少三个不重复原始边形成闭合有向路径，返回原生见证点的位移、最大值和 RMS。它只检查实际原始两帧估计的一致性；共享像素带来的相关性仍保留，不称独立真值验证。

### 图联合精化的实际范围

联合目标比较真实支持区域内的原生见证点。若明确提供真实实体目标像素，使用其原生值；否则使用独立成对 G 的投影作为启动目标。残差按每边冻结的正工程权和点数归一化。稀疏 Jacobian 的每条边只连接它的两个 H，LM 通过 SciPy 稀疏线性系统同时求解所有活动 H。每个连通分量固定一个规范，不跨无证据区间填边。

`GraphRefinement` 返回初始/最终原生见证 RMS、完整帧状态、原始初始化快照、优化记录和秩状态。变量数不超过 128 时计算完整实际残差 Jacobian 的秩及弱方向；更大的图明确写 `actual_residual_rank=None`，不会用局部边秩或不完整稀疏谱假冒整图满秩证明。

**所有成对 H / flow / pose 图因子都仅用于启动与诊断；即使来自原始两帧估计，也必须从最终全源物理目标删除。**

## 7. 明确未实现的部分

- 自动图像特征、视频解码、网络训练、外部代码/权重、跨帧实体身份发现。
- 自动可见分支推断、同一材料多个自由候选分支的联合选择、verified run-order 耦合、profile query 等式和反向覆盖因子。具有这些语义的源记录需要物理层对应因子，不能把本局部比较器当作其验收结果。
- 将 projective 图集自动升级为共同欧氏世界、联合恢复所有共享模板自由变量、固定云台物理接收。图模块当前优化共同 H，模板形状作为显式静态启动候选固定。
- 对整个生产镜头的计算性能/数值收敛保证，以及完整大图秩证书。

## 8. 已运行的针对性验证

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_registration.py -v
```

11 项测试覆盖：像素中心和双向尺度变换；实体 DLT/共线退化/`h33=0`；有限端点反例；未知连接和重复 source ID 语义；soft-min 正偏差/归一化/尺度律；圆弧和 Bézier 连续最近点；单圆弱方向；有限求积填充和地平线限制；已知 H 的短多尺度恢复与不可变快照；完整 1248 帧多锚正反传播；真实循环位移/合成循环拒绝；有偏初值的联合稀疏图精化。这些验证不构成真实镜头物理拟合完成声明。
