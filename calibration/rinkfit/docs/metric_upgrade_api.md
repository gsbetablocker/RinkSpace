# S3：显式度量图表到同一相机的初值原语

本页描述 rinkfit 历史后端的模块合同，适用范围见[模块入口](../README.md)；当前 rebuild 的 native／GPU 能力见[实现状态](../../../docs/REBUILD.md)。

`src/rinkfit_start/metric_upgrade.py` 实现一个可运行的 S3 初值步骤：给定**同一原生帧**中几个真正平面的图集 homography、显式度量升级 `Q_j` 和共享世界中的物理嵌入 `M_j`，联立恢复一个共同的 `3×4` 投影矩阵 `P`。

该原语对应 v1.1 方案第 8.6 节和 `design_contract.json` 的 `registration_frontend.metric_upgrade`。它不估计未知图集的度量，不分解自由的物理 `4×3` 块，不跨焦段变化的帧叠加固定 `K`，也不生成最终物理约束。输出只供后续共同内参、正深度和固定云台初值回收使用。

## 输入关系

统一采用：

\[
q=Q_j\xi,\quad X_h=M_j\xi,\quad
H_j^{atlas}Q_j\sim PM_j.
\]

其中 `xi=(xi1,xi2,1)` 是调用者明确给出的物理平面图表，`q` 是 atlas 坐标；`H_atlas_to_native` 将 atlas 坐标投到原生图像；`M` 在同一个固定世界规范中嵌入物理平面。`M` 的最后一行为 `[0,0,1]`，不允许用一个自由投影 `4×3` 矩阵假扮欧氏嵌入。不同平面中的原点、方向、尺度与安装关系均由输入图表负责，不能由独立 atlas 规范凭空推导。

```python
from rinkfit_start.metric_upgrade import (
    PlaneHomographyInput, common_projection_from_planes,
)

inputs = [
    PlaneHomographyInput(
        plane_id="ice", frame_key="shot_1/pts_256256",
        M=M_ice, H_atlas_to_native=H_ice,
        Q_metric_to_atlas=Q_ice,
        metric_chart_provenance="explicit source/derivation for this Q and physical chart",
        genuine_plane=True,
        metric_chart_kind="conditional_world_seed",
    ),
    PlaneHomographyInput(
        plane_id="straight_board_face", frame_key="shot_1/pts_256256",
        M=M_board, H_atlas_to_native=H_board,
        Q_metric_to_atlas=Q_board,
        metric_chart_provenance="explicit source/derivation for the same-world board chart",
        genuine_plane=True,
        metric_chart_kind="conditional_world_seed",
    ),
]
result = common_projection_from_planes(inputs)
if result.ready:
    P_initial = result.require_projection()
```

示例中的矩阵必须由调用方填入；模块不会发明缺失矩阵。`Q_metric_to_atlas=None` 或缺少图表来源会返回 `missing_metric_upgrade`。使用旧世界种子且条件设 `Q=I` 时，必须写为 `conditional_world_seed`；这只能产生条件度量初值，不能自证旧世界真实。只有独立提供并有来源的物理图表，才可标记 `independent_metric_chart`。代码不会仅凭 `Q=I` 推断是哪种情况。

输入还要声明 `genuine_plane=True`。曲面围挡、任意图像 patch、未知三维广告面不能为了凑平面数量改成真正平面。所有 `frame_key` 必须一致，平面 ID 不得重复。模块检查 `M,H,Q` 的数值秩；单平面即使 `H,Q` 完全精确，也返回退化。

## 联立 DLT 与规范

每个平面产生九个齐次方程：

\[
\sum_{l=0}^{3}P_{rl}M_{j,lc}
-\lambda_j(H_jQ_j)_{rc}=0.
\]

未知量是 `P` 的 12 个行优先系数和每个平面一个比例 `lambda_j`。`m` 个平面形成 `9m × (12+m)` 设计矩阵。每平面的 `M` 和 `H Q` 先以 Frobenius 范数缩放，再做行范数缩放和列均衡，所有数值缩放均记录在结果中；这些是数值处理，不是额外量测或正则项。

完整 SVD 在欠定情形保留全部零空间方向。恰好确定的无噪声共同 `P` 需要除总比例外没有其他零空间，即设计秩 `12+m-1`。有噪声时齐次方程通常不再有精确非零解，数值秩可以是 `12+m`；最小奇异向量仅作为最小二乘初值，必须同时查看代数残差、谱间隙和逐平面重放。

`P` 输出为 Frobenius 范数 1；绝对值最大元素取正，仅用于确定项目比例的存储符号。**这个符号规则不是正深度检查**，没有恢复世界镜像/相机方向结论。

## 状态和诊断

| 状态 | 输出和解释 |
|---|---|
| `initializer_ready` | 数值条件通过的共同 `P` 初值；仍须读取 `metric_status` |
| `weak_initializer` | 保留候选 `P` 和弱方向诊断，但 `require_projection()` 拒绝把它当作已通过的初值 |
| `degenerate` | 保留谱/秩诊断，`P=None`；包括单平面及只有重复几何支撑的情况 |
| `missing_metric_upgrade` | 缺少明确 `Q` 或来源，`P=None`；没有自动 `Q=I` |
| `invalid_input` | 跨帧、不是真平面、奇异输入、非法物理嵌入或数值分解失败 |

主要诊断包括：

- `normalized_singular_values`：归一化设计的完整谱，欠定情形显式补零。
- `numerical_rank`、`numerical_nullity` 和 `expected_projective_rank`。
- `nullspace_gap`：倒数第二与最小奇异值之比。
- `identifiable_subspace_condition`：最大与倒数第二奇异值之比，保留其余方向的弱性。
- `normalized_algebraic_residual`：最小方向在归一化设计中的残差。
- `plane_replay`：在原始 `M`、`H Q` 上重新求每平面比例并报告相对矩阵误差，逐条保留图表类型和来源。

默认 `rank_relative_tolerance=1e-10`、`minimum_nullspace_gap=10`、`maximum_identifiable_condition=1e8`，是需要冻结的工程数值设置。任何一种弱性条件失败均返回 `weak_initializer`，而不是使用 ridge 或静默固定自由方向。

这些谱量来自当前初值的线性设计，**不是最终原生观测似然的信息矩阵**，也不是内参、云台或尺度已可辨识的证书。相对 homography 矩阵误差亦不能替代原生有限材料像素误差；所有阶段仍须回到相同支持上的原生比较，再回到全部原始义务。

## 同后续物理流程的边界

本原语得到一个共同 projective camera 初值。后续 Codex 实现需要在统一输出/畸变/RS 假设下恢复共同低维内参和 `R,C`，检查正确方向及全部真实支持的正深度，再把一组帧初值回收到固定云台。最终物理求解删除这些派生 `H/P` 伪观测，重新使用完整原始材料观测。

RS 和畸变存在时，原始图像 `H` 本身只是已经声明的近似，不能因为这个 DLT 的残差小就升级为严格物理解释。`initializer_only=True` 与 `final_physical_camera_recovered=False` 始终保留在输出中。

## 运行验证

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_metric_upgrade.py -v
```

仅提供两项有针对性的验证：一个由同一已知 `P` 生成的多平面案例，含两个不同显式 `Q`，核验恢复矩阵及新三维点的投影；一个单平面退化案例，同时检查未知 `Q` 不会变成单位矩阵。没有把这些合成检查称为真实场景拟合结果。

## 已实现与未实现能力

**已实现：** 同帧真平面合同、显式 `M/Q` 来源、共同 `P` 与每平面比例的联立 DLT、明确数值规范、完整谱/秩/零空间及弱解状态、条件度量初值标签、原始矩阵上的逐平面重放、两项合成测试。

**未实现：** 未知 `Q` 的度量估计、相机 `K/R/C` 的 RQ 分解与正深度方向认证、完整畸变/RS 等价映射、固定云台回收、原生像素最终因子、真实镜头拟合或最终验收。缺失图表明确返回未升级，单平面明确返回退化，不作静默升级。
