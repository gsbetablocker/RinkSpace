# 物理、时间与有限材料 API

本模块实现 v1.1 方案第 3–6 节及第 11 节的可运行参考内核。依据是本启动包所保留的 `spec_body.html` 与 `design_contract.json`；本文描述代码能力，不能替代设计合同。这里的单元测试是合成数学反例与导数核验，**没有执行或宣称完成 1248 帧真实数据拟合**。

## 1. 文件和职责

| 文件 | 职责 |
|---|---|
| `src/rinkfit_start/camera.py` | Python float64 固定云台、低维光学、原生时间样条、严格前向状态、RS 求根与隐式 Jacobian |
| `src/rinkfit_start/geometry.py` | 有限材料、连续闭合边界、展开围挡坐标、共享广告形状、三类观测接口 |
| `cpp/include/rinkfit/physical.hpp` | 可由 `double` 或 Ceres `Jet` 实例化的等式核心，无直接 Ceres 依赖 |
| `tests/test_physics.py` | 物理、有限支撑、RS 和导数的反例测试 |
| `cpp/tests/test_physical.cpp` | 标准 C++17 单独编译的核心验证 |

Python 对输入形状、有限数值、正长度/焦距、旋转与轴约束进行显式校验。失败投影的 `uv` 为 `None`；C++ 失败像素为 NaN，并带 `DomainStatus`。调用者必须使用 `valid`/`valid()` 或 `require_uv()`，不能直接把无效投影写成零残差。有限像素预算违例仍是可计算的约束，不属于算术域失败。

## 2. 坐标、单位与固定机构

世界右手系，冰面 `Z=0`；默认冰场原点名为 `rink_center`，`+X` 为已确认的长边方向。内部单目尺度使用 `length=1`，所有 `P,d,a,ell` 与材料尺寸使用同一尺度。代码接受正长度数值，**不会把 60 米、标准圆角、操作者身高或旧解参数自动设为事实**。米制尺度必须由外部证据单独声明。

`R` 为世界到相机旋转。相机轴为 x 向右、y 向下、z 向前；像素为原生整数中心，角度为弧度，时间为秒。

```python
import numpy as np
from rinkfit_start.camera import FixedHead

head = FixedHead(
    R0=np.eye(3), n=[0, 1, 0], b=[1, 0, 0],
    P=[0, 0, -3], d=[0.03, -0.02, 0.04],
    a=0.01, pupil_linear=0.02,
)
pose = head.pose(alpha=0.03, beta=-0.02, z=0.1)
```

这个例子仅示范坐标和接口，不是本镜头的物理安装种子。

实际实现严格为：

\[
R=\exp(\beta[b]_\times)R_0\exp(\alpha[n]_\times),\quad
h=\frac{n\times R^Tb}{\|n\times R^Tb\|},
\]

\[
C=P+R^T[d+\ell(z)e_z]+ah,\qquad
\ell(z)=\ell_1z+\ell_2z^2.
\]

`P,n,b,d,a` 是整段共享量，没有逐帧成员。默认 `axis_model="orthogonal"`；验证 `||n||=||b||=1`、`b.T @ R0 @ n=0`。入口瞳基底没有常数项，故 `ell(0)=0` 严格成立，不与 `d_z` 重复。二次项默认为零，若释放必须作为已声明的有限候选。

`pose_and_velocity(alpha,beta,z,alpha_dot,beta_dot,z_dot)` 同时返回 `(Pose,R_dot,C_dot)`。速度来自同一机构的解析导数；没有独立逐帧速度。非正交云台、任意 `delta_xi`、OIS 流和独立 focus 流均未实现，不能添加到最终状态后假装已经接入。

## 3. 光学和真实定义域

`ZoomOptics` 的主要字段：

| 字段 | 含义 |
|---|---|
| `f_ref` | 固定的参考焦距；选定后不优化 |
| `principal_point=(cx0,cy0)` | 默认 `(959.5,539.5)`；这是初始参数，不是机型量测 |
| `eta=0` | 共享纵横比候选，`fx=f exp(eta/2)`、`fy=f exp(-eta/2)` |
| `principal_linear=(cx1,cy1)` | `c(z)=c0+c1 z`，默认斜率零 |
| `k1=(k10,k11), k2=(k20,k21)` | 共享的常数/线性焦段 Brown 径向系数 |
| `p1,p2` | 整段共享切向系数 |
| `radius_max` | 预先声明的归一化射线有效圆盘半径，默认工程值 2 |

`at(log_f)` 返回 `(Intrinsics,Brown,z)`，其中 `z=log_f-log(f_ref)`。`project(X,pose,K,D)` 使用 Brown 的 `k1,k2,p1,p2` 完整公式。没有 skew、额外输出仿射或逐图像变换。

每次投影分别检查有限坐标、严格正深度、射线圆盘、有限除法与有限像素。`Brown.no_fold_lower_bound()` 在当前焦段上精确求两个径向二次多项式的区间最小值，再减去 `6*sqrt(p1²+p2²)*radius_max`。正值是整圆盘无折叠的保守充分认证；非正值返回 `DISTORTION_UNCERTIFIED`，**不宣称已经证明模型无效**。本版本不自动用更强认证算法替代它，也不把离散焦段检查说成连续焦段认证。

`PhysicalCamera.time_derivative(X,t)` 解析传播 `R,C,f,c,k1,k2` 的时间导数；测试用独立中心差分检查。它是固定世界点的偏导。若世界点随曲线站位变化，固定时间的 `pi_s` 必须由相同曲线与投影另行传播，再进入 RS 隐式校正。

## 4. 原生时间三次 B 样条与严格角度规范

```python
from rinkfit_start.camera import (
    UniformCubicBasis, UniformCubicSpline, GaugeFixedSpline, CameraTrajectory,
)

basis = UniformCubicBasis.for_native_frames(
    n_frames=1248, fps=50.0, first_frame_time=0.0, max_abs_readout=0.02,
)
alpha = GaugeFixedSpline(basis, np.zeros(basis.n_controls - 1), reference_time=0.0)
beta = GaugeFixedSpline(basis, np.zeros(basis.n_controls - 1), reference_time=0.0)
log_f = UniformCubicSpline(basis, np.full(basis.n_controls, np.log(900.0)))
trajectory = CameraTrajectory(alpha, beta, log_f)
```

20 ms 在这里是必须先冻结的工作范围示例，不能称为厂家测得的读出时间。真实 PTS 的均匀间隔、时基、首帧原点和同镜头身份由输入层核验；本构造器接收已经核验的原生帧数与 fps。

控制位置为 `first_control_time+j*dt`。每个时间单元仅评价四个相邻控制值，支持 0、1、2、3 阶导数。合法时间闭区间为第 1 至第 `N-2` 个控制位置；右端点使用最后一个合法单元的 `u=1`。边界控制由允许的最大绝对读出时间分配，覆盖第一帧之前和末帧之后的半读出。`require_coverage()` 会拒绝不足的时间域。行时间不得 clamp，轨迹不得跨切镜头拼接。

`GaugeFixedSpline` 存储 `N-1` 个自由控制量。在参考时刻选择绝对基底权重最大的控制量 `c_d`，严格代入：

\[
c_d=-\frac{\sum_{j\ne d}B_j(t_*)c_j}{B_d(t_*)}.
\]

因此零约束作用于 `alpha(t_ref)`、`beta(t_ref)` 的函数值。被消去的控制量通常不为零。`jacobian_free(t,derivative)` 包含消元后的完整依赖，不能用原始四点行直接替代。数值计算仍有双精度舍入；这里“严格”指代数消元，绝非大罚项。

## 5. RS 根、恢复与隐式 Jacobian

`row_time(t_frame,tau,v,H)` 使用有符号 `tau`：

\[
t(v)=t_{frame}+\tau\left(\frac{v}{H-1}-\frac12\right).
\]

不再另外释放读出符号、时钟平移或逐帧速度。`rolling_shutter_project(project_at_time,project_time_derivative,...)` 求解有限原生行区间 `[0,H-1]` 中的 `h(v)=v-pi_y(X,t(v))=0`，采用有区间保护的 Brent 方法。`PhysicalCamera.project_rs()` 先检查整个所需时间范围的控制覆盖，再调用同一求根器。

### 状态与可接受性

| 状态 | 含义与后续动作 |
|---|---|
| `VALID` | 唯一性有完整区间依据，根等式与非奇异性检查通过，像素在声明图幅内 |
| `NO_ROOT` | 未发现合法行根；查看 `search_complete_certified` 区分有认证的无根和有限搜索结果 |
| `MULTIPLE_ROOTS` | 搜到多个相关根，保留根列表；不能任取一个作为最终物理预测 |
| `SINGULAR_ROOT` | 根存在，但 `I-pi_t w^T` 奇异或超过条件数门限 |
| `UNIQUENESS_UNCERTIFIED` | 搜到一个规则候选，但有限采样未证明连续区间唯一性 |
| `TIME_DOMAIN_INVALID` | 首尾半读出越出样条分配域 |
| `NON_POSITIVE_DEPTH` / `DISTORTION_DOMAIN_INVALID` | 真实前向算术或声明域失败 |
| `DISTORTION_UNCERTIFIED` | 当前便宜光学认证不足，不是物理无效证书 |
| `OUTSIDE_IMAGE` | 行根存在，但横坐标越出声明的原生图幅 |
| `ROOT_SEARCH_FAILED` | 求根迭代或等式算术容差未通过 |

单一候选根会保留 `uv` 供诊断，但只要状态非 `VALID`，`require_uv()` 就拒绝把它当成有效前向误差。`roots` 逐根保留行、时刻、像素、等式误差、隐式分母和条件数。

### 为什么默认不会把少量采样当唯一性证明

任意连续投影回调可能在采样之间隐藏根、切根或定义域缺口。本版本可发现符号变化的根，并以导数符号变化寻找切根候选；这不是完整区间根隔离。

`RSIntervalCertificate(interval,vertical_speed_bound,provenance)` 表示调用方已有的全区间证据：投影连续且定义域有效，并且在整个区间有 `|pi_y,t| <= bound`。当 `abs(tau)*bound/(H-1)<1` 时，行方程严格单调。求根器验证证书覆盖时间范围，且不会接受已经被实际评价导数直接反驳的上界。**把采样最大值或任意大的工程数填入证书，不能完成证明义务。** 零读出是可直接精确处理的例外。

本包尚未实现对完整机构/样条/Brown 自动传播区间界的算法。生产 C++ 重放必须补上等价严格求根和所选唯一性策略，或在物理状态足够收敛后以正式的全区间界认证。本参考不会为了让启动演示显示“成功”而省掉该限制。

### 无根时保留显式等式

`explicit_rs_equality(v,instant_projection)` 返回 `v-pi_y(X,t(v))`。当前状态无隐式根但瞬时算术有效时，可保留有界 `v`，把该等式接入外层等式 AL。每次试探仍使用真实瞬时物理投影，不是图像变形。恢复后必须切回真实隐式前向，并通过相关唯一根、等式与非奇异性重放。该辅助变量不能取消原始观测义务。

### 导数合同

```python
J = implicit_jacobian(
    pi_theta, pi_t, signed_readout_time=tau, image_height=1080,
    explicit_time_theta=t_theta,
)
```

`pi_theta` 为固定时间偏导，形状 `(2,n)`；`pi_t` 为同一物理投影的时间导数；`t_theta` 为固定输出行时的显式时间偏导。实现：

\[
J=(I-\pi_t w^T)^{-1}(\pi_\theta+\pi_t t_\theta),
\qquad w=(0,\tau/(H-1))^T.
\]

若某列参数是 `tau`，该列 `t_theta=v/(H-1)-1/2`；若参数是曲线站位，该项为零，但 `pi_theta` 必须包含曲线导数。不要先把行反馈塞进 `pi_theta`，然后再调用本函数，否则时间链重复计算。默认门限为 `abs(1-pi_y,t*tau/(H-1))>=1e-8` 和条件数不超过 `1e8`，是需要冻结的工程算术设置。

## 6. 有限曲线、闭合冰场与围挡展开

`FiniteLine(A,B)` 的站位为 `[0,1]`；`CircleArc(center_xy,radius,angle_start,angle_end)` 的站位为展开弧度，跨度不超过一圈。越界评价显式抛错。圆的相位/闭环 seam 可以通过一个明确的展开角区间表达，不按观测数组位置固定世界站位。

`FiniteLine.closest_station()` 仅提供世界欧氏距离的约束最优点，包含活动端点；它不是完整像素 AL 目标的 `S*` 消元。图像残差中的 `S` 仍由求解器与同一当前目标联合优化，尤其不能把跨点相关目标拆成独立最近点后宣称等价。

`RoundedRink` 的八段依次是底部长边、右下角、右边、右上角、顶部长边、左上角、左边、左下角，按逆时针闭合。四角共享 `radius_x,radius_y`，默认候选 `p=2`，是椭圆角而不要求圆角。所有直段正长、角部正半径、整体凸且接点相切。

**站位 `s∈[0,8]` 是分段参数，不是弧长或米制长度。** 接点单位切向连续，参数速度可以跳变，因此本实现不宣称关于 `s` 的 C1 或 C2。`point_unwrapped(s)` 明确使用周期展开坐标，支持材料连续跨过 `s=8` 的 seam。

旧种子中的 `p≈1.9051596` 既不能固定为真值，也不能在此内核中被静默改为 2。传入任何 `p!=2` 会抛 `UnsupportedGeometryModel`；接入稳定超椭圆端点与参数导数前，这部分种子必须在阶段状态中标记为未支持。

`BoardSurface(boundary,height)` 实现 `S(s,h)=B(s)+h e_z`，第一版仅支持共享常数高度。实际踢脚板、广告边缘等需要各自实际 `h`，不能自动统设为围挡顶高。

`BoardSplineMaterial` 使用固定三次局部 B 样条，控制点直接存储在展开的 `(s,h)` 中。控制点及其合法实体板区间为整段共享；构造器需要 `material_id` 与真实支持来源 `support_provenance`。没有帧参数，没有自由图稿 H8，也没有叠加的图稿坐标变换。形状系数修改后应创建新的共享场景版本，并让所有帧引用该版本。未知形状不提供已知米制长度。

## 7. 实体点、普通曲线和真实 profile

| 接口 | 语义 |
|---|---|
| `physical_point_residual(predict,PhysicalPoint,uv)` | 固定物理角点/端点的二维误差；API 没有 `s` 参数 |
| `ordinary_curve_residual(predict,curve,s,uv,allowed_interval)` | 有限材料上的二维原生误差；`s` 为显式合法未知站位 |
| `profile_auxiliary(predict,curve,s,alpha,query)` | 真实查询元数据的相交等式、法向误差和有限查询约束 |

`predict` 可以返回带 `require_uv()` 的相机结果，也可返回已经由上层保证合法的原生二维像素。前一种可保留完整失败状态。没有真实 profile 元数据时，应调用普通二维接口。重复 `source_point_id`、采样索引、图像 x 排序、裁剪位置都不产生固定物理对应。

`ProfileQuery` 要求真实的正交单位 `tangent,normal`、有限 `alpha_interval`、分别冻结的 `sigma_n,epsilon_n` 和查询来源。`profile_auxiliary()` 返回：

- `equality_xy = prediction - uv - alpha*normal`；这是独立等式 AL 义务。
- `tangent_equality_px = tangent.T @ (prediction-uv)`。
- `native_normal_error_px`、`statistical_residual=e_n/sigma_n` 和 `budget_q=e_n²/epsilon_n²`。
- 两个有限查询不等式 `alpha_min-alpha`、`alpha-alpha_max`。

这些几何等式/区间不能随公共像素预算 `b` 膨胀。无相交时可以有 `budget_q=0` 而 `tangent_equality_px!=0`；测试明确保留此失败。`find_profile_intersections()` 只作有限区间诊断搜索，返回无交点、多交点或一个尚未认证的候选；未实现完整交点隔离、切根认证及用于消元的分支 Jacobian。未恢复时使用显式 `(s,alpha)`。

`verified_order_inequalities()` 只接受 `verified=True`，允许相等站位。闭曲线需要上层先明确展开 seam 和方向；不同遮挡 run 不会自动串联。反向覆盖需要完整可见与完整取样证据，不由本模块生成。

## 8. C++ 集成与验证命令

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_physics.py -v
g++ -std=c++17 -Wall -Wextra -pedantic -Icpp/include cpp/tests/test_physical.cpp -o /tmp/rinkfit_test_physical
/tmp/rinkfit_test_physical
```

C++ 类型为 `rinkfit::Vec2<T>,Vec3<T>,Mat3<T>`，其中矩阵是行优先 `std::array<T,9>`。核心调用为：

```cpp
auto optical = rinkfit::zoom_optics(optics, log_f);
auto pose = rinkfit::camera_pose(head, alpha, beta, optical.z);
auto X = rinkfit::finite_line_point(A, B, station);
auto prediction = rinkfit::project(X, pose, optical.intr, optical.brown);
if (!prediction.valid()) return false;  // domain failure; never zero residual
```

Ceres 试探中的有限站位可以先用 `valid_station()` 检查并返回 `false`；本头文件的曲线函数对违法域抛 `std::domain_error`。支撑预算或等式违例应交给 AL，不要混为试探算术域错误。`cubic_weights/cubic_value` 只处理已经选好的合法局部单元；控制分配与完整时间边界在轨迹层。`gauge_dependent_control()` 对参考四点支持实施严格函数值消元。

头文件使用可由 ADL 找到的数学函数，支持 Ceres `Jet`；关于认证下界的比较仅用于真实定义域判定，不是拟合残差。最终 C++ 阶段还必须接入全部共享参数的约束流形、完整原生时间索引、显式曲线站位、真实相关结构、RS/profile 恢复与独立最终重放，不能用这个头文件冒充完整优化器。

## 已实现与未实现能力

**已实现并验证：** 固定机构的参考点不动与可移动光心；完整 Brown/低维焦段光学；原生均匀三次时间控制与代数函数值规范；解析相机时间导数；区间保护的 RS 根搜索、全区间证书接口、无根/多根/奇异/未认证状态及隐式 Jacobian；有限线/圆；共享闭合相切椭圆角边界；展开围挡与静态共享广告曲线；实体点、普通曲线和真实 profile 的不同义务；Python 与独立 C++ 测试。

**尚未实现：** 自动 RS 区间认证、C++ 完整隐式根求解/导数包装、稳定 `p!=2` 超椭圆与曲率过渡、非正交云台、面板高度/倾斜变化、真实可见性推断、光学/曝光输出备选、完整 profile 根隔离及分支消元、逆射线与自适应原生渲染、生产级批处理优化和全镜头真实拟合。请求这些模型时应停在明确的未支持阶段；不得静默替换候选或宣称验收完成。
