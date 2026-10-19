# 事件种子转换：可直接加载的新状态向量

`bootstrap.py` 读取本事件的 `data/seeds.json` 与 `data/camera_seeds.npz`，把保留的 1248 帧历史相机候选转换成新命名状态。它只做初值适配，不使用原始像素进行拟合，也不声称相机、场景或旧误差已经通过验收。

```bash
PYTHONPATH=src python -m rinkfit_start.bootstrap --data data --output /tmp/bootstrap_physical_state.json
```

函数调用为 `bootstrap_state(seed_catalog_path,camera_seed_path)`，返回可以直接用 `json.dumps(...,allow_nan=False)` 保存的字典。默认要求帧 1–1248、同镜头均匀 PTS、时基 `1/12800`、已有 `L=1` 世界种子。读取的数组均有明确字段名，不依赖旧 aux 列号。

## 转换过程

1. 使用 `legacy_head` 中的固定轴、参考旋转、支点、杆臂与轴间距。先从 tilt 轴不变量解析获得角度分支，再逐帧以两个变量的 SO(3) 旋转残差 `least_squares` 恢复 `alpha,beta`，随后展开连续角度。
2. 将首帧两个角度常量移入新的 `R0`，使同一机构严格满足首帧角度函数规范。轴、支点和杆臂仍然是整段共享量。
3. 使用每原生帧一个位置的三次 B 样条插值全部角度与 `log f`。额外边界控制初始化为局部线性延续，覆盖首尾半读出；这些边界初始化方程不是最终优化约束。角度用 `GaugeFixedSpline` 的 `N-1` 自由坐标序列化，重新构造时再次严格函数值消元。
4. 参考焦距取历史有效焦距的几何均值并固定；焦段坐标为 `z=log(f/f_ref)`。主点和径向系数投影到现有常数加线性焦段基底，切向项和纵横比为共享常量。所有转换差异都有回放记录。
5. 用历史光心相对固定机构的相机轴向偏差初始化一个共享二次入口瞳规律。常数项吸收到 `d_z`，保留 `ell(0)=0`。输出的 pupil 只是历史种子的一次低维转换，不是瞳位置量测。
6. 保留场景归一化尺寸和原有超椭圆 `p`；不会把旧 `p≈1.9051596` 改成 2，不产生任何米制场地真值。

## 输出结构

| 字段 | 内容 |
|---|---|
| `shared_head` | 可直接传给 `FixedHead(**state["shared_head"])` 的共享机构参数 |
| `camera_controls` | `UniformCubicBasis` 参数、两个角度自由控制、log-f 控制、函数参考与边界说明 |
| `optics` | 可直接传给 `ZoomOptics(**state["optics"])` 的当前低维光学候选 |
| `rolling_shutter` | 历史有符号读出种子与已分配的时间覆盖范围；未认证唯一根 |
| `scene_seed` | `L=1` 尺寸、角部 `p` 和共享板高；保留明确实现状态 |
| `clock` | 全部原生帧、PTS、固定原点和时基 |
| `conversion` | 全帧旋转、焦距、光心回放，光学基底近似差异及失败认证计数 |
| `provenance` | 两个输入文件的 SHA-256、原始候选与场景散列及转换角色 |
| `blockers` | 未实现超椭圆及可能出现的低维光学定义域认证不足 |

重新装载相机示例：

```python
from rinkfit_start.camera import (
    FixedHead, ZoomOptics, UniformCubicBasis, GaugeFixedSpline,
    UniformCubicSpline, CameraTrajectory, PhysicalCamera,
)

c = state["camera_controls"]
basis = UniformCubicBasis(**c["basis"])
trajectory = CameraTrajectory(
    GaugeFixedSpline(basis, c["alpha_free_controls"], c["reference_time"]),
    GaugeFixedSpline(basis, c["beta_free_controls"], c["reference_time"]),
    UniformCubicSpline(basis, c["log_f_controls"]),
)
camera = PhysicalCamera(FixedHead(**state["shared_head"]),
                        ZoomOptics(**state["optics"]), trajectory)
```

## 必须读取的限制

历史 Brown 系数随焦段变化不一定能由本内核的线性基底精确表达。转换会记录系数误差，并检查每个原生帧当前 `radius_max` 圆盘的保守无折叠下界；失败会进入 `blockers`，不会清零畸变或暗中切换光学模型。默认半径 2 是公开的工程定义域，后续候选必须依据实际所需射线范围与同一原生投影回放明确评估。

相机对象能被重建，不代表完整场景已经可前向运行。当前 `p!=2` 仍为显式不支持；后续 Codex 应实现稳定超椭圆端点与导数，并接入每条原始材料的有限站位/分支，随后执行真实物理拟合。旧 RS 时间只是一枚初始化种子，不能作为读出量测，也不能借此声称行根唯一。

## 已实现与未实现能力

**已实现：** 本事件全 1248 帧命名种子读取与核验、两角机构转换、首帧严格函数规范、全部原生控制及边界覆盖、共享 pupil/光学低维转换、可序列化新状态、输入散列与逐帧回放。

**未实现：** 超椭圆 `p!=2` 内核、光学/RS 全域认证、全部原始观测的站位/分支适配、真实像素拟合与最终验收。所有缺口均写入状态或能力说明，不将历史初值包装成物理结果。
