# rinkfit：配准与物理求解接口

本模块包含有限材料的平面配准、跨帧图、条件度量初始化、固定云台与光学原语，以及 Ceres/PHR 和九镜头联合求解代码。公共数学和接口在本目录；赛事、节目源绑定和模型参数在 events 中。

## 入口与依赖

| 入口 | 用途 |
|---|---|
| [launch.py](launch.py) | S0 输入准备、S1 配准、图、条件升级、bootstrap、后端导出及 fit 命令；使用 --help 查看参数 |
| [registration_api](docs/registration_api.md)、[metric_upgrade_api](docs/metric_upgrade_api.md)、[bootstrap_api](docs/bootstrap_api.md) | 初始化接口与退化状态 |
| [physics_api](docs/physics_api.md)、[solver_backend](docs/solver_backend.md) | 固定机构、有限材料、原生状态与 Ceres 后端契约 |
| [event_integration](docs/event_integration.md) | 初始首镜头包的输入 ABI、统计/语义与事件接入合同 |
| [src/rinkfit_joint](src/rinkfit_joint/)、[tools/joint9.py](tools/joint9.py) | 多镜头装配和恢复接口；具体状态按所选节目绑定读取 |
| [历史验证](docs/validation.md)、[来源](docs/provenance.md) | 2026-10-17 初始包的实测范围与来源，不是当前完整节目认证 |

Python 依赖见 [requirements.txt](requirements.txt)，环境分层见[项目环境](../../environments/README.md)。初始 Ceres 构建曾在 Linux/GCC 13.3/CMake 3.28/Ceres 2.2/Eigen 3.4/SuiteSparse 7.6 上检查；不据此宣称 Windows 原生已验证。构建目录和运行输出应位于配置的 Data 或仓库外目录：

~~~console
cmake -S calibration/rinkfit -B <仓库外构建目录> -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build <仓库外构建目录>
ctest --test-dir <仓库外构建目录> --output-on-failure
~~~

## 输入、输出与限制

初始首镜头合同描述 1248 帧、182017 条原始 UV、171 个标签，来源主键为 (shot_id, native_pts, source_point_id)，不是单独 point ID。工程预算 E 与 sigma/完整分组 W 分开；局部配准和图一致性只能作初始化，不能替代最终物理误差与独立回放。

当前公开源码没有附带原启动包完整 data/、spec/ 和 package_manifest.json。因此 launch.py 的 prepare/start/inspect、tools/check_package.py 以及文档中的包内输入例子不能作为无需材料的公开快试。应按[节目审阅快照](../../events/ankara_2026/programmes/okada_mei_sp/review/README.md)与来源记录定位相应输入；源码/API 保留不等于原包已经完整重建。基础试用使用根 README 的合成 demo。

后端输入保存源行、PTS、UV、字符串表、材料/语义和统计身份；完整状态还需参数、station、分支、RS/query witnesses、约束乘子及数值续算状态。输出和续算必须绑定同一冻结输入，不能用局部 H checkpoint 代替物理状态。

原始统计 W 的消费、全部有限材料的真实重放、物理可辨识性与完整镜头准确性须分别验证。历史初值、合成后端、局部 RMS 和数值退出均不表示真实拟合成功。场地广告按连续围挡展开面定义；米尺度、相机事实及资格均按赛事来源核实，见[主验证说明](../../docs/VALIDATION.md)。
