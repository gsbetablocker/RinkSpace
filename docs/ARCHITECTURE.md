# 架构与现有接口

RinkSpace 的长期目标是面向所有花样滑冰转播的通用本地空间分析工具。公共软件与处理流程适配不同赛事、冰场、机位和节目；每个赛事建立有来源的三维冰面、围挡、广告和相机模型，在摇镜、变焦、切镜中保持场地对应，恢复选手空间位置与连续运动轨迹，并扩展冰刀接触、速度和覆盖分析。

这是目标架构，不是已经完成的能力声明。项目目标不随开发样本或本轮任务变化；任务范围、训练版本、运行状态、误差与验收结果单独记录。公开使用从 [README](../README.md) 开始，选定结果及数据边界见[离线审阅](GITHUB_REVIEW.md)。

## 公共基础、赛事与节目

| 层级 | 负责内容 | 复用边界 |
|---|---|---|
| 公共基础 | 源视频/PTS 索引、机位分类、观测接口、场景与相机数学、求解器、视觉模块、审阅和统计 | 复用项目代码与接口，不为每个节目复制求解器 |
| 赛事 | 源登记、冰场几何与标线、围挡广告实例及拓扑、安装时期与机位、观测来源、模型版本和尺度依据 | 同赛事节目引用同一模型；跨赛事复用方法，分别建立场景数据 |
| 节目 | 镜头与节目时钟、逐曝光相机状态、人物观测、位置/接触分段、检查点和审阅结果 | 绑定所属赛事模型与真实源时间，保留节目特有结果 |

圈、Lane、logo、广告数量、位置、尺寸和相机安装参数属于赛事数据。不能把某场的布局当作所有冰场的标准，也不能跨赛事借用广告对应、划痕或相机数值。赛事内冰面重刷、广告更换或机位移动也应记录适用时期。

这三层是数据与接口的归属原则，现有适配器尚未全部完成解耦。`src/rinkspace/`、`scripts/` 与 `calibration/solver_runtime/` 保存项目代码；`configs/` 保存小型入口和兼容配置；最终视觉参数在 `models/`，赛事和节目模型在 OneDrive `events/<event_id>/`。原视频、批量帧图、训练数据、大数组、完整求解检查点、编译缓存和诊断产物放在本机配置的外部 Data。必要模型输入、人工事实与约束通过事件 manifest 明确归属；旧路径保留兼容，不能把 Data 快照当作唯一核心源码。路径不能代替源哈希、PTS、模型版本和观测身份。

## 处理链路

```text
已登记真实转播 → 源 PTS 与完整分镜 → 物理机位分类与安装时期
                                      ↓
                 多视角真实观测 → 赛事共同三维场景与相机模型
                                      ↓
                    每镜姿态/变焦与连续观测 → 原片全部模型叠加
                                      ↓
                       有定义的选手空间位置与分段轨迹
                         └─ 连续性辅助场地与相机核验
                                      ↓
                         接触、速度与覆盖分析
```

摇镜和变焦不自动产生新机位；机位代表图不能替代完整镜头表。局部配准、分割和参照检索提供观测或初始化，不能代替共同模型与独立定位验证。

## 接口地图与实现边界

| 入口 | 已有用途与限制 |
|---|---|
| [storage.py](../src/rinkspace/storage.py)、[media.py](../src/rinkspace/media.py)、[workspace.py](../src/rinkspace/workspace.py) | 项目/Data 资源桥、源登记、解码与 PTS/代理对应；浏览器 `currentTime` 不是精确源帧身份 |
| [event_survey.py](../src/rinkspace/event_survey.py)、[camera_classification.py](../src/rinkspace/camera_classification.py)、[camera_timeline.py](../src/rinkspace/camera_timeline.py) | 转播调查、镜头目录与机位归属；检测候选和外观类别仍需与物理机位区分 |
| [scene_camera.py](../src/rinkspace/calibration_companion/scene_camera.py) | `Scene`/`Camera` 三维几何、含畸变投影及像素射线与冰面交点；射线交点不等于人体的世界垂直投影 |
| [joint_scene.py](../src/rinkspace/calibration_companion/joint_scene.py)、[pivot_camera.py](../src/rinkspace/calibration_companion/pivot_camera.py)、[wide_angle_camera.py](../src/rinkspace/calibration_companion/wide_angle_camera.py) | `SharedGeometry`/`JointSceneProblem`、`TripodPivotJointProblem` 和广角扩展；共享场景、逐曝光相机及声明的镜头/安装参数联合估计，未观测几何仍有条件性 |
| [advertisement_tracks.py](../src/rinkspace/calibration_companion/advertisement_tracks.py)、[advertisement_topology.py](../src/rinkspace/calibration_companion/advertisement_topology.py) | 广告实例、围挡表面细节与用户确认的拓扑；图像观测和估计出的 XYZ 分开，示意位置不升级成测量 |
| [active_scene.py](../src/rinkspace/calibration_companion/active_scene.py) | 当前三维模型选择与哈希/契约检查，仍含赛事特例；不能与旧二维 atlas/delivery 格式混用 |
| [runtime.py](../calibration/solver_runtime/runtime.py)、[config.py](../calibration/solver_runtime/config.py)、[event_factories.py](../calibration/solver_runtime/event_factories.py) | 项目拥有的持久求解接口；公共数值核心接受调用者的因子、完整状态和物理 guard，`named66` 等赛事适配器仍依赖原赛事图与输入 |
| [vision_data.py](../src/rinkspace/vision_data.py)、[vision.py](../src/rinkspace/vision.py)、[vision_geometry.py](../src/rinkspace/vision_geometry.py) | 有源绑定的观测清单、GPU 视觉训练/推断与几何诊断；当前配置与部分曝光证明仍绑定开发样本 |
| [registration.py](../src/rinkspace/registration.py)、[local_registration.py](../src/rinkspace/local_registration.py)、[rink_reference.py](../src/rinkspace/rink_reference.py) | CPU SIFT/LK 与二维候选配准、矩阵桥接和记录校验；不自动提供三维场景或已验证米制尺度 |

现有联合场景接口仍有特定圈/Lane 身份约束，部分运行入口仍使用安卡拉和冈田样本。以上模块可复用不等于任意赛事已可直接运行；跨赛事能力必须用新的冰场与转播证据验证。

## 几何、时间与状态

相机世界到相机约定为 `X_camera = R @ (X_world - C)`；冰面是场景的 `Z=0`。光心、安装支点、杆臂与图像主点分开。固定光心是旧射线路线的条件假设；现有枢轴模型允许姿态相关的光心移动和声明的变焦轴向项，不能把固定光心写成项目规则。

二维配准采用列向量：`p_dst ~ H_dst_from_src @ p_src`。每份矩阵记录方向、源/目标坐标域、分辨率、畸变状态、参照与模型版本。冰面单应只映射相应平面；带畸变的原图使用完整非线性投影链，广告板上的点使用其三维围挡表面。

源帧身份使用源哈希、整数 PTS 与 `time_base`。展示频率、独立曝光频率、节目时钟及回放时钟分别记录。60 Hz 展示与官方 50 Hz 曝光的对应只能沿核实的源证据复用，不能推广到所有转播。代理帧、重复曝光或插值图像不制造新源观测。

完整检查点保留共同几何、相机状态、源身份、权重/profile、mask、尺度和数值续算状态；新增源包沿现有接口接入。模型版本变化使下游轨迹和统计的适用性需要重新检查，旧失败与原始证据仍可追溯。

## 展示与能力声明

原片审阅投影同一套全部已建模网格、圈、Lane、其他冰面标线和广告框编号；俯视图画同一模型的标记与广告位置。观测置信度可另行显示，不能以置信度或贴合好坏隐藏模型线。缺失相机状态明确标缺失，未建模广告边界不补造。

人体当前实现范围与长期能力分开记录。骨盆中心、骨盆的世界垂直投影、像素射线与冰面交点、冰刀接触是不同对象；二维人体点经冰面 H 映射不能直接称为骨盆垂直投影。详见 [算法说明](ALGORITHMS.md) 与 [验证说明](VALIDATION.md)。

`pyproject.toml` 声明基础 Python/NumPy/OpenCV/Pillow/PyAV 依赖；视觉 GPU、SciPy 扩展和原生求解后端有各自运行条件。模块可导入、数值收敛、网格平滑或历史测试通过，都不自动证明真实画面准确或通用目标完成。

## 目录导航

| 目录 | 职责 |
|---|---|
| src/ | 公共视频、时间、视觉、几何、反馈与本地 UI |
| calibration/ | 配准、物理求解、后端及源审阅；legacy 保留实际 import/pickle 加载契约 |
| events/<event_id>/ | 本赛事事实、模型、来源与机位安装；节目绑定在 programmes/<programme_id>/ |
| configs/ | 小型启动配置、资源桥和精确旧路径映射，现有机器地址需在新机器配置 |
| environments/ | 环境定义、构建 recipe 和来源身份；安装二进制不随源码分发 |
| models/ | 最终视觉参数及来源，部分权重经 Git LFS 保存 |
| scripts/ | 参数化工具和诊断入口；部分 diagnostics 仍被测试/运行导入 |
| tests/ | 基础合成用例、高级数值和专用回归；依赖范围见环境说明 |

不可替代人工事实、观测与必要失败状态按 manifest 保留，不能仅按后缀或 legacy/diagnostics 名称清除。工作台与生成审阅页在 Data，本机操作见[维护说明](../MAINTENANCE.md)。
