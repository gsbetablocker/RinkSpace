# 通过 GitHub 阅读与离线审阅

公开前必读：[本次瘦身与隐私检查](PRIVACY_REVIEW.md)。当前 OneDrive 工作目录已就地脱敏，私人原件与旧 Git 历史在本地 Data 保全；公开仓库为[gsbetablocker/RinkSpace](https://github.com/gsbetablocker/RinkSpace)。日常只维护这个目录，不需要第二份项目或导出同步。

项目总入口仍是 [START_HERE.html](../START_HERE.html)。本文说明仓库里的实际文件、阅读顺序和离线数据读取方式，不授权启动历史任务。

## 先理解项目

RinkSpace 从花滑转播中恢复共同冰场、连续围挡、广告与相机，再连接选手位置和轨迹。公共方法放在源码，赛事几何与人工事实放在赛事目录，节目保存自己的源时钟、观测与模型绑定。

| 要审阅的内容 | 仓库位置 |
|---|---|
| 项目目标、已有能力和范围 | [README](../README.md)、[架构](ARCHITECTURE.md)、[算法](ALGORITHMS.md) |
| 原片时钟、取点资格、真实验证 | [验证要求](VALIDATION.md)、[源选择](../events/ankara_2026/programmes/okada_mei_sp/primary_source_selection.json) |
| 公共工具、视觉与持续纠错 | [src/rinkspace](../src/rinkspace/)、[src/rink_feedback](../src/rink_feedback/)、[反馈记录](../events/ankara_2026/feedback/index.html) |
| 原有场地与节目求解器 | [solver_runtime](../calibration/solver_runtime/README.md)、[program_tracking](../calibration/program_tracking/) |
| 最近的九镜头物理求解实现 | [rinkfit/src](../calibration/rinkfit/src/)、[C++ 实现](../calibration/rinkfit/cpp/)、[joint_solver.py](../calibration/rinkfit/src/rinkfit_joint/joint_solver.py) |
| 冰场、广告与机位的事实/约束 | [model_constraints](../events/ankara_2026/model_constraints/)、[模型](../events/ankara_2026/models/) |
| 当前失败态与关键审阅数据 | [节目 review](../events/ankara_2026/programmes/okada_mei_sp/review/README.md) |
| 依赖、版本、原生构建与第三方来源 | [环境说明](../environments/README.md)、[运行依赖](RUNTIME_DEPENDENCIES.md)、[第三方资产](THIRD_PARTY_ASSETS.md) |

历史启动包中的 CODEX_START、执行计划和命令是当时的交接材料。它们不代表当前工作已完成，也不覆盖 2026-10-19 的停机状态。

## 当前离线审阅快照

快照为 2026-10-19 用户要求停止后的全九镜头恢复失败状态。官方单节目 O6j8L8TmfKo，50/1 fps，time_base=1/12800，源 PTS 256256–2415104，共 8434 帧。当前模型没有通过全九镜头定位验收；原始 W 的统计似然尚未被最后工程求解实际消费，约束与独立重放仍未闭合。原始数据、W、失败状态和资格召回记录分别保留，不能因归档就变为合格源或成功模型。

review 目录直接提供可读的状态、当前模型、配置、材料映射和关键驱动源码。其余必要观测、完整状态、W、源资格记录和旧基线压成少量 ZIP；清单把每个原路径对应到实际归档成员。不是让审阅者去访问本机 Data，也没有将整个历史目录复制到 OneDrive。

完整源视频、批量解码帧、安装环境、编译中间件、可重建的 4.7 GB 屏幕网格数组和大批诊断视频仍在 Data。仓库保留数值审阅数据，比赛截图与叠加图不公开；完整 8434 帧的独立视觉验收仍需官方源视频。旧 60 帧全场来源只保留历史对齐事实，不用作本节目的新处理源。

## 离线读取，无需本机 Data

下载完整仓库后，审阅归档本身只需 Python 3.11 标准库，无需安装求解器或访问网络：

~~~powershell
python -B scripts/read_offline_review.py --check
python -B scripts/read_offline_review.py --get "Data/runs/okada_rec_1019/demo_grid_recovery_v1/candidate/model.json"
~~~

第一条只核对 ZIP 目录、成员存在性与大小，不展开文件。第二条显示该项的实际位置与身份。要取出单项，追加 --output 和一个新的绝对文件名。要批量提取，使用 --extract 和仓库外的新绝对目录；工具拒绝批量展开到项目中，也不会覆盖已有目录或执行归档内的程序。

~~~powershell
python -B scripts/read_offline_review.py --extract D:\RinkSpace_Data\offline_review
~~~

提取结果采用短文件名，并提供 resource_map.json，将原路径和逻辑路径映射到实际文件。原 JSON 中的历史绝对地址是清单查询键；私人标识已改为公开别名，变化成员登记原始和公开 SHA，不能据此声称原运行身份不变。也可在 Python 中导入 Review，直接 read(原路径或逻辑路径) 取得原始字节，不必全量解压。数组的 dtype、shape、offset 和行语义来自原 packet/manifest，不能只按后缀猜测。

完整拟合、GPU、人体和原生库的跨机器运行仍按环境文档逐项准备；离线审阅不依赖这些安装。models/rink_vision.pt 属 Git LFS，未来克隆权重需安装 Git LFS 并拉取 payload；本次技术审阅 ZIP 使用普通 Git 文件。

## 存储与推送准备

OneDrive 保存源码、依赖/构建定义、模型、人工事实与必要审阅归档。Data 保存大容量运行产物和本机 Git 数据库。压缩保全工具只处理显式清单，在 Data 中构建后再发布到项目，不自动扫描并拷贝历史，不覆盖已有快照。新增路径均检查 Windows UTF-16 长度；全树检查覆盖忽略的环境与缓存。

三个历史入口的默认批量输出已通过现有 AssetStore 放到 Data：build_camera_timelines.py、survey_event_broadcasts.py、track_scene3d_observations.py。最后一项的来源 provenance 继续记录原版身份；默认输出改动是之后的维护修改，没有篡改来源记录。

推送前可运行：

~~~powershell
python -B scripts/check_onedrive_paths.py
python -B scripts/check_github_readiness.py --help
~~~

检查包含工作区/索引/HEAD 的大小与 LFS 状态、可达历史的大对象，以及明确限定范围的凭证候选位置；被跳过的文本必须按报告解释。工作区的 LFS 属性不等于已提交指针或远端已有 payload。GitHub 对普通文件超过 50 MiB 警告、超过 100 MiB 阻止，见[官方说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。本项目不另设文件数量或仓库容量门槛。

本次旧历史已原封本地保全，同一工作目录接入空的新公开历史；使用GitHub公开的noreply身份，origin指向公开仓库。项目采用[分项许可声明](../LICENSE.md)，第三方原许可保留。本仓库的公开准备不等于完整节目已准确复现。
