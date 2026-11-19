# 公开数据与离线审阅

本页说明仓库内容、原素材来源和复现边界。试用见 [README](../README.md)，能力见 [REBUILD](REBUILD.md)，准确性见 [VALIDATION](VALIDATION.md)。

<a id="public-cleanup"></a>
## 公开资产与存储边界

公开树包含项目源码、场景定义、已登记的场景和运动控制参数、带来源的数值观测及冻结技术审阅包。当前场地与四节目相机见[安卡拉赛事](../events/ankara_2026/README.md)。人体和鞋刀控制的字段范围见[许可核查](LICENSING_REVIEW.md#body-parameters)。

比赛原片、截图、叠加视频、完整第三方广告图稿、外部模型、运行环境和编译库保存在 Data。日常调试报告、运行日志、检查点及任务交接随各次运行归档。原始 UV、整数 PTS、统计 W、人工确认与失败证据保留在相应数值记录和冻结包中。原件与[公开派生身份](../configs/privacy_derivatives.json)各自使用对应字节的 SHA-256。

公开前按下游引用核对场景、节目和观测数据。运行产物的定位见[维护说明](../MAINTENANCE.md)，第三方来源与分发边界见[许可核查](LICENSING_REVIEW.md)。[独立 PTZ 工具](../tools/ptz_verifier/README_zh.md)的原片配对图和特征缓存保留在本机，其技术报告只覆盖该次抽样。

## 当前模型与数值来源

[赛事指定记录](../events/ankara_2026/scene/current_fit.json)绑定当前共同场地和四套节目。各节目登记记录相机、源时间和回放身份。广告数值源可查 [Ad3](../events/ankara_2026/programmes/okada_mei_sp/ad3_sources/correspondence.json)与 [Ad7](../events/ankara_2026/programmes/okada_mei_sp/ad7_sources/correspondence.json)的对应表，其中保存数值行、源 UV 和 PTS。

## 官方视频与源时间

| 节目 | 官方单节目来源 | 拟合处理区间（原片首末帧） |
|---|---|---|
| 冈田芽依短节目 | [ISU Skating：Mei OKADA](https://www.youtube.com/watch?v=O6j8L8TmfKo) · [源选择记录](../events/ankara_2026/programmes/okada_mei_sp/primary_source_selection.json) | 00:22.120–03:08.680，PTS 283136–2415104，8,329 个曝光 |
| 金书贤短节目 | [ISU Skating：Shuxian JIN](https://www.youtube.com/watch?v=SW1IewLFUyI) · [源选择记录](../events/ankara_2026/programmes/jin_shuxian_sp/primary_source_selection.json) | 00:32.300–03:20.720，PTS 413440–2569216，8,422 个原生帧 |
| Chen Xi Jin 短节目 | [ISU Skating：Chen Xi JIN](https://www.youtube.com/watch?v=f0K6hM9vj7Y) · [节目源与范围](../events/ankara_2026/programmes/chen_xi_jin_sp/rebuild_scope.json) | 00:27.040–03:18.120，PTS 346112–2535936，8,555 个曝光 |
| Nayeon Ko 短节目 | [ISU Skating：Nayeon KO](https://www.youtube.com/watch?v=4orLyh1JO3M) · [节目源与范围](../events/ankara_2026/programmes/nayeon_ko_sp/rebuild_scope.json) | S02–S10，00:20.800–03:08.420，PTS 266240–2411776，8,382 个曝光；不含 S01 |

四个单节目源均为 1920×1080、50 fps，时基为 `1/12800`。全场回放的帧对应、源哈希与下载表征由各节目源记录维护。冈田 2026-10-19 冻结失败快照使用历史范围 00:20.020–03:08.680，共 8,434 帧。


## 冻结九镜头失败快照

[节目 review](../events/ankara_2026/programmes/okada_mei_sp/review/README.md)冻结于 2026-10-19，保留九镜头失败状态、模型、配置、材料映射、完整状态、源观测、统计 W、资格召回和必要驱动。该候选未通过完整镜头定位验收，最后工程求解尚未消费原始 W 的统计似然。

归档清单把逻辑路径和历史地址映射到实际 ZIP 成员。模型与关键 JSON 可直接阅读，其余内容按需查询：

~~~console
python -B scripts/read_offline_review.py --check
python -B scripts/read_offline_review.py --get "Data/runs/okada_rec_1019/demo_grid_recovery_v1/candidate/model.json"
~~~

仅需 Python 标准库，无需本机 Data、求解环境或网络。`--check` 核对 ZIP 目录、成员和大小，不重新散列全部内容；`--get` 查询位置与身份。单项提取追加 `--output <新的绝对文件名>`，批量提取使用 `--extract <仓库外的新绝对目录>`；工具拒绝覆盖已有目标或批量展开到项目内，不执行包内代码。

Python 的 `Review.read(原路径或逻辑路径)` 可读取字节。`resource_map.json` 对应旧路径与短文件名；数组 dtype、shape、offset 和行语义按原 packet／manifest 解释。原生文本 checkpoint 即使以 `.json` 结尾，也不能直接当 JSON。

其他有独立输入和状态的历史试验见[节目历史](../events/ankara_2026/programmes/okada_mei_sp/README.md#历史索引)。

<a id="joint6-20261021"></a>
2026-10-21 六镜头候选未通过收敛或验收，详见[状态](../events/ankara_2026/programmes/okada_mei_sp/fit_joint6/status.json)和[范围](../events/ankara_2026/programmes/okada_mei_sp/fit_joint6/scope.json)。

## 复现限制

数值观测、模型身份与资格记录可用于技术审阅；原像素、颜色剖面和完整镜头视觉验收需要对应素材。场地图、分类掩膜与广播画面按各自来源登记。完整拟合、GPU 与人体能力还需[相应环境](../environments/README.md)；`models/rink_vision.pt` 使用 Git LFS，指针存在不等于 payload 已取得，技术审阅 ZIP 则是普通 Git 文件。
