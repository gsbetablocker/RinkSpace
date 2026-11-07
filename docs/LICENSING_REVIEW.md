# 公开分发与分项许可

本页汇总已有许可核查及分发范围。基础核查记录于 2026-10-19，2026-11-02 补核三维播放器及现用人体依赖；后续场景数值资产沿用相同边界；不是对全部私有历史、已安装环境或所有用途的许可认证。项目采用[分项许可与权利范围声明](../LICENSE.md)，第三方来源见 [THIRD_PARTY_ASSETS](THIRD_PARTY_ASSETS.md)。

## 项目与第三方内容

项目自有且未另行许可的内容目前不额外授予使用、修改、分发、再许可或商用许可；第三方原有授权继续有效，不能将整个仓库改标 MIT。

项目辅助代码和场地图的生成／适配身份保存在来源记录中。第三方标志、图稿、源码和权重的权利不会因数值参数化、裁切或项目声明而改变。完整广告 SVG 原件不随公开树分发，来源 URL、文件身份与历史地址仍保留。

| 范围 | 已登记证据与边界 |
|---|---|
| SAM2 基础权重 | `facebook/sam2.1-hiera-tiny`，固定修订 `de431c4043854a71d8101e17995dfe596bf101a5`；下载收据、本地 SHA 与[固定提交 LFS](https://huggingface.co/facebook/sam2.1-hiera-tiny/commit/de431c4043854a71d8101e17995dfe596bf101a5)一致，模型卡标 Apache-2.0 |
| rink_vision.pt | SHA256 `26e792f2381a9bd66c484b8cfa38b13993f5c6cf11b52ef84ee22da65041f208`，对应训练 step 9200；162 个编码器张量／108876544 字节与基础权重一致，208 个存储均为模型状态，未发现额外原图缓冲区；[来源与训练记录](../models/rink_vision.provenance.json)、[SAM2 许可](../models/SAM2_LICENSE.txt)保留 |
| GVHMR 补丁 | 固定上游提交 `6ec3ca39336c50492c0fae65fba2fb831fc7d866`；[原许可](../environments/third_party_patches/GVHMR_LICENSE.txt)、补丁和研究引用见[资产说明](THIRD_PARTY_ASSETS.md)，保留上游教育／研究／非营利及衍生修改非商业条件 |
| ExMR camera-only补丁 | 固定提交 `eac72e47339b404c0443f5fb323412e2ce432410`；[完整上游许可与署名](../environments/third_party_licenses/EXMR_LICENSE.txt)原字节保留，教育／研究／非营利及衍生修改非商业条件继续适用；完整源码、SAM/HaMeR/RTMW权重及网络原始输出留Data，固定资产身份见[依赖索引](../events/ankara_2026/programmes/okada_mei_sp/spin_models_1104_audit.json) |
| Ceres／Abseil、SciPy、oneTBB | 原许可证和署名保留；分别适用 BSD、嵌入组件的 Apache 等原条件，项目声明不缩减第三方授权 |
| 项目辅助代码 | ViTPose helper 与 calibration companion 按项目分项声明处理；[helper 来源](../calibration/program_tracking/body_tracking/source_helpers/run_vitpose_h.provenance.json)、[companion 来源](../src/rinkspace/calibration_companion/provenance.json)保留，实际运行依赖仍适用原许可 |
| 外部模型与框架 | GVHMR、HMR2、ViTPose、SMPL-X、YOLO、RAFT 及 RTMW 运行实现／权重留在 Data；[FootMR 与 CoTracker 的当前源码接线、固定提交和权重身份](../environments/body-runtime-sources.json)已实际接入，不能再将 CoTracker 描述成仅有参考链接。PIDNet／SMP 仍按原核查范围解释 |

<a id="viewer3d"></a>
## 三维播放器与外部窗口依赖

[播放器](RESULTS_3D_PLAYER_ZH.md)连接外部 AITViewer 1.14.2；本机该发行物附 MIT 许可，Pyglet 2.1.19 附 BSD 三条款许可。原文、安装元数据 SHA 与源码证据见[播放器依赖记录](../environments/viewer3d-provenance.json)。这描述指定版本的现有字节，不给全部历史版本或整个仓库改标 MIT。

本项目显式选择 Pyglet 窗口后端。AITViewer 上游默认配置和安装依赖仍含 PyQt6；本机 PyQt6 6.12.0 发行物附 GPLv3，PyQt6-Qt6 6.12.0 元数据标 LGPL v3，二者不能混为一项 LGPL 依赖。[Riverbank 官方说明](https://en.riverbankcomputing.com/software/pyqt)列出 GPLv3 与商业许可两种路径；本记录未取得商业授权材料。若改为 Qt 后端或分发含 Qt／PyQt 的组合程序，需按实际使用与原许可另行核对，不能沿用项目保留权利声明作为第三方授权。

公开树仅保留连接源码、依赖定义、许可原文与身份记录，不附播放器安装环境或 Qt／PyQt 实现。安装锁中出现库、运行时实际加载库以及组合分发是不同事实；默认后端选择不替代整个依赖闭包的许可核查。

完整人体显示还使用外部 SMPL-X Python 包和已登记模型，继续适用其各自条款。完整表面、姿态显示缓存及检查图均留在 Data；数值输出和显示几何的存在不授予模型再分发或商业许可。

<a id="body-parameters"></a>
## S01 人体、足鞋与开头参数

| 参数组 | 内容与身份记录 |
|---|---|
| `body_fullbody_1102`、`boot_integration_1102`、`opening_repair_1102` | 节目控制量、源时间、尺度、绑定与来源身份；足鞋参数另含项目简化鞋壳／内腔拓扑和足部顶点索引，开头参数包含缓存姿态转换的局部旋转、初值与修正量 |
| 开头时序／右臂、鞋刀、动作 A/B 与根修正、历史 S01 整合 | 控制向量、P/Q、局部旋转与先验、连续修正层及结点、共享鞋形、源 PTS 和算法／绑定身份 |
| 提刀、支撑脚参照、连续入旋／质心重力初值及控制 | 小型控制层、P/Q／局部手姿态、共享鞋参数、源身份和初始化记录；分段体积质量近似仅保存小型分组质心、正归一化体积分数与假设，不含完整 SMPL-X 表面或模型字节 |
| 源支撑旋转与 GVHMR 自身初值／联合控制 | 连续控制层、源 PTS、固定身份和重放初值；原始 ExMR／SAM 网络输出保留在 Data |
| 2026-11-05 连续人体 v1/v2 与局部换腿修复 | 根及局部 SO(3) 控制、初值 P/Q/局部与手旋转、显式 SU(2) 分支参考、源 PTS 与固定身份；不含模型模板、形变基、完整表面或广播像素 |
| 2026-11-05 靴轮廓／鞋轴与runner支撑形态候选 | 既有绝对人体曲线与冻结鞋绑定、小型控制/源时间/身份及有限先验参数；不含新增鞋体自由度、模型权重、完整人体表面或广播像素；仍未通过刀刃姿态与接冰验收 |
| 2026-11-05／06 整段人体与相对鞋曲线、完整源轮廓及条件支撑位置候选 | 绝对人体与踝枢轴相对鞋 SO(3) 控制、原生 PTS、固定绑定／输入身份和保留的失败参数；源描线 JSON 只含像素坐标、结构解释与来源哈希，不含截图、完整人体表面或模型权重；对应逐字段审计继续由同一参数索引记录 |

2026-11-06／07 全节目候选另外保存共同 P/Q 与身体／相对鞋连续控制、共享形状及条件尺度、原生 PTS、源关联和有限先验身份；尺度／深度及关联失败参数同样仅为数值控制记录，不含完整表面、模板、蒙皮权重、网络原始输出或广播画面。

具体文件 SHA、字节数、字段名、dtype 和 shape 见[字段索引](../events/ankara_2026/programmes/okada_mei_sp/body_parameters_1103_fields.json)，较早三份参数的完整身份由[节目绑定](../events/ankara_2026/programmes/okada_mei_sp/README.md#历史人体候选导航)定位。

这些参数不含 SMPL-X 模板、形变基、蒙皮权重、人体面表、完整人体表面或广播像素缓冲区；紧凑模型、模型文件和表面缓存继续留在 Data。重放需要相符模型与本地输入。[SMPL-X 官方模型条款](https://smpl-x.is.tue.mpg.de/modellicense.html)与代码包许可分别核对，当前网页不替代已安装版本附带的许可原文。字段清单不认证第三方先验或数值输出可自由再分发、商用，也不将独立优化控制值直接等同于上游附带的模型动画。

公开参数保留失败初值和未验收候选，不能作为人体、轨迹或接触准确性的证明。开放鞋口与通用下鞋／冰刀参考实现的用户交接来源见[第三方资产记录](THIRD_PARTY_ASSETS.md#boot-handoff)及[鞋刀交接记录](THIRD_PARTY_ASSETS.md#skate-handoff)；项目分项许可与第三方原条款保持不变。

## 比赛素材与数值观测

公开仓库不分发 ISU 比赛视频、截图或叠加画面。原片和图像在 Data 保留，公开说明提供[官方来源与源时间](GITHUB_REVIEW.md#官方视频与源时间)；原 UV、PTS、源 SHA、人工事实和数值观测保留。

冻结 review 的 [manifest](../events/ankara_2026/programmes/okada_mei_sp/review/manifest.json)通过 `public_withheld/public_media_derivative` 登记原机位图片、native 截图及叠加画面的排除与派生身份。分类掩膜、项目场地图、观测坐标、方差、状态和统计 W 按各自类型保存，不因为文件名或含色值测量而当作完整广播图像删除。

观测 JSON／GZ／NPZ 中的颜色采样、剖面、ROI 测量与原 SHA 保持原值，以保留数据和状态合同；这些数据不会因此获得第三方额外许可。小型相机控制数组、广告数值模板及 Ad3／Ad7 数值源库不含广播图像缓冲区。历史模板、导入快照及回执按[归档映射](../events/ankara_2026/scene/archive.json)定位。

[YouTube 许可说明](https://support.google.com/youtube/answer/2797468?hl=en)与[使用条款](https://www.youtube.com/static?template=terms)不自动授予观看者服务外再分发权。两条具体视频的实际 CC 字段未取得，不能把平台默认值当作视频许可证据；当前公开范围不分发这些画面。

公开包支持代码、模型、数值定义与失败状态审阅。原像素核对、颜色剖面重放及完整镜头视觉验收需要本地原素材。

## 训练与依赖的已知缺口

rink_vision.pt 的 SAM2 基础身份和原许可已有证据，训练输入来自历史广播源 `oxXu1avhrvs` 与项目标注，既有记录中未提供相应授权材料。移出原图不能证明所有训练使用或派生模型公开／商业用途均已获授权；具体训练例外、输出权利与专利范围没有在本核查中形成结论。

[pyproject.toml](../pyproject.toml)的 advanced 声明 quadprog 0.1.13，[发布者元数据](https://pypi.org/project/quadprog/0.1.13/)标 GPLv2+。公开包未附该库二进制，已有实际 import 见环境 smoke；依赖声明与实际组合分发分别判断，适用条件见 [GNU 说明](https://www.gnu.org/licenses/gpl-faq.en.html#IfLibraryIsGPL)。本页记录来源和缺口，不变更项目许可或运行依赖。
