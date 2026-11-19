# 公开分发与分项许可

本页记录公开树中的资产、适用许可及证据范围。项目自有内容的主声明见[许可声明](../LICENSE.md)，第三方版本与文件身份见[第三方资产](THIRD_PARTY_ASSETS.md)。

## 项目与第三方内容

项目自有且未另行许可的内容目前不额外授予使用、修改、分发、再许可或商用许可；第三方原有授权继续有效，不能将整个仓库改标 MIT。

项目辅助代码和场地图的生成／适配身份保存在来源记录中。第三方标志、图稿、源码和权重的权利不会因数值参数化、裁切或项目声明而改变。完整广告 SVG 原件不随公开树分发，来源 URL、文件身份与历史地址仍保留。

| 范围 | 已登记证据与边界 |
|---|---|
| SAM2 基础权重 | `facebook/sam2.1-hiera-tiny`，固定修订 `de431c4043854a71d8101e17995dfe596bf101a5`；下载收据、本地 SHA 与[固定提交 LFS](https://huggingface.co/facebook/sam2.1-hiera-tiny/commit/de431c4043854a71d8101e17995dfe596bf101a5)一致，模型卡标 Apache-2.0 |
| rink_vision.pt | SHA256 `26e792f2381a9bd66c484b8cfa38b13993f5c6cf11b52ef84ee22da65041f208`，对应训练 step 9200；162 个编码器张量／108876544 字节与基础权重一致，208 个存储均为模型状态，未发现额外原图缓冲区；[来源与训练记录](../models/rink_vision.provenance.json)、[SAM2 许可](../models/SAM2_LICENSE.txt)保留 |
| GVHMR 补丁 | 固定上游提交 `6ec3ca39336c50492c0fae65fba2fb831fc7d866`；[原许可](../environments/third_party_patches/GVHMR_LICENSE.txt)、补丁和研究引用见[资产说明](THIRD_PARTY_ASSETS.md)，保留上游教育／研究／非营利及衍生修改非商业条件 |
| ExMR camera-only补丁 | 固定提交 `eac72e47339b404c0443f5fb323412e2ce432410`；[完整上游许可与署名](../environments/third_party_licenses/EXMR_LICENSE.txt)原字节保留，教育／研究／非营利及衍生修改非商业条件继续适用；完整源码、SAM/HaMeR/RTMW权重及网络原始输出留Data，固定资产身份见[依赖索引](../events/ankara_2026/programmes/okada_mei_sp/spin_models_1104_audit.json) |
| HTD-Refine 与 PVA-Net | 固定提交 `2fcd6ddef3c4eb75a636062245f80a0136c09b7e` 的外部源码适用 AGPL-3.0-only；公开内容为项目输入适配与复核代码，checkpoint 许可另行核对。 [来源与署名](THIRD_PARTY_ASSETS.md#htd-refine) |
| Ceres／Abseil、SciPy、oneTBB | 原许可证和署名保留；分别适用 BSD、嵌入组件的 Apache 等原条件，项目声明不缩减第三方授权 |
| 项目辅助代码 | ViTPose helper 与 calibration companion 按项目分项声明处理；[helper 来源](../calibration/program_tracking/body_tracking/source_helpers/run_vitpose_h.provenance.json)、[companion 来源](../src/rinkspace/calibration_companion/provenance.json)保留，实际运行依赖仍适用原许可 |
| 外部模型与框架 | GVHMR、HMR2、ViTPose、SMPL-X、YOLO、RAFT 及 RTMW 运行实现／权重留在 Data；[FootMR 与 CoTracker 的当前源码接线、固定提交和权重身份](../environments/body-runtime-sources.json)已实际接入，不能再将 CoTracker 描述成仅有参考链接。PIDNet／SMP 仍按原核查范围解释 |

<a id="viewer3d"></a>
## 三维播放器与外部窗口依赖

[播放器](RESULTS_3D_PLAYER_ZH.md)连接外部 AITViewer 1.14.2；本机该发行物附 MIT 许可，Pyglet 2.1.19 附 BSD 三条款许可。原文、安装元数据 SHA 与源码证据见[播放器依赖记录](../environments/viewer3d-provenance.json)。这描述指定版本的现有字节，不给全部历史版本或整个仓库改标 MIT。

本项目显式选择 Pyglet 窗口后端。AITViewer 上游默认配置和安装依赖仍含 PyQt6；本机 PyQt6 6.12.0 发行物附 GPLv3，PyQt6-Qt6 6.12.0 元数据标 LGPL v3，二者不能混为一项 LGPL 依赖。[Riverbank 官方说明](https://www.riverbankcomputing.com/software/pyqt/)列出 GPLv3 与商业许可两种路径；本记录未取得商业授权材料。若改为 Qt 后端或分发含 Qt／PyQt 的组合程序，需按实际使用与原许可另行核对，不能沿用项目保留权利声明作为第三方授权。

公开树仅保留连接源码、依赖定义、许可原文与身份记录，不附播放器安装环境或 Qt／PyQt 实现。安装锁中出现库、运行时实际加载库以及组合分发是不同事实；默认后端选择不替代整个依赖闭包的许可核查。

完整人体显示还使用外部 SMPL-X Python 包和已登记模型，继续适用其各自条款。完整表面、姿态显示缓存及检查图均留在 Data；数值输出和显示几何的存在不授予模型再分发或商业许可。

<a id="body-parameters"></a>
## 人体与鞋刀控制参数

公开的控制档案保存位置和旋转样条、鞋刀参数、身体尺度与形状系数、原生曝光 PTS 和来源身份。全节目档案也可包含优化状态向量及外部检查点地址。逐文件 SHA-256、字段、dtype 和形状见[参数字段索引](../events/ankara_2026/programmes/okada_mei_sp/body_parameters_1103_fields.json)，当前全节目结果见[节目登记](../events/ankara_2026/programmes/okada_mei_sp/README.md#人体与足鞋结果)。

人体模型模板、形变基、蒙皮权重、面表、完整表面、网络原始输出和广播像素留在 Data。回放依赖单独取得的模型及节目素材。SMPL-X 模型和软件按所用发行版本的条款处理。研究使用、再分发和商业使用的范围见[官方模型条款](https://smpl-x.is.tue.mpg.de/modellicense.html)。字段核查说明文件构成。若使用第三方先验、模型或输出，还须核对相应权利与授权条件。[鞋与冰刀来源](THIRD_PARTY_ASSETS.md#skate-handoff)

## 比赛素材与数值观测

公开仓库不分发 ISU 比赛视频、截图或叠加画面。原片和图像在 Data 保留，公开说明提供[官方来源与源时间](GITHUB_REVIEW.md#官方视频与源时间)；原 UV、PTS、源 SHA、人工事实和数值观测保留。

冻结 review 的 [manifest](../events/ankara_2026/programmes/okada_mei_sp/review/manifest.json)通过 `public_withheld/public_media_derivative` 登记原机位图片、native 截图及叠加画面的排除与派生身份。分类掩膜、项目场地图、观测坐标、方差、状态和统计 W 按各自类型保存，不因为文件名或含色值测量而当作完整广播图像删除。

观测 JSON／GZ／NPZ 中的颜色采样、剖面、ROI 测量与原 SHA 保持原值，以保留数据和状态合同；这些数据不会因此获得第三方额外许可。小型相机控制数组、广告数值模板及 Ad3／Ad7 数值源库不含广播图像缓冲区。历史模板、导入快照及回执按[归档映射](../events/ankara_2026/scene/archive.json)定位。

安卡拉共同场景公开四套节目的源时钟、相机和共同冰场参数。广告记录包括用于拟合的数值轮廓及有出处的测量。官方视频链接用于定位原片，传播权按具体来源核对。[赛事登记](../events/ankara_2026/scene/current_fit.json) · [YouTube 使用条款](https://www.youtube.com/static?template=terms)

公开包支持代码、模型、数值定义与失败状态审阅。原像素核对、颜色剖面重放及完整镜头视觉验收需要本地原素材。

## 训练与依赖的已知缺口

rink_vision.pt 的 SAM2 基础身份和原许可已有证据，训练输入来自历史广播源 `oxXu1avhrvs` 与项目标注，既有记录中未提供相应授权材料。移出原图不能证明所有训练使用或派生模型公开／商业用途均已获授权；具体训练例外、输出权利与专利范围没有在本核查中形成结论。

[pyproject.toml](../pyproject.toml)的 advanced 声明 quadprog 0.1.13，[发布者元数据](https://pypi.org/project/quadprog/0.1.13/)标 GPLv2+。公开包未附该库二进制，已有实际 import 见环境 smoke；依赖声明与实际组合分发分别判断，适用条件见 [GNU 说明](https://www.gnu.org/licenses/gpl-faq.en.html#IfLibraryIsGPL)。本页记录来源和缺口，不变更项目许可或运行依赖。
