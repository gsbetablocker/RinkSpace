# 公开分发与分项许可

本页汇总已有许可核查及分发范围。基础核查记录于 2026-10-19，后续场景数值资产沿用相同边界；不是对全部私有历史、已安装环境或所有用途的许可认证。项目采用[分项许可与权利范围声明](../LICENSE.md)，第三方来源见 [THIRD_PARTY_ASSETS](THIRD_PARTY_ASSETS.md)。

## 项目与第三方内容

项目自有且未另行许可的内容目前不额外授予使用、修改、分发、再许可或商用许可；第三方原有授权继续有效，不能将整个仓库改标 MIT。

项目辅助代码和场地图的生成／适配身份保存在来源记录中。第三方标志、图稿、源码和权重的权利不会因数值参数化、裁切或项目声明而改变。完整广告 SVG 原件不随公开树分发，来源 URL、文件身份与历史地址仍保留。

| 范围 | 已登记证据与边界 |
|---|---|
| SAM2 基础权重 | `facebook/sam2.1-hiera-tiny`，固定修订 `de431c4043854a71d8101e17995dfe596bf101a5`；下载收据、本地 SHA 与[固定提交 LFS](https://huggingface.co/facebook/sam2.1-hiera-tiny/commit/de431c4043854a71d8101e17995dfe596bf101a5)一致，模型卡标 Apache-2.0 |
| rink_vision.pt | SHA256 `26e792f2381a9bd66c484b8cfa38b13993f5c6cf11b52ef84ee22da65041f208`，对应训练 step 9200；162 个编码器张量／108876544 字节与基础权重一致，208 个存储均为模型状态，未发现额外原图缓冲区；[来源与训练记录](../models/rink_vision.provenance.json)、[SAM2 许可](../models/SAM2_LICENSE.txt)保留 |
| GVHMR 补丁 | 固定上游提交 `6ec3ca39336c50492c0fae65fba2fb831fc7d866`；[原许可](../environments/third_party_patches/GVHMR_LICENSE.txt)、补丁和研究引用见[资产说明](THIRD_PARTY_ASSETS.md)，保留上游教育／研究／非营利及衍生修改非商业条件 |
| Ceres／Abseil、SciPy、oneTBB | 原许可证和署名保留；分别适用 BSD、嵌入组件的 Apache 等原条件，项目声明不缩减第三方授权 |
| 项目辅助代码 | ViTPose helper 与 calibration companion 按项目分项声明处理；[helper 来源](../calibration/program_tracking/body_tracking/source_helpers/run_vitpose_h.provenance.json)、[companion 来源](../src/rinkspace/calibration_companion/provenance.json)保留，实际运行依赖仍适用原许可 |
| 外部模型与框架 | GVHMR、HMR2、ViTPose、SMPL-X、YOLO 未打包进公开 Git 内容；RAFT 位于被忽略的本地缓存，PIDNet／SMP／co-tracker 在已核查实现中仅有参考链接 |

## 比赛素材与数值观测

公开仓库不分发 ISU 比赛视频、截图或叠加画面。原片和图像在 Data 保留，公开说明提供[官方来源与源时间](GITHUB_REVIEW.md#官方视频与源时间)；原 UV、PTS、源 SHA、人工事实和数值观测保留。

冻结 review 的 [manifest](../events/ankara_2026/programmes/okada_mei_sp/review/manifest.json)通过 `public_withheld/public_media_derivative` 登记原机位图片、native 截图及叠加画面的排除与派生身份。分类掩膜、项目场地图、观测坐标、方差、状态和统计 W 按各自类型保存，不因为文件名或含色值测量而当作完整广播图像删除。

观测 JSON／GZ／NPZ 中的颜色采样、剖面、ROI 测量与原 SHA 保持原值，以保留数据和状态合同；这些数据不会因此获得第三方额外许可。小型相机控制数组、广告数值模板及 Ad3／Ad7 数值源库不含广播图像缓冲区。历史模板、导入快照及回执按[归档映射](../events/ankara_2026/scene/archive.json)定位。

[YouTube 许可说明](https://support.google.com/youtube/answer/2797468?hl=en)与[使用条款](https://www.youtube.com/static?template=terms)不自动授予观看者服务外再分发权。两条具体视频的实际 CC 字段未取得，不能把平台默认值当作视频许可证据；当前公开范围不分发这些画面。

公开包支持代码、模型、数值定义与失败状态审阅。原像素核对、颜色剖面重放及完整镜头视觉验收需要本地原素材。

## 训练与依赖的已知缺口

rink_vision.pt 的 SAM2 基础身份和原许可已有证据，训练输入来自历史广播源 `oxXu1avhrvs` 与项目标注，既有记录中未提供相应授权材料。移出原图不能证明所有训练使用或派生模型公开／商业用途均已获授权；具体训练例外、输出权利与专利范围没有在本核查中形成结论。

[pyproject.toml](../pyproject.toml)的 advanced 声明 quadprog 0.1.13，[发布者元数据](https://pypi.org/project/quadprog/0.1.13/)标 GPLv2+。公开包未附该库二进制，已有实际 import 见环境 smoke；依赖声明与实际组合分发分别判断，适用条件见 [GNU 说明](https://www.gnu.org/licenses/gpl-faq.en.html#IfLibraryIsGPL)。本页记录来源和缺口，不变更项目许可或运行依赖。
