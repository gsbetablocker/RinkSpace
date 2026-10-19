# RinkSpace 许可与权利范围声明

本仓库采用分项许可与权利范围声明。它包含项目原创代码、第三方组件及研究数据和素材，各部分分别适用其明确标注的许可证或有效授权条件；本声明不授予覆盖整个仓库的 MIT、GPL、AGPL 或其他统一开源许可，也不对所有第三方内容施加统一的非商业限制。

## 项目自有内容

仅对于项目作者确实拥有相应权利、且未另行明确许可的内容，作者保留其依法享有的权利。本声明本身不额外授予使用、修改、分发、再许可或商业利用的许可。已经存在的有效授权不因本声明而撤回或缩减。

本声明不声称全部仓库内容的版权归项目作者所有。目录位置、文件可见、来源保全、人工标注或加入 Git，均不等于证明作者权属或取得再许可权。

## 第三方内容

第三方内容继续适用原权利人的许可证和授权条件，包括相应的版权、归属、修改说明、免责及其他保留声明。第三方许可证已经授予的权利不受本声明撤回或限制；涉及这些内容时，以对应的有效原始许可及授权条件为准。

以下是当前主要范围的导航，不替代原许可证，也不表示已经完成全部组件的分发许可核查：

| 范围 | 对应许可与说明 |
|---|---|
| Ceres 头文件、相关补丁及嵌入组件 | 保留 [cg22 原许可](environments/cg22/LICENSE)与[cg22p 原许可](environments/cg22p/LICENSE)，以及各文件中的 BSD、Apache 等原有声明 |
| SciPy 派生滤波实现 | 保留 [SCIPY_LICENSE.txt](calibration/rinkfit/cpp/source_image_gpu/SCIPY_LICENSE.txt)及源码归属说明 |
| oneTBB 相关保全材料 | 保留 [Apache 原许可](environments/native_support_preserved/round3_legacy/tbb_control/LICENSE.txt)；外部运行依赖按其各自条件处理 |
| GVHMR 衍生补丁 | [gvhmr-local.patch](environments/third_party_patches/gvhmr-local.patch)适用[原始 GVHMR 许可](environments/third_party_patches/GVHMR_LICENSE.txt)，其中的教育、研究、非营利及衍生修改非商业限制继续有效 |
| SAM2 上游部分与派生权重 | SAM2 上游部分保留 [Apache-2.0 原文](models/SAM2_LICENSE.txt)。[rink_vision.pt](models/rink_vision.pt)的固定基础来源、字节核验与修改见[来源记录](models/rink_vision.provenance.json)，训练材料与使用边界见[许可核查](docs/LICENSING_REVIEW.md)；本声明不把完整派生权重重新许可为 MIT |
| 用户确认由 ChatGPT 生成的辅助代码 | ViTPose helper 和 calibration companion 的生成来源已由用户确认，来源记录保留；适用项目分项声明及实际第三方条件，见[许可核查](docs/LICENSING_REVIEW.md) |
| 视频、截图、用户提供素材、标注及其他研究数据 | 独立比赛截图和叠加图不随公开仓库分发；原有测量记录保留；项目图、标注及数值观测按各自来源判断。软件许可证不自动覆盖其他第三方材料 |

可选软件、模型和运行依赖也分别适用其原许可；项目自有接口或本声明不改变这些条件。分发组合程序时，还需满足实际适用的第三方条件。

## 未确认授权与既有权利

来源、权属或公开再分发授权尚未确认的内容，不在本项目授予的许可范围内。本声明不表示该等内容已经获准公开分发，不能替代原权利人的授权，也不授予超出原许可范围的权利。

本声明不限制适用法律、GitHub 平台条款或其他有效授权已经赋予的权利。

具体来源、范围、原许可证位置及未解决事项见 [LICENSING_REVIEW.md](docs/LICENSING_REVIEW.md)和[第三方资产登记](docs/THIRD_PARTY_ASSETS.md)。
