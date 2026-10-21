# RinkSpace 许可与权利范围声明

本仓库采用分项许可：项目开发内容按本声明处理，第三方源码、模型、数据和素材分别适用各自的许可证或授权条件。

## 项目开发内容

项目开发代码（含 AI 辅助开发）按同一许可声明处理。对于项目作者依法享有权利、且未另行许可的内容，作者保留其权利；本声明不额外授予使用、修改、分发、再许可或商业利用的许可。已经存在的有效授权继续有效。


## 第三方内容

第三方内容继续适用原权利人的许可证和授权条件，包括相应的版权、归属、修改说明、免责及其他保留声明。第三方许可证已经授予的权利不受本声明撤回或限制；涉及这些内容时，以对应的有效原始许可及授权条件为准。

主要第三方范围及原许可证如下：

| 范围 | 对应许可与说明 |
|---|---|
| Ceres 头文件、相关补丁及嵌入组件 | 保留 [cg22 原许可](environments/cg22/LICENSE)与[cg22p 原许可](environments/cg22p/LICENSE)，以及各文件中的 BSD、Apache 等原有声明 |
| SciPy 派生滤波实现 | 保留 [SCIPY_LICENSE.txt](calibration/rinkfit/cpp/source_image_gpu/SCIPY_LICENSE.txt)及源码归属说明 |
| oneTBB 相关保全材料 | 保留 [Apache 原许可](environments/native_support_preserved/round3_legacy/tbb_control/LICENSE.txt)；外部运行依赖按其各自条件处理 |
| GVHMR 衍生补丁 | [gvhmr-local.patch](environments/third_party_patches/gvhmr-local.patch)适用[原始 GVHMR 许可](environments/third_party_patches/GVHMR_LICENSE.txt)，其中的教育、研究、非营利及衍生修改非商业限制继续有效 |
| SAM2 上游部分与派生权重 | SAM2 上游部分保留 [Apache-2.0 原文](models/SAM2_LICENSE.txt)。[rink_vision.pt](models/rink_vision.pt)的固定基础来源、字节核验与修改见[来源记录](models/rink_vision.provenance.json)，训练材料与使用边界见[许可核查](docs/LICENSING_REVIEW.md)；本声明不把完整派生权重重新许可为 MIT |
| 视频、截图、用户提供素材、标注及其他研究数据 | 独立比赛截图和叠加图不随公开仓库分发；原有测量记录保留；项目图、标注及数值观测按各自来源判断。软件许可证不自动覆盖其他第三方材料 |

可选软件、模型和运行依赖也分别适用其原许可；项目自有接口或本声明不改变这些条件。分发组合程序时，还需满足实际适用的第三方条件。

## 未确认授权与既有权利

来源、权属或公开再分发授权尚未确认的内容，不在本项目授予的许可范围内。本声明不表示该等内容已经获准公开分发，不能替代原权利人的授权，也不授予超出原许可范围的权利。

本声明不限制适用法律、GitHub 平台条款或其他有效授权已经赋予的权利。

具体来源、范围、原许可证位置及未解决事项见 [LICENSING_REVIEW.md](docs/LICENSING_REVIEW.md)和[第三方资产登记](docs/THIRD_PARTY_ASSETS.md)。
