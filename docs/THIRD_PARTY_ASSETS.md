# 第三方资产与来源

项目分项许可见 [LICENSE](../LICENSE.md)，公开边界见 [LICENSING_REVIEW](LICENSING_REVIEW.md)。第三方源码、权重、嵌入数据与项目辅助代码分别保留原来源和许可，不能把所有资产认定为随仓库分发或适用同一许可。

## 当前资源位置与身份记录

当前视觉与人体运行使用 Data 中的 GPU 环境、通用权重和人体上游资源；旧逻辑地址 `.venv-gpu`、`.venv-body`、`models/cache` 和 `environments/body` 通过存储映射解析。位置、包版本与隔离检查见[环境说明](../environments/README.md)、[原 GPU 本机记录](../environments/gpu-local-validation.json)及[人体上游身份](../environments/body-upstream-provenance.json)。

[third-party-assets.json](../environments/third-party-assets.json)与[观察版本](../environments/observed-python311-windows.json)是 2026-10-11 的历史盘点，保留迁移前地址和当时元数据，不能作为今天的运行配置。历史大模型 SHA 继承原资源记录；原盘点的存在性／大小检查不等于重新散列全部权重。

| 资源 | 身份与许可证据 | 边界 |
|---|---|---|
| SAM2／rink_vision.pt | [模型来源记录](../models/rink_vision.provenance.json)、[SAM2 原许可](../models/SAM2_LICENSE.txt) | 固定基础修订、编码器逐字节对应与既有训练事实；派生模型不重标 MIT |
| GVHMR | 上游提交 `6ec3ca39336c50492c0fae65fba2fb831fc7d866`，[固定许可](https://raw.githubusercontent.com/zju3dv/GVHMR/6ec3ca39336c50492c0fae65fba2fb831fc7d866/LICENSE)与[本地许可副本](../environments/third_party_patches/GVHMR_LICENSE.txt) | 自定义教育／研究／非营利及衍生修改非商业条件；本地三处补丁另行保留 |
| PyTorch3D | 上游提交 `75ebeeaea0908c5527e7b1e305fbc7681382db47`，观察版本 0.7.8，[固定许可](https://raw.githubusercontent.com/facebookresearch/pytorch3d/75ebeeaea0908c5527e7b1e305fbc7681382db47/LICENSE) | BSD 三条款及 LICENSE-3RD-PARTY；根许可不代替全部嵌入组件核查 |
| FootMR | 提交 `530da95641f906b42ba6792df8e55c1dd667f98d`；[固定来源与文件 SHA](../environments/body-runtime-sources.json)、[固定许可](https://raw.githubusercontent.com/twehrbein/FootMR/530da95641f906b42ba6792df8e55c1dd667f98d/LICENSE) | 上游文件保留在 Data；自定义研究／非营利及 GVHMR 衍生条件。本地 namespace 接线单独记录 |
| CoTracker | 提交 `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`；[固定来源与文件 SHA](../environments/body-runtime-sources.json)、[固定许可](https://raw.githubusercontent.com/facebookresearch/co-tracker/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/LICENSE.md) | 上游源码与权重留 Data；所选源码许可为 CC BY-NC 4.0，不额外授予商业或分发权利 |
| RTMW／RTMLib | [现有权重来源、归档与 ONNX SHA](../environments/body-runtime-sources.json) | 外部推理依赖；源码、模型及环境不随本次公开树分发，分别适用上游条款 |
| HMR2／ViTPose／人体工具 | 由 GVHMR 树使用，具体来源见机器清单 | 独立上游修订与全部许可尚有缺口，不自动继承 GVHMR 根许可 |
| smplx Python 包 | 历史观察版本 0.1.28；随安装包保存的 LICENSE，另见[官方代码许可](https://github.com/vchoutas/smplx/blob/main/LICENSE) | METADATA 为 UNKNOWN，随包条款含非商业、非转让及禁止分发条件，精确 wheel 身份未确定 |
| Ultralytics | 历史环境分别观察到 8.4.131／8.2.42，[官方仓库](https://github.com/ultralytics/ultralytics) | 相应 METADATA 标 AGPL-3.0；当前进程版本由所选专用环境确定 |

## 项目连接代码与补丁

[body_prior.py](../calibration/program_tracking/body_tracking/body_prior.py)连接 GVHMR／SMPL-X，[observations.py](../calibration/program_tracking/body_tracking/observations.py)连接 ViTPose／YOLO，[bootstrap.py](../calibration/program_tracking/runtime/bootstrap.py)解析本机 GPU 环境。旧盘点中的 FSrotation、FSpathing、SkatePoseSingles 和 MotionTrail 地址是迁移来源，当前入口按上述本机配置解析。

[run_vitpose_h.py](../calibration/program_tracking/body_tracking/source_helpers/run_vitpose_h.py)及[来源记录](../calibration/program_tracking/body_tracking/source_helpers/run_vitpose_h.provenance.json)保留 helper 身份，原副本 SHA256 为 `8da96207e0c5a4f20a6d43aacb3f718d3d86c265716ae1fd23b2b166d1a4c172`。helper 与 [calibration companion](../src/rinkspace/calibration_companion/provenance.json)按项目自有辅助代码登记，默认 helper 已直接使用该仓库副本，Data 中的历史副本字节相同；实际加载的 GVHMR／ViTPose 等依赖仍适用各自条款。

GVHMR 的 `geo_transform.py`、`preproc/__init__.py` 和 `wis3d_utils.py` 修改为延迟导入 KNN、SimpleVO 和 Wis3D。[补丁](../environments/third_party_patches/gvhmr-local.patch) SHA256 为 `0b1ff6e654a618ea85bc9eabe2facccf2dbbfd0d1378bae7c16c570c97e0d2f7`；恢复需匹配上游修订及本地文件身份。相关研究为 Zehong Shen 等，*World-Grounded Human Motion Recovery via Gravity-View Coordinates*，SIGGRAPH Asia 2024：[论文](https://arxiv.org/abs/2409.06662)、[官方项目](https://zju3dv.github.io/gvhmr)。

GVHMR 的 DPVO 子模块在历史盘点中为空，[固定安装说明](https://github.com/zju3dv/GVHMR/blob/6ec3ca39336c50492c0fae65fba2fb831fc7d866/docs/INSTALL.md)将其列为可选。RinkSpace 的对应人体路径使用自有相机输入，不据此认定 DPVO 已安装。

## 权重与小型数据

| 资产 | 登记字节数 | 来源与限制 |
|---|---:|---|
| `gvhmr_siga24_release.ckpt` | 163508011 | 完整 SHA 与 URL 见机器清单 |
| HMR2 `epoch=10-step=25000.ckpt` | 2709494041 | 同上 |
| `vitpose-h-multi-coco.pth` | 2549075546 | helper 使用固定完整 SHA 检查 |
| `SMPLX_NEUTRAL.npz` | 108752058 | [官方获取入口](https://smpl-x.is.tue.mpg.de/)与[模型条款](https://smpl-x.is.tue.mpg.de/modellicense.html)，按账户许可取得 |
| `yolo11n-pose.pt` | 6255593 | 已登记文件 SHA，精确上游 release 与原下载记录仍未确定；[模型族说明](https://docs.ultralytics.com/models/yolo11/) |

GVHMR／HMR2／ViTPose 的历史下载链固定 Hugging Face 镜像修订 `21b32d5389e2e59c0737d4c4095bbc0b8c23f66b`，按镜像 LFS SHA 核验下载字节，缺少官方 Drive 发布者散列，不能写成官方认证相同字节。人体映射、回归器、均值和顶点编号等小型数据也属于完整依赖；仅复制 Python 文件不足以恢复运行。

## 历史 GTSAM 构建资源

| 依赖 | 登记身份与缺口 |
|---|---|
| GTSAM | conda-forge `gtsam-4.2.2-py311h69872ac_0`，URL／SHA 继承 conda record；BSD 元数据差异保留，[官方仓库](https://github.com/borglab/gtsam) |
| Eigen | vendor `eigen-5.0.1-h5112557_0`，Eigen/Core 含 MPL-2.0；完整包身份与全部许可未核验 |
| Boost | vendor `libboost-headers-1.90.0-h57928b3_1`，随包 BSL-1.0，[官方文本](https://www.boost.org/LICENSE_1_0.txt) |
| pybind11 | vendor `pybind11-global-3.0.3-pyhc8003f9_0`，随包 BSD 三条款；源提交与 archive SHA 未确定 |
| oneTBB | runtime `tbb-2023.1.0-hdcfe883_0`，headers 2023.1.0／interface 12190，[官方发布](https://github.com/uxlfoundation/oneTBB/releases/tag/v2023.1.0)，Apache-2.0；历史盘点未验证 ABI |

这些是历史独立构建身份，不是当前 native LM 的完整依赖表。保全源码和清单见[环境入口](../environments/README.md)及[原生 recipe](../environments/native-build-recipes-observed.json)。

## 已知缺口

项目内资源迁移和隔离检查已有记录，空白机器的完整 Windows 人体构建／推理仍未验证。部分上游定义包含 Linux wheel，不能直接当作本机环境锁。各模型、人体组件和嵌入数据的许可闭合程度分别判断；来源登记不授予额外再分发或商业权利。
