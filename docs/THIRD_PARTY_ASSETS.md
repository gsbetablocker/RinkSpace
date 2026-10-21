# 第三方源码、模型与现用身份

当前分项许可与权利范围见 [LICENSE.md](../LICENSE.md)；原第三方许可证继续适用。2026-10-19 对当前实际仓库内容的 MIT 适用性核查见 [LICENSING_REVIEW.md](LICENSING_REVIEW.md)。下文主要保留2026-10-11的外部运行来源登记，不能把列出的所有外部权重认定为已随仓库打包。

项目入口见 [README](../README.md)。本登记观察于 **2026-10-11**，机器可读详情见 [third-party-assets.json](../environments/third-party-assets.json)，完整已安装包版本见 [环境快照](../environments/observed-python311-windows.json)。登记范围是人体、视觉路径及历史 GTSAM 原生构建实际引用的外部资产；它不是整个 Python 依赖树的许可审计，也不是新的运行配置。

以下2026-10-11盘点只读取源码、Git 身份、现有元数据和小型许可文件，并保全一个独立 helper 与三处现用源码修改的补丁。未下载模型、安装包、导入模型、训练、编译、改动旁项目或切换现用路径。大模型 SHA-256 继承已有资源记录；本次核对了当前文件存在及字节数，没有重新读取约 5.6 GB 的权重来宣称新的散列验证。

## 2026-10-19 公开范围补充

- SAM2 基础来源及当前派生模型已精确核实，固定修订、SHA、162个编码器张量的逐字节对应、模型修改和既有训练事实见[模型来源记录](../models/rink_vision.provenance.json)；原始[SAM2 许可](../models/SAM2_LICENSE.txt)保留。
- ViTPose helper 与 calibration companion 统一按项目开发内容处理；生成和适配历史分别保存在[helper 来源记录](../calibration/program_tracking/body_tracking/source_helpers/run_vitpose_h.provenance.json)与[companion 来源记录](../src/rinkspace/calibration_companion/provenance.json)。
- 比赛截图与叠加图不随公开包分发，测量记录不逐字段改写；实际范围和许可边界见[许可核查](LICENSING_REVIEW.md)。
- GVHMR 三处本地修改仅为延迟导入 KNN、SimpleVO 和 Wis3D，修改文件和上游提交在原补丁及机器清单登记；保留[原始许可](../environments/third_party_patches/GVHMR_LICENSE.txt)。引用：Zehong Shen 等，*World-Grounded Human Motion Recovery via Gravity-View Coordinates*，SIGGRAPH Asia Conference Proceedings，2024；[论文](https://arxiv.org/abs/2409.06662)、[官方项目](https://zju3dv.github.io/gvhmr)。

## 已在 OneDrive 项目内的连接代码

| 文件 | 作用与仍然存在的外部引用 |
|---|---|
| [body_prior.py](../calibration/program_tracking/body_tracking/body_prior.py) | 原始源时钟的 GVHMR/SMPL-X 接口。infer/materialize 检查 FSrotation 专用解释器前缀，读取外部 GVHMR 和模型，并切换工作目录。 |
| [observations.py](../calibration/program_tracking/body_tracking/observations.py) | 原片人物关联与 ViTPose-H 观测。默认 `--h-code` 是 SkatePoseSingles 的 helper；补检使用 FSpathing 的 YOLO 模型和包目录。 |
| [bootstrap.py](../calibration/program_tracking/runtime/bootstrap.py) | 追加主项目及旁项目 MotionTrail 的 CUDA site-packages。这里读取环境，不足以证明需要 MotionTrail 应用源码。 |
| [vision.py](../src/rinkspace/vision.py)、[vision_network.py](../src/rinkspace/vision_network.py) | 项目视觉训练/网络代码；torch/CUDA 依赖仍需独立定义并验证。 |
| [environment.yml](../calibration/program_tracking/runtime/environment.yml) | 历史 GTSAM 环境起点，包含 conda-forge、Python 3.11、gtsam 4.2.2、NumPy <2、OpenBLAS；不包括全部 native 构建资源。 |

本次另外保全了 [run_vitpose_h.py](../calibration/program_tracking/body_tracking/source_helpers/run_vitpose_h.py) 和 [来源记录](../calibration/program_tracking/body_tracking/source_helpers/run_vitpose_h.provenance.json)。源文件位于旁项目 `SkatePoseSingles/tools/`，在该旁项目 Git 中未跟踪；源/副本 SHA-256 均为 `8da96207e0c5a4f20a6d43aacb3f718d3d86c265716ae1fd23b2b166d1a4c172`。该 helper 属于项目的 AI 辅助开发，来源已由历史创建和修改记录核实，适用根目录分项许可声明。保全副本依赖 NumPy、OpenCV、torch，动态加载 GVHMR 中的 pose builder 与 UDP decoder，并核验固定 H 模型散列。**observations.py 仍指向原路径，尚未切换运行入口。**

## 上游代码与必要的本地修改

| 资产 | 实际观察身份 | 许可证据与重建边界 |
|---|---|---|
| GVHMR | 外部 `FSrotation/cache/g0_gvhmr_runtime/GVHMR`；Git HEAD `6ec3ca39336c50492c0fae65fba2fb831fc7d866`。 | 现用树有三处 tracked 修改，单凭 clone 该提交无法恢复。已保全下述补丁。根 LICENSE 是教育、研究、非营利用途的自定义许可，要求保留声明并对修改开源，含商业用途限制。见[该提交的官方 LICENSE](https://raw.githubusercontent.com/zju3dv/GVHMR/6ec3ca39336c50492c0fae65fba2fb831fc7d866/LICENSE)。 |
| PyTorch3D | 外部同级 `pytorch3d`；Git HEAD `75ebeeaea0908c5527e7b1e305fbc7681382db47`；已安装版本 0.7.8，未观察到 tracked 修改。 | [该提交的官方 LICENSE](https://raw.githubusercontent.com/facebookresearch/pytorch3d/75ebeeaea0908c5527e7b1e305fbc7681382db47/LICENSE) 为 BSD 三条款，另有 `LICENSE-3RD-PARTY`。仅记录根许可不等于完成全部嵌入组件审计。 |
| GVHMR 内的 HMR2、ViTPose、人体工具 | 实际使用的是上述 GVHMR 树内代码。README 致谢 4D-Humans 和 ViTPose-Pytorch；人体工具 README 说明来自 HuMoR 修改。 | 这些组件各自的独立上游修订和适用许可尚未确定，不把原始模型论文或 GVHMR 根 LICENSE 自动等同于全部组件的权限。 |
| smplx Python 包 | 外部 GVHMR venv 中版本 0.1.28；源提交与 wheel 身份未知。 | METADATA 的 License 字段为 UNKNOWN，但随安装包保存的 LICENSE 有非商业研究/教育/艺术、非转让及禁止分发条款；官方[代码许可入口](https://github.com/vchoutas/smplx/blob/main/LICENSE)用于查阅，不代替现用版本身份。 |
| Ultralytics | FSpathing 安装 8.4.131；GVHMR 环境另有 8.2.42。 | FSpathing 分发 METADATA 标记 AGPL-3.0，源码来源为[官方仓库](https://github.com/ultralytics/ultralytics)。实际包选择受进程 import 顺序影响；本轮未执行检测来验证优先级。 |

GVHMR [现用提交](https://github.com/zju3dv/GVHMR/tree/6ec3ca39336c50492c0fae65fba2fb831fc7d866) 的三处修改位于 `hmr4d/utils/geo_transform.py`、`hmr4d/utils/preproc/__init__.py` 和 `hmr4d/utils/wis3d_utils.py`，内容为延后加载 KNN、SimpleVO 和 Wis3D。已保存 [gvhmr-local.patch](../environments/third_party_patches/gvhmr-local.patch)，SHA-256 为 `0b1ff6e654a618ea85bc9eabe2facccf2dbbfd0d1378bae7c16c570c97e0d2f7`；原始提交、当前文件散列和检查结果都在机器清单中。补丁通过对现用树的 `git apply --reverse --check`，这只是只读适用性检查，未应用或撤销任何修改。以后在**独立 checkout** 中恢复精确上游提交，再检查并应用该补丁；完整运行尚未重建验收。

GVHMR `.gitmodules` 还列出 DPVO，但本机该子模块目录为空。[现用安装说明](https://github.com/zju3dv/GVHMR/blob/6ec3ca39336c50492c0fae65fba2fb831fc7d866/docs/INSTALL.md) 把它列为可选；本次检查的 RinkSpace 人体路径使用自有相机输入，未据此新增 DPVO 下载或依赖。

## 模型和上游小型数据

| 资产 | 当前文件大小 | 身份与获取证据 |
|---|---:|---|
| `gvhmr_siga24_release.ckpt` | 163,508,011 字节 | 既有资源记录 SHA-256 `4fae7da2…`，完整值在机器清单；官方安装说明给出 Drive 来源。 |
| HMR2 `epoch=10-step=25000.ckpt` | 2,709,494,041 字节 | 既有资源记录 SHA-256 `2dcf7963…`，同上游模型目录。 |
| `vitpose-h-multi-coco.pth` | 2,549,075,546 字节 | 既有资源记录 SHA-256 `50e33f40…`，helper 的硬校验使用同一完整散列。 |
| `SMPLX_NEUTRAL.npz` | 108,752,058 字节 | 既有资源记录 SHA-256 `37602144…`；从[SMPL-X 官方入口](https://smpl-x.is.tue.mpg.de/)按账户许可获取，模型许可单独见[官方条款](https://smpl-x.is.tue.mpg.de/modellicense.html)。 |
| FSpathing `yolo11n-pose.pt` | 6,255,593 字节 | 本轮实际计算完整 SHA-256；文件名、字节身份和包版本已知，原始下载记录与上游 assets 精确 release 未确定。模型族入口见[官方 YOLO11 文档](https://docs.ultralytics.com/models/yolo11/)。 |

GVHMR、HMR2、ViTPose 权重的历史记录来自 `FSrotation/artifacts/v03/pretrained_body_pilot/run01/g1/`，实际下载源为固定 Hugging Face 镜像修订 `21b32d5389e2e59c0737d4c4095bbc0b8c23f66b`，来源链和具体文件 URL 已登记。历史记录用镜像 Git LFS SHA-256 核验了下载字节，**没有官方 Drive 发布者散列**。这与“官方已认证相同字节”是不同证据，不能混写；本轮也未重新下载这些链接。GVHMR 官方安装入口保留为[冻结修订的说明](https://github.com/zju3dv/GVHMR/blob/6ec3ca39336c50492c0fae65fba2fb831fc7d866/docs/INSTALL.md)，没有用 latest 替代现用身份。

上游 `hmr4d` 还包含人体映射、回归器、均值与顶点编号等小型数据；既有资源清单的五项已逐项登记并复核当前散列。它们随精确 GVHMR 源树恢复，不能只复制 `.py` 就认定人体路径齐全。历史 GVHMR 的 `yolov8x.pt` 也登记在机器清单，但不是 observations.py 当前补检所用的 YOLO11 文件。

## 历史 GTSAM 的独立构建资源

| 依赖 | 实际身份与本地证据 |
|---|---|
| GTSAM | conda-forge `gtsam-4.2.2-py311h69872ac_0`，包 URL 与 SHA-256 继承已安装 conda record。conda 记录标 BSD-3-Clause，Python METADATA 写 Simplified BSD，差异如实保存；未审计该二进制的全部组件。项目入口见[官方 GTSAM 仓库](https://github.com/borglab/gtsam)。 |
| Eigen headers | Data vendor 目录 `eigen-5.0.1-h5112557_0`，`Eigen/Core` 带 MPL-2.0 声明；完整包来源/散列和其他文件许可尚未核验。 |
| Boost headers | Data vendor `libboost-headers-1.90.0-h57928b3_1`，现有 `LICENSE_1_0.txt` 为 BSL-1.0，与[官方文本](https://www.boost.org/LICENSE_1_0.txt)入口对应。 |
| pybind11 headers | Data vendor `pybind11-global-3.0.3-pyhc8003f9_0`，随包 LICENSE 为 BSD 三条款；精确源提交与 archive 散列未知。 |
| oneTBB | 现用 conda runtime `tbb-2023.1.0-hdcfe883_0`；独立 headers/import libraries 的 `version.h` 为 2023.1.0/interface 12190。现有 README 记录[官方 v2023.1.0 Windows 发布包](https://github.com/uxlfoundation/oneTBB/releases/tag/v2023.1.0)来源，LICENSE 为 Apache-2.0；本轮未下载或验证 ABI。 |

这些路径仍位于 Data，历史 `build.bat` 硬编码了它们、MSVC 和 conda prefix。环境文件并不能自动补全 vendor headers、导入库与已认证 DLL。完整编译身份及主求解器的限制另见[环境说明](../environments/README.md)与[原生 recipe 快照](../environments/native-build-recipes-observed.json)。

## 仍未完成的可重建与公开边界

- 完整 GVHMR/PyTorch3D 上游树和外部环境尚未作为本项目可重建依赖落地；helper 与补丁已保全，现用路径仍保持原状。
- 当前 GVHMR 安装的 Python/CUDA/PyTorch3D 栈与冻结上游 requirements 不同；上游定义含 Linux wheel，不能直接视为现用 Windows 环境重建方案。独立新环境仍需安装与运行核验。
- 模型和人体软件的许可、嵌入组件的声明，以及 RinkSpace 自有源码的许可证选择分别处理。本清单记录事实与缺口，不代表已取得再分发权，也不替项目所有者选择 LICENSE。

本轮网络查阅只使用项目官方仓库、官方文档与模型发布方站点。Eigen 网页与 pybind11 v3.0.3 网页本次读取失败，因此仅依据本机现有头文件/许可文本登记；失败状态保存在机器清单中。
