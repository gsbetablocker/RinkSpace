# 来源与复现

本页描述 rinkfit 历史后端的模块合同，适用范围见[模块入口](../README.md)；当前 rebuild 的 native／GPU 能力见[实现状态](../../../docs/REBUILD.md)。

初始包的数学方案原保存在 `spec/SPEC.html`，其中包含 US11176706B2、WACV 2020/2022、Javan 论文/演讲及 Ceres/OpenCV 等一手资料的核对与适用边界。`spec/source_review.json` 保存资料逐项用途。这些 spec 文件没有随当前公开源码提供，输入范围见[模块 README](../README.md)。本模块重新实现数学接口，没有引入 sportsfield_release 的网络代码、权重或训练资产；也没有移植专利正文作为项目代码。

项目代码适用[根许可声明](../../../LICENSE.md)。NumPy／SciPy、Ceres／Eigen／SuiteSparse 分别适用原依赖许可，本目录不分发这些库的二进制。

## 数据与旧候选

`data/input_manifest.json` 是五个原始数组的字节身份。`seeds.json` 的几何和相机都来自已审核的历史运行快照，始终是条件初值。世界 L=1 只是规范；原子来源不能证明历史相机或形状已经正确。

`tools/build_inputs.py` 可从完整审核 bundle 重建输入（bundle 顶层具有 `repo/` 与 `data/`）。所需只读数学解码器 `tools/frozen_camera_math.py` 已一并带入。该描述针对原完整启动包；当前公开源码不附完整 data/，使用前须取得并核对相应材料。若重建，随后用 `bootstrap` 模块重新生成 bootstrap_state 并建立新的 manifest/run，不能混用旧校验值。

```bash
python tools/build_inputs.py /path/to/review_bundle --out /path/to/new_input
PYTHONPATH=src python -m rinkfit_start.bootstrap --data /path/to/new_input --output /path/to/new_input/bootstrap_state.json
```

完整来源 packet、grouped W、profile 和 verified order 按 event_integration.md 从现有本地工程读取；未包含视频和全部历史缓存。
