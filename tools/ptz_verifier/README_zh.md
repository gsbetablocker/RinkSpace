# RinkSpace 独立 PTZ 验证器

**版本：0.1.0 · 面向 rebuild@1ce8ae1 · 2026-10-27**

只用已有转播视频里的纹理对应关系，检查一套相机内参能否解释跨帧背景运动。它不读冰场三维几何、不用当前相机解初始化独立分支、不调用原生/GPU 主求解器，也不把参数写回项目。

## 先看哪一个文件

- `REAL_RUN_zh.md`：本次真实 S01 试跑范围、数字与失败条件。
- `METHODS.md`：公式、模型假设、筛查口径、不可测项目。

公开目录仅提供源码、测试、配置示例和历史试跑摘要。原交付的 `evidence/`、图像报告和本机交接文件不随仓库分发；原始交付清单保留在本机 Data 的公开整理快照中。`MANIFEST.sha256` 只核对本目录公开文件，`evidence_provenance.json` 记录原试跑来源，不表示克隆后已取得原素材或完成复现。

**本次真实数据结论是“证据不足”，不是获得了新的相机真值。** 在 26.92–32.92 秒的 31 个曝光上，自由内参模型将 `fx/fy` 推至验证器的 2.0 数值上界；这只说明该候选及当前数据组合不能给出可靠物理结论，绝不证明真实像素比例是 2。

## 安装与运行

Python 3.11+，NumPy、SciPy、带 SIFT 的 OpenCV，以及命令行 `ffmpeg` / `ffprobe`。不需要 CUDA、模型权重、PyAV、联网模型或完整 RinkSpace 环境。先用项目已有的兼容解释器；缺依赖时放入独立环境，不能改坏项目主环境。

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

这里的安装是用户/Codex 显式执行的命令；程序自身不自动安装任何东西。

### 直接读取本次离线审核包

在本工具目录内运行；下面两个路径替换为本机已经存在的审核包与一个全新的 Data 输出目录：

```powershell
python verify.py bundle --bundle "D:\Data\RinkSpace_S01_fit_PRO_1ce8ae1_20261027.zip" --out "D:\Data\ptz_check_S01_001"
```

一条命令完成：校验源身份 → 精确 PTS 抽帧 → 背景特征/整轨迹划分 → 自动连通图 → 独立模型多初值拟合 → 最后才载入原相机作只读对照 → HTML/JSON/CSV。

默认在已有 S01 全区间 22.120–44.960 秒采 116 帧、约 0.2 秒间隔。**这不是逐帧全 S01 标定。** `--component all` 使用所有至少含 4 帧的连通分量；每个分量有独立旋转参考系，仅共享同一物理机位的低维内参规律，不伪造分量间对应。

默认三个候选、每个三个焦距初值、每个最多 180 次目标评估。程序有进度打印和求解预算；按 Ctrl+C 保留已完成诊断，不做自动无限续跑。输出目录已存在会拒绝执行，不能覆盖既有结果。

### 精确复现本次最大连通分量试验

```powershell
python verify.py bundle --bundle "D:\Data\RinkSpace_S01_fit_PRO_1ce8ae1_20261027.zip" --out "D:\Data\ptz_check_S01_largest_001" --component largest --max-nfev 120 --profile-aspect 0.8 1 1.15 1.3571692722 1.6
```

OpenCV/FFmpeg 版本、解码像素和特征实现会影响逐位复现。审核包提供的选中帧 BGR 哈希不匹配时直接报错，不绕过校验。库版本见证据文件。

### 复用已经提取的对应关系

```powershell
python verify.py refit --graph "D:\Data\ptz_check_S01_001" --out "D:\Data\ptz_refit_S01_002" --models square_center square_pp free_pp free_pp_radial --profile-aspect 1 1.15 1.3571692722 1.6 --camera-npz "D:\Data\S01_bundle\current\camera_trajectory.npz"
```

不重新解码、不重新匹配。`--camera-npz` 可省略；省略后只生成独立结果。`--graph` 需要本机已有的 `graph.json` 和 `matches.npz`，这些运行产物不在公开源码中；没有全尺寸源帧时跳过图像预览，仍可对已有数值对应重新拟合。

### 对已有原片的其它单镜头运行

```powershell
python verify.py video --video "D:\Data\official_video.mp4" --start-pts 283136 --end-pts 575488 --time-base 1/12800 --rig-id camera_1 --shot-id S01 --out "D:\Data\ptz_video_S01_001"
```

必须使用该原片的真实原生 PTS 和 time base，不能把秒或帧号直接当 PTS。此入口只接受一个明确连续、单物理机位的镜头；不自动跨转播切镜。若已知镜头存在不同裁切/画幅，应分开处理。本工具不会自动扩大到其它镜头或机位。

默认遮罩是针对 S01 的保守矩形：记分牌、右下计时器与画面中央选手区。它不是人物分割，可能遗漏动态物体，也可能遮去有用背景。其它镜头使用 `--masks` 明确给出相应矩形；格式见 `examples/masks_s01.json`。用 `examples/masks_overlays_only.json` 可以做现有画面上的遮罩敏感性对照，但不能默默选“最有利”遮罩作为最终证据。

## 输出与解释

| 输出 | 含义 |
|---|---|
| `report.html` | 离线对照页、少量自动挑选的真实匹配图、限制与停止状态 |
| `summary.json` | 总结、筛查前后留出误差、触边/退化/未收敛标志 |
| `graph.json` / `matches.npz` | 明确源 PTS、训练/留出轨迹身份、连接关系；可重算 |
| `graph_diagnostics.json` | 拒绝连接的具体原因、未覆盖帧和各分量 |
| `*_fit.npz` / `*_cameras.csv` | 独立候选逐曝光内参和相对旋转；不是可自动写回的物理标定 |
| `*_observability.json` | 局部缩放 Jacobian 奇异值、消去姿态/焦距后的共享参数弱方向 |
| `aspect_profile.json` | 固定 `fx/fy`，重新优化其它参数；不是置信区间 |
| `current_comparison.json` | 原内参+原旋转，以及固定原内参重新估计旋转的两种对照 |
| `run_config.json` / `source.json` | 配置、版本、哈希、解码时间与实际范围 |

独立模型没有米制尺度、相机高度、三脚架杆臂或世界坐标位置。所有相机旋转仅相对于各连通分量的首个曝光。它不能独自证明“光心在 2 米以下”，更不能把 `physical_camera_certified` 改成 true。

“整轨迹留出”只表示当前识别为同一纹理轨迹的观测不跨训练/检验；不是独立采集的新数据，也不能保证碎裂轨迹从未属于同一物理纹理。全部留出点与经过训练单应性筛查的留出子集同时报告，不能只引用筛后指标。

## 与项目的关系

建议作为独立目录 `tools/ptz_verifier/`，由 Codex 适配现有源文件定位。运行输出放本机 Data，不进 OneDrive 项目目录或公开仓库；不要提交比赛画面、解码帧、运行缓存。

本工具实现了独立验证流程，没有修改或重跑生产求解器。此前审核补丁不是它的依赖。接口不兼容时应明确报错，不能悄悄猜测参数列号。
