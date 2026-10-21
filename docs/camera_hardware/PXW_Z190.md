# 安卡拉样本：PXW-Z190 相机证据与接口

本页解释现有安卡拉／冈田样本的设备信息，不限定 [项目总目标](../../README.md) 的赛事或相机型号。其他转播使用自己的设备与校准记录。

## 事实与未知项

型号信息来自用户转述摄制团队的回复，原始邮件未进入项目；用户将 Z190 关联至长边主、副机位和低位端部。三个安装独立，序列号和现场设置没有逐台记录。高位机位及其他赛事不继承此关联。

- [型号注册表](../../configs/camera_hardware/sony_pxw_z190.json) 保存厂家能力。
- [样本机位关联](../../configs/camera_hardware/okada_sp_installations_z190_v1.json) 保存来源、安装身份与历史散列。
- 用户转述原记录（本地保存，未随仓库分发）：`D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/camera_hardware_20261001/user_report.md` 保留原证据等级。
- [样本现行求解输入](../../configs/solver_runtime/okada_named66.json) 指向实际采用的机位安装与围挡外位置事实；关联表中的旧记录不代表最新运行采用值。

厂家给出的固定镜头为 25 倍光学变焦，物理焦距 3.7–92.5 mm，35 mm 等效范围 28.8–720 mm；“1/3 型”三片 CMOS 不等于已测得感光宽度。[Sony 规格](https://www.sony.jp/xdcam/products/PXW-Z190/spec.html)

设备支持镜片位移防抖与 HD 数字扩展。手册的三脚架防抖建议不能当成现场开关证据；实际防抖、数字扩展、输出裁切、有效感光窗口、畸变、滚动快门读出和曝光设置仍分别记录为未知或假设。[产品页](https://pro.sony/ue_US/products/handheld-camcorders/pxw-z190)、[操作手册](https://pro.sony/support/res/manuals/4740/1bf6ebbd9def311f789db07687817956/47407001M.pdf)

## 几何与时钟边界

有效焦距使用原视频像素单位。毫米焦距、光学投影、数字扩展、输出缩放和裁切不能混成同一个已知量；未知现场设置不自动给出主点或逐帧自由偏移。

固定三脚架／云台支点不等于固定光心。使用 [物理相机接口](../../calibration/program_tracking/z190_physical_camera.py) 时，各台安装偏移与光学参数独立，光心按姿态和实际启用的偏移计算；不能因为同型号就共享全部数值。

当前节目处理使用 ISU 官方 1920×1080、50 fps 单节目原片，time_base=1/12800；第一镜头为源时间 20.020–44.960 秒的 1,248 个原生曝光。旧全场文件的 60 Hz 显示与重复曝光对应记录仅保留作历史，不再作为当前节目处理时钟。

用户在装配记录中明确要求“第一版先把偏移视为不随变焦变化，跑出结果后再看是否需要加入变焦项”，这不是永久固定光学参数的要求。第一镜头的显式 `shared_zoom_camera` 绑定现可启用 [共享变焦接口](../../calibration/program_tracking/first_shot_zoom_camera.py)：主点横纵、k1、轴向光心四个共享斜率，统一通过原物理相机、行曝光求解和绘制路径。有效像素焦距是条件驱动，并非镜头编码器测量；系数由本赛事原片估计，不能声称是 Sony 标定值。相机仍有同一固定支点，各曝光保留姿态与有效焦距。启用或测试通过均不代表尾段已贴合，实际修复状态见 [节目审阅快照](../../events/ankara_2026/programmes/okada_mei_sp/review/README.md)。

## 复用与历史

[camera_hardware_metadata.py](../../calibration/program_tracking/camera_hardware_metadata.py) 提供 load_metadata() 与 calibration_context()：验证源身份与安装关联，返回设备上下文，不生成标定系数。其默认镜头表与部分注册记录仍含迁移前路径，调用者需要显式提供并核对真实输入；本页不声称该旧默认加载器已完成迁移。

旧设备候选线索（本地保存，未随仓库分发）：`D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/history/tracking_runs/solver_acceleration/round7_repair/before/camera_hardware_leads.md` 保留失败历史。旧共享 K4／零 Brown 对照出现副机位退步及异常 skew，不能称为已接受的设备标定。组件投影或导数检查通过，也不等于现场设置已确认或完整节目已贴合。
