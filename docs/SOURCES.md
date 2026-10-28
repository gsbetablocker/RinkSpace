# 技术来源

这些资料支持技术选择，项目实际能力见[实现状态](REBUILD.md)，准确性与跨赛事验证见[验证说明](VALIDATION.md)。

| 一手资料 | 使用范围 |
|---|---|
| [OpenCV：平面单应](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html) | 平面映射的适用边界；三维围挡和人体需要各自的空间表示 |
| [FFmpeg：ffprobe](https://ffmpeg.org/ffprobe.html)、[PyAV：时间](https://pyav.org/docs/stable/api/time.html) | 视频元数据、PTS 与时间基准；显示帧率不能直接替代曝光时钟 |
| [TVCalib](https://arxiv.org/abs/2207.11709) | 已知场地线段与相机标定的研究参考 |
| [Rink-Agnostic Hockey Rink Registration](https://arxiv.org/abs/2401.01003) | 跨冰场配准参考；不能直接继承冰球标线作为花滑赛事事实 |
| [BroadTrack](https://arxiv.org/abs/2412.01721)、[作者代码](https://github.com/evs-broadcast/BroadTrack) | 转播相机安装、运动与连续场地追踪的对照 |
| [SAM 2 官方代码](https://github.com/facebookresearch/sam2)、[PIDNet 官方代码](https://github.com/XuJiacong/PIDNet)、[torchvision RAFT](https://pytorch.org/vision/stable/models/raft.html) | 已有视觉模块使用或参考的分割特征、边界结构与光流；实际采用项见任务配置 |
| [LightGlue](https://arxiv.org/abs/2306.13643)、[Efficient LoFTR](https://arxiv.org/abs/2403.04765) | 可选匹配路线；列入来源不代表已经接入或在花滑视频验证 |
| [SciPy：平滑样条](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_smoothing_spline.html) | 平滑方法参考；平滑不等于真实定位或接触恢复 |

实际使用外部代码与权重时记录来源、版本、许可和散列；不同工具的授权不互相替代。研究参考、已接入实现与真实视频验证分别记录。设备规格及安卡拉样本的相机事实见 [PXW-Z190](camera_hardware/PXW_Z190.md)，不作为所有赛事的设备前提。
