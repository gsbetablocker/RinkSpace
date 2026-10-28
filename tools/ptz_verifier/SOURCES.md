# 方法与接口来源

核查日期：2026-10-27。下列为作者论文、作者机构页面、官方软件文档或指定仓库源码。实现代码重新编写，未复制论文作者的软件或第三方权重。

1. S. N. Sinha, M. Pollefeys. *Pan–Tilt–Zoom Camera Calibration and High-Resolution Mosaic Generation*, CVIU, 2006.
   https://snsinha.github.io/pdfs/SinhaCVIU2006.pdf
   使用依据：第2节旋转相机与变焦内参模型、跨图像单应性约束；第3节图像对齐和优化思想。
   不能照搬的前提：低变焦全景与逐级变焦的特定采集；近似旋转中心与光心重合。

2. L. de Agapito, E. Hayman, I. D. Reid. *Self-Calibration of Rotating and Zooming Cameras*, IJCV 45(2), 2001.
   https://www.robots.ox.ac.uk/~lav/Papers/agapito_etal_ijcv2001/agapito_etal_ijcv2001.html
   使用依据：无限远单应性和对内参加约束的必要性；不能所有内参逐帧任意放开。

3. E. Hayman, D. W. Murray. *The Effects of Translational Misalignment when Self-Calibrating Rotating and Zooming Cameras*.
   https://www.robots.ox.ac.uk/~lav/Publications/hayman_murray_ouel225002/hayman_murray_ouel225002.html
   使用依据：光心与转轴不一致会影响焦距/旋转估计；不能无条件接受纯旋转近似。

4. B. J. Tordoff, D. W. Murray. *The impact of radial distortion on the self-calibration of rotating cameras*, CVIU 96 (2004), 17–34.
   https://www.robots.ox.ac.uk/ActiveVision/Papers/tordoff_murray_cviu2004/tordoff_murray_cviu2004.html
   使用依据：旋转自标定对径向畸变的敏感性；独立模型仍需畸变敏感性检验。

5. SciPy 官方 least_squares 文档。
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
   使用方式：有界 TRF、稀疏 Jacobian、LSMR、robust loss；不是全局求解保证。

6. RinkSpace 指定源码接口。
   https://github.com/gsbetablocker/RinkSpace/blob/1ce8ae125c32457ec27ec4d133e6a2e8146386d4/src/rinkspace/core/native.py
   仅读取 evaluate_motion_camera 返回字段合同，未调用或复制生产投影代码。

7. 用户离线包 RinkSpace_S01_fit_PRO_1ce8ae1_20261027.zip。
   使用已有视频、source clock 与当前 camera_trajectory；不使用场地模型作为独立分支的约束。
