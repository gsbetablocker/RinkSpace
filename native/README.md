# 原生共同数学核心：当前实现边界

当前库是 **ABI 2**，实现共同几何、静态及完整时变相机前向，包括预测行隐式滚动快门（RS）。Python 薄绑定调用同一份 C++ 数学，原片回放和广告安装展示已使用这些能力；prepare 负责输入合同和数据准备。尚无 GPU 残差/导数批处理、LM 优化器或 solve/resume 生命周期；回放、构建和结构图均不表示新拟合。

## 已实现

- F/R 共列、配对共行、四行等距及上下等边距；八组靶圈共用四层半径。Lane 等宽、右边共线，每行仅三条实体边，无左封口；连接中线的绘制支持独立，并严格停在右边线 N 左侧。
- C0 在原点；B1/B2/C 和两门线平行短边，端点由同一冰场轮廓求交。A1–A4 圆心上下、左右对称；半椭圆蓝区及梯形复用同一围挡。
- 有限线线/线圆交点及固定分支；通用实体路径采样、固定节点与完整模型预览。
- 与 d78 同族的极坐标超椭圆连续围挡，独立 `rx/ry` 与指数；弧长缓存、Gauss 求积及单调反解。上下沿共用 XY，未缠绕全局弧长可跨几何段及闭合接缝。
- `Xc=R(X-C)`、Brown 畸变及世界点解析投影导数；完整时变前向含三次 pan/tilt/log-f 控制、云台轴几何、非零杆臂、随变焦变化的畸变/入瞳，以及预测行隐式 RS。
- 广告 material UV → 展开 `(s,z)` → 连续围挡 XYZ → 同一相机。kind5 及 `sample_print_chart` 共用展开变换，允许广告跨弯角；`print_square_vertices` 在展开面中生成实际正方形外框。
- 调用方分配 C ABI 数组；批量前向保留逐点错误状态，失败点不会钳回有效像素。完整签名、参数布局和状态在 `include/rinkspace/core_api.h`，薄绑定在 `src/rinkspace/core/native.py`。

CMake 当前仅构建 C++17 CPU 库，无第三方数学运行依赖。头文件的 CUDA 标记不代表已经实现或验证 GPU 求解。尺寸来自调用方的初始化参数，尚未重新拟合，也不是已验证米制；赛事、源曝光及观测准入由上层合同管理。

## 构建与检查

使用已有 Python、CMake 与编译工具链，构建目录置于源码树外：

```powershell
python scripts/build_core.py --out D:\CodexData\RinkSpace_Data\build\core
```

可用 `--cmake <现有cmake.exe>` 指定工具。输出 `build.log/build.json`，后者登记实际库路径及 `cuda_built=false`、`nonlinear_solver_implemented=false`。Windows/MSVC 构建路径已实际使用；脚本不安装工具链。

也可直接构建并运行原生检查：

```powershell
cmake -S native -B D:\CodexData\RinkSpace_Data\build\core -A x64
cmake --build D:\CodexData\RinkSpace_Data\build\core --config Release
ctest --test-dir D:\CodexData\RinkSpace_Data\build\core -C Release --output-on-failure
```

原生检查验证数学和状态合同，不代替完整原片的视觉审核。已实现 CLI、输入输出与待实现求解合同见 [INTERFACES.md](../docs/INTERFACES.md)。
