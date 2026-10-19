# 独立 oneTBB global_control 实验封装

此目录只用于新启动的 `gtsam-env/python.exe` 基准进程。没有修改 GTSAM 环境、RinkSpace 项目或正在运行的主进程，也没有运行整图。

GTSAM 的真实运行时为 `gtsam-env/Library/bin/tbb12.dll`，其 conda 元数据版本是 **2023.1.0**，SHA256 为 `05545b5d5ba2fc0306c5f0a287002f069f02f4f7a861ab4fa9c913fc96ccc66b`。官方 [oneTBB v2023.1.0 Windows 发布包](https://github.com/uxlfoundation/oneTBB/releases/tag/v2023.1.0) 的 SHA256 与发布页给出的 `cf6ee0c600fcb5c3a9b65e3e6e4781669d06f1bb1e37970d145fcde08eed8da9` 一致。这里仅提取该包的头文件、x64 VC14 导入库和许可证；编译时使用官方 `global_control` 类型及函数声明，没有手写 TBB ABI 结构。

`build.bat` 构建 `tbb_control.dll`。它导出：

- `int tbb_control_set_max_allowed_parallelism(size_t value)`：返回 0 表示成功；`value=0` 释放进程内由本 DLL 创建的限制。
- `size_t tbb_control_active_max_allowed_parallelism()`：返回 oneTBB 的当前有效限制。

新进程应先 `import gtsam`，确认 `tbb12.dll` 来自该环境，再加载本 DLL 并在建图前设置；不要复制或优先加载发布包中的另一份运行时。Python 调用形式：

```python
import ctypes
import gtsam
lib = ctypes.CDLL(r"D:/CodexData/RinkSpace_program_tracking_tmp/solver_acceleration/round3/native/tbb_control/tbb_control.dll")
lib.tbb_control_set_max_allowed_parallelism.argtypes = [ctypes.c_size_t]
lib.tbb_control_set_max_allowed_parallelism.restype = ctypes.c_int
lib.tbb_control_active_max_allowed_parallelism.argtypes = []
lib.tbb_control_active_max_allowed_parallelism.restype = ctypes.c_size_t
assert lib.tbb_control_set_max_allowed_parallelism(1) == 0
assert lib.tbb_control_active_max_allowed_parallelism() == 1
# Run only the separately authorized benchmark in this new process.
assert lib.tbb_control_set_max_allowed_parallelism(0) == 0
```

`verify.py` 是无图、无求解的独立进程自测，见 [smoke_report.json](smoke_report.json)：实际加载路径和模块 handle 在加载封装前后相同；运行时与头文件 interface version 都是 12190；有效限制 20 → 1 → 20。自测命令：

```powershell
& 'D:/CodexData/RinkSpace_program_tracking_tmp/gtsam-env/python.exe' -B 'D:/CodexData/RinkSpace_program_tracking_tmp/solver_acceleration/round3/native/tbb_control/verify.py'
```
