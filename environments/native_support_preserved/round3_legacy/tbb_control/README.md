# 历史 oneTBB global_control 封装

本目录保留独立 oneTBB global_control 实验的源码、构建定义与[原许可](LICENSE.txt)。它属于[历史原生支持](../README.md)，不增加当前 native LM 的生产依赖。原始实验说明已归档，身份见[来源清单](../source_preservation_manifest.json)。

## 接口

[tbb_control.cpp](tbb_control.cpp)导出：

- `int tbb_control_set_max_allowed_parallelism(size_t value)`：返回 0 表示成功；value=0 释放本 DLL 创建的限制。
- `size_t tbb_control_active_max_allowed_parallelism()`：返回 oneTBB 当前有效的并行上限。

调用时先在匹配环境中导入 GTSAM，确认其实际加载的 oneTBB 运行库，再加载封装 DLL 并在建图前设置限制。头文件、导入库和运行时须匹配；封装不替换进程中的 oneTBB 运行库。

## 构建与来源

[build.bat](build.bat)构建 `tbb_control.dll`。历史环境使用 oneTBB 2023.1.0、interface 12190，头文件与 x64 VC14 导入库来自[官方 v2023.1.0 Windows 发布包](https://github.com/uxlfoundation/oneTBB/releases/tag/v2023.1.0)，具体文件身份由来源清单保存。

公开目录不附编译库、vendor 或原 smoke 报告；原脚本的绝对路径描述历史环境。原实验的验证结果不等于当前机器重建或完整节目求解验证，当前依赖见[环境说明](../../../README.md)。
