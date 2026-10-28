# 历史 GTSAM 引用回调接口

本目录保留 GTSAM 4.2.2 的 CustomFactor 引用回调构建定义，属于[历史原生支持](../README.md)，不是当前 native LM 的运行入口。原始实验说明已归档，原件与公开文档身份见[来源清单](../source_preservation_manifest.json)。

## 接口

`reference_callback.pyd` 导出 `create(noise, keys, callback)`，创建原生 `gtsam::CustomFactor`。同步回调中的 factor 和 Values 使用引用传递，引用只在回调期间有效，不能保存后继续使用。

在已准备匹配模块的环境中，先导入 GTSAM，再加载该扩展：

~~~python
import gtsam
import reference_callback

factor = reference_callback.create(noise_model, [key], callback)
~~~

GTSAM 的 JacobianVector 为 module_local 类型。扩展在需要导数时创建对应的 `gtsam.JacobianVector`，按原长度填入矩阵，再逐项写回 GTSAM 输出；无 Jacobian 请求时传入 None。此路径保留矩阵往返拷贝，回调不能改变容器长度。

## 构建与复现范围

[build.bat](build.bat)和[build_manifest.json](build_manifest.json)保留历史构建方法及身份：MSVC 14.44、Python 3.11、GTSAM 4.2.2，配套 Eigen 5.0.1、Boost 1.90.0、pybind11 3.0.3 与 oneTBB。

脚本中的本机路径和 vendor 布局属于原实验环境。公开目录不附编译扩展、完整 vendor 或 smoke／scaling 报告；构建定义与历史测量不能证明当前环境可直接运行，也不代表完整节目定位通过。当前环境入口见[环境说明](../../../README.md)。
