# GTSAM CustomFactor 引用回调原型

`reference_callback.pyd` 是独立编译的 pybind11 模块，导出 `create(noise, keys, callback)`。它创建原生 `gtsam::CustomFactor`，不修改 GTSAM DLL、Python 安装或项目代码。回调中的 `factor` 和 `Values` 用 `py::return_value_policy::reference` 显式传入，避免默认 `std::function` 桥接把整份 `Values` 复制给每个因子。引用只在同步回调期间有效，回调不得保存它们。py::function 由带 GIL 的 shared_ptr 最终 deleter 持有。

该 GTSAM 包的 `JacobianVector` 是 pybind11 `module_local` 类型，独立模块无法直接将其 C++ 指针 `py::cast` 成主模块类。本原型在需要 Jacobian 时创建**真正的** `gtsam.JacobianVector`，按原长度填入矩阵，回调完成后逐项写回原 GTSAM 输出指针；无 Jacobian 时传 `None`。这保留现有 Python 回调接口与数值，但每个 Jacobian 矩阵有额外往返拷贝。不得改变 JacobianVector 长度。

编译使用本机 GTSAM 4.2.2 的 conda 构建配方中精确 host 版本的头文件：Eigen 5.0.1 `h5112557_0`、Boost headers 1.90.0 `h57928b3_1`、pybind11 3.0.3 `pyhc8003f9_0`；另用相邻 `tbb_control/include` 和导入库。开发头仅解包到本目录 `vendor/`，未安装到现有环境。Windows MSVC 14.44、Python 3.11、Boost shared_ptr holder 均与 GTSAM 对齐。`build.bat` 可在此目录重新编译。

新进程中先 `import gtsam`，再把本目录加入 `sys.path` 并 `import reference_callback`。调用：

```python
factor = reference_callback.create(noise_model, [key], callback)
```

若仅在隔离基准中临时替代构造函数，应先保存原 `gtsam.CustomFactor` 类，加载本模块后再赋值 `gtsam.CustomFactor = reference_callback.create`，且不要在此后把该名字用于 `isinstance` 检查。

`smoke.py` 只创建一个 11 维合成因子并调用 `graph.error` 与 `graph.linearize`。原绑定与本工厂的误差同为 1.25125，Jacobian 和 RHS 最大差均 0；两条路径传给 Python 的 Jacobian 类型都是 `gtsam.JacobianVector`，进程正常退出。结果见 [smoke_report.json](smoke_report.json)。尚未在此目录执行真实节目图或整图求解。

在本目录运行编译命令 `cmd /c build.bat`；构建输入和产物 SHA256 见 [build_manifest.json](build_manifest.json)。已备好的 [synthetic_scaling.py](synthetic_scaling.py) 按原 `round3/callback_scaling.py` 的 600/1800/4000 个独立 11 维因子设置，逐一对照原构造器与 `create` 的 `linearize`、iSAM 插入和估计时间、回调次数及估计值；此脚本由主代理协调串行运行。

主代理已串行完成 `synthetic_scaling.py`、真实小图与合成规模探针。4000极简因子插入和取估计合计约62.6×，10160状态表的真实因子规模探针8.18×；真实253状态小图1.20×且全部残差/Jacobian/求解结果一致。完整范围和限制见 [第三轮报告](../../README.md)，原始数据为 `scaling_report.json`、`../../batch/reference_results.json`、`../../batch/reference_padding_results.json`。这些均不是完整节目求解。
