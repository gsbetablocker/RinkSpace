# 历史原生支持定义保全

返回 [运行依赖说明](../../../docs/RUNTIME_DEPENDENCIES.md)。逐项来源、SHA-256、字节数与源 stat 稳定性见 [source_preservation_manifest.json](source_preservation_manifest.json)。

2026-10-13 按原始字节保全 Data 中七个小型文件。复制前后核对源 stat，重读原件并逐字节比较副本；原件保持原位。目标已存在时只接受完全相同内容，不覆盖不同内容。

| 子目录 | 保全内容 | 作用与范围 |
|---|---|---|
| reference_callback/ | build.bat、build_manifest.json、README.md | 历史 GTSAM reference-preserving CustomFactor bridge 的构建方法与来源；C++ 主源码已在 calibration/program_tracking/native/reference_callback.cpp |
| tbb_control/ | tbb_control.cpp、build.bat、README.md、LICENSE.txt | 原 README 所述的独立 oneTBB global_control 实验封装和相关发布包声明；不据此增加 named66 的生产后端依赖 |

保全文件内的旧编译器、环境、include/lib 和输出路径均原样保留。这些文件是历史定义，并非可直接在另一台机器运行的构建入口。本次未执行 build、加载原生库、复制 vendor 或编译产物，也没有切换活动运行地址；runtime_activated=false。

reference_callback 的历史 build_manifest 保存当时源码、扩展、GTSAM DLL 与三个 vendor archive 的身份记录。此次只对七个保全文本/源码文件重新计算 SHA，没有重读 archive、DLL、PYD 或环境二进制来认证其当前字节。

tbb_control/LICENSE.txt 随其原 README 记述的官方 oneTBB 发布包头文件、x64 VC14 导入库和许可一起提取；其适用内容与来源以该原 README 及 LICENSE 原文为准。这份第三方声明不为整个 RinkSpace 项目赋予许可证，也不说明其他 vendor 或人体模型的许可。原 README 的历史实验和核验结果保持原文，不表示本次重新执行。
