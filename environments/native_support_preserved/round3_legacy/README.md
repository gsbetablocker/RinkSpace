# 历史原生支持

本目录保留 reference_callback／oneTBB 实验的五份原始构建、源码及许可文件，并提供对应接口说明。原文件身份、历史地址及原始说明的归档位置见 [source_preservation_manifest.json](source_preservation_manifest.json)。当前环境入口见 [environments/README](../../README.md)。

| 子目录 | 公开内容 |
|---|---|
| [reference_callback](reference_callback/README.md) | GTSAM CustomFactor 引用回调接口、build.bat 与 build_manifest |
| [tbb_control](tbb_control/README.md) | oneTBB global_control 接口、C++ 源码、build.bat 与 LICENSE |

构建文件保留旧编译器、环境、vendor 与输出地址，不是当前机器的即用构建入口。原始 README 字节保存在既有 Data 文档归档中，清单的 archived_documents 分别登记原件和公开说明的身份。原 smoke／scaling／第三轮报告未随公开仓库分发。

`runtime_activated=false`，这些文件不增加 named66 或当前 native LM 的生产依赖。build_manifest 中的 archive、DLL 和 PYD 身份属于原实验记录。oneTBB 的原声明仅适用于所述第三方组件，不授予整个项目额外许可。
