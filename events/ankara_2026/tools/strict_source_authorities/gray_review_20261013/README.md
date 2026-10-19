# 灰色材料审阅的冻结源码保全

这里保存 3 份原先仅位于 Data 的冻结 authority 源码版本：一个源测量 reader 和两份 predecessor。原始文件及既有 Data 快照继续保留；出处、文件状态、SHA-256、静态依赖和既有冻结快照路径见 [source_preservation_manifest.json](source_preservation_manifest.json)。3 个副本共 90,109 字节，逐字节匹配原件，也匹配既有 `exact_source_authority_relocation_receipt.json` 声明的散列。

目录保留了 reader 与 `model_context_pre_single_gray_support_20261013/` 的相对位置。reader 的 `--compare-default-predecessor` 分支显式加载这两份 predecessor，分别使用 `rinkspace._gray_source_saved_predecessor` 和 `rinkspace._gray_source_saved_guides_predecessor` 模块名；它们的相对导入依赖已有 OneDrive `src/rinkspace/` 代码。reader 的其他明示导入也指向已有项目代码、标准库或第三方环境。本次没有复制该 authority 包里另外 353 份原始来源已在 OneDrive 的历史源码快照。

这些文件只用于保全被冻结的来源方法与历史版本。`runtime_activated=false`；当前 solver、dataset、receipt、profile、来源绑定和原路径均未修改。所检查 v6 dataset 的直接 `source_pack` 引用中没有上述 authority 派生 NPZ，因此不能把这三份源码称为已接入当前求解链的模块。

保全不等于批准对应源点进入 FIT。场地拟合只允许原片核实属于冰面或围挡本体的静止实体与材料；归属、静止性或像素来源仍待核实的点继续排除。历史版本里的处理方法和文字不承担本次源点审核，也不证明定位准确性。

副本没有被导入或执行。reader 保留硬编码项目根和基于 `__file__` 的 Data 资源布局；两份 predecessor 保留原模块身份的相对导入语义。以后需要重放时，应先分别处理源码位置、运行资源和冻结来源身份，不能直接把复制目录当作可运行的完整数据包。本次未读取视频或大数组、运行测试、求解、安装或编译。
