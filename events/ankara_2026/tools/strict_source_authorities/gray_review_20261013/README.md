# 灰色材料审阅的冻结源码保全

这里保存 3 份原先仅位于 Data 的冻结 authority 源码版本：一个源测量 reader 和两份 predecessor。原始文件及既有 Data 快照继续保留；出处、文件状态、SHA-256、静态依赖和既有冻结快照路径见 [source_preservation_manifest.json](source_preservation_manifest.json)。3 个副本共 90,109 字节，逐字节匹配原件，也匹配既有 `exact_source_authority_relocation_receipt.json` 声明的散列。

目录保留 reader 与 predecessor 的相对位置。`--compare-default-predecessor` 分支以 `rinkspace._gray_source_saved_predecessor` 和 `rinkspace._gray_source_saved_guides_predecessor` 加载两个旧版本；相对导入仍依赖项目 src/rinkspace，其他明示依赖见来源清单。

`runtime_activated=false`；这些文件用于追查被冻结的方法和版本，不代表接入当前 native 求解。所检查历史 v6 dataset 的直接 source_pack 没有这些 authority 派生 NPZ。

保全不等于批准对应源点进入 FIT。场地拟合只允许原片核实属于冰面或围挡本体的静止实体与材料；归属、静止性或像素来源仍待核实的点继续排除。历史方法的保全不证明源点资格或定位准确性。

reader 保留原硬编码根目录和基于 __file__ 的 Data 布局，predecessor 保留原模块身份及相对导入。重放需同时匹配源码位置、资源和来源身份；本目录不是可直接运行的完整数据包，静态保全记录不表示已执行原始读取或验收。
