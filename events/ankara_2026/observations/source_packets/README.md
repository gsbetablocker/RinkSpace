# 已保全的赛事源包

返回 [赛事说明](../../README.md)；首轮文件索引见 [资产保全清单](../../asset_preservation_manifest.json)，strict v6／q32 的后续补充见 [2026-10-13 增量清单](../../asset_preservation_manifest_20261013.json)。两份清单有重合，复用资源不再次复制。

本目录保留当前声明输入中原先只有 Data 副本的 JSON 源包、输入政策和材料角色事实。2026-10-14 已将超出 Windows 完整路径限制的文件移至同赛事或节目的 `observations/p/` 短目录，文件字节与 SHA-256 不变；旧地址由项目 `configs/path_aliases.json` 登记。每项原地址、当前项目地址、字节数与 SHA-256 在清单中登记。已有逐字相同项目资源直接复用，没有再复制。

世界初始化保存在赛事 `models/initializations/`；节目人工笔划、registration 与 guard 等不可替代事实保存在 `programmes/okada_mei_sp/observations/`。完整优化状态、大运动数组、源视频和批量源图继续在 Data。

文件保留原字节及内部地址。现有运行配置仍使用原输入，不会因副本出现而换源；后续资源解析应使用清单的对应关系，保持源身份和人工事实。清单明确记录尚未保全的嵌套外部依赖，因此这些副本不能单独证明完整节目可在其他机器运行。
