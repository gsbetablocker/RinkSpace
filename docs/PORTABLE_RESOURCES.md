# 可选资源解析与配置模板

目录分工见[架构说明](ARCHITECTURE.md)，安装和运行条件见[环境说明](../environments/README.md)。本页说明独立的整理工具；现有求解器和 AssetStore 没有接入此工具，活动 profile、源包和原生库仍使用原路径。

## 工具保存什么

[resource_registry.py](../src/rinkspace/resource_registry.py) 把旧绝对地址、`@project/` 和 `@data/` 引用解析到明确的项目根及 Data 根。项目与 Data 必须是分离目录；未登记外部地址、路径越界、未知命名根和相互冲突的别名会报错。相对地址以选定项目根为准，不随进程当前目录改变。

解析器读取既有存储映射，以及本次源码、参数和附件来源清单。已保全资源可用登记的 SHA-256 核对身份；单纯路径存在不等于源内容正确。它不会改原 JSON、安装 import hook、修改环境变量、创建输出目录或加载 DLL。

2026-10-13 的增量参数清单，以及 `calibration/`、`environments/` 下独立保全的源码清单也进入可选发现范围。登记冲突仍会报错，不按日期静默覆盖旧身份。下表模板保存的是各自生成时的映射；后续新增保全映射不会自动重写已有模板，活动 profile 也没有切换。

[prepare_portable_profile.py](../scripts/prepare_portable_profile.py) 为显式选择的 profile 保存资源模板，另可在 Data 中生成一份解析结果。模板保留原 profile 身份、非路径选项及完整附加元数据，包括赛事、源时钟、续算身份和 `strict_model_scope_policy`。附加元数据里的内部地址保留原样，后续仍需在经过核验的适配器里处理；只转换直接选项不能证明整个图可以搬到另一台机器。

输出使用独立的准备 schema，`runtime_activated=false`、`portable_complete_run_verified=false`。它不会调用启动脚本，也不认证观测点是否符合场地范围。当前仍须遵循 [AGENTS.md](../AGENTS.md)：只有原片核实属于冰面或围挡本体的静止实体／材料可以进入场地拟合，模型范围外或归属未确认的点排除，不能借未知深度继续参与。

## 已保存的两个样本

| 模板 | 来源与限制 |
|---|---|
| [okada_named66.resources.json](../configs/portable/okada_named66.resources.json) | 2026-10-11 的旧默认 profile 地址样本；保留历史，不当作符合现行 strict 源筛选规则的运行入口 |
| [strict_v6_source_fit_20261013.resources.json](../configs/portable/strict_v6_source_fit_20261013.resources.json) | 2026-10-13 所选 strict v6 profile 的直接资源与政策快照；使用元数据模式生成，没有重读参数、检查点或数组内容 |

v6 的原 profile 与冻结数据集仍各在原位置，模板不会替换它们。捕获之后隔壁会话可能更新实现或配置；模板中的 source SHA 只标识被捕获的那个版本，不能冒称实时“最新”。`source_eligibility_certified_by_preparation=false` 保留了数据准入与准确性验收的边界。

## 轻量使用方式

以下是将来显式调用的方式。模板只允许写入 `configs/portable/`，解析结果只允许写入 Data 的独立验证目录；不同内容的已有文件会被拒绝覆盖。

```powershell
& D:\Python\Python311\python.exe -B scripts\prepare_portable_profile.py template `
  --metadata-only `
  --profile events\ankara_2026\programmes\okada_mei_sp\strict_v6_source_fit_20261013.json `
  --output configs\portable\strict_v6_source_fit_20261013.resources.json
```

`--metadata-only` 只读取所选 profile、来源清单与文件存在性，不计算参数、权重、NPZ 或检查点的散列。记录中的 expected SHA 来自已登记清单，明确标为本次未重新核验；原 profile 本身的小文件 SHA 仍用于发现读取期间的变更。不带该选项时会读取已登记资源内容以核对身份，因此不宜在要求避开 CPU／磁盘负荷时运行。

在其他位置查看解析结果时，显式传入项目与 Data 根，并把输出放入相应 Data 的 `runtime_tools/repository_validation/portable_profiles/`。解析结果的 `candidate_profile` 只是准备材料，不能作为完整续算、物理 guard 或新机器运行已通过的证明。若 profile 有未登记的直接地址，工具保留原因并拒绝具化，不自行猜测。

## 附件来源

[external_resource_preservation.json](../events/ankara_2026/external_resource_preservation.json) 登记七个已被人工事实引用的临时剪贴板原图地址。在 2026-10-11 的核对中，七项都找到已有 Data 字节相同副本，共 11,598,940 字节；没有重复复制媒体，也没有改原件或活动引用。可选解析器使用这张表把临时地址映射到已登记 Data 副本。该表是当时的身份记录，本轮没有重哈希图片。

[资源边界测试](../tests/test_resource_registry.py) 覆盖搬家、Windows 地址、根目录边界、别名、身份及元数据保留。2026-10-11 已有 16 项轻量测试通过；2026-10-13 的整理仅补充元数据保留检查并做小型标准库断言，没有运行全项目测试或实际拟合。

此次静态审阅还修复了两个可选接口边界：同一规范前缀不能登记成不同目标；同一目标后来补充 SHA 时，所有旧别名同步保留这一身份。最小标准库断言已通过，未借这些检查核验活动求解器或数组内容。
