# 可选资源解析与配置模板

本页说明独立的资源整理工具。现有求解器及 AssetStore 未接入 ResourceRegistry；实际存储解析见 [MAINTENANCE](../MAINTENANCE.md#本机工具与存储解析)，不要把模板当成可直接续算的运行输入。

## 工具保存什么

[resource_registry.py](../src/rinkspace/resource_registry.py)将旧绝对地址、@project/ 与 @data/ 引用解析到明确且分离的项目/Data 根。相对地址以项目根为准；未登记外部地址、越界、未知根、冲突别名会报错。同一规范前缀不能指向不同目标，补充的 SHA 会同步到该目标的既有别名。

解析器使用既有存储映射与保全清单，可按登记 SHA 核对身份；路径存在不等于内容正确。它不改原 JSON、不安装 import hook、不改环境变量、不建输出目录或加载 DLL。新增清单不自动改写已有模板，也不切换活动 profile。

[prepare_portable_profile.py](../scripts/prepare_portable_profile.py)为显式所选 profile 保存资源模板。模板保留身份、非路径选项及附加元数据，包括赛事、源时钟、续算身份和 strict_model_scope_policy；附加元数据中的内部地址保留原样，不能据直接地址转换宣称整个依赖图已可移植。

输出使用独立准备 schema，明确 runtime_activated=false、portable_complete_run_verified=false、source_eligibility_certified_by_preparation=false。它不启动任务，也不认证[观测准入](rink_model/BUILD_SPEC.md#observation-eligibility)。

## 已保存的两个样本

| 模板 | 来源与限制 |
|---|---|
| [okada_named66.resources.json](../configs/portable/okada_named66.resources.json) | 2026-10-11 旧 profile 地址样本，不是现行运行入口 |
| [strict_v6_source_fit_20261013.resources.json](../configs/portable/strict_v6_source_fit_20261013.resources.json) | 2026-10-13 strict v6 的直接资源与政策快照；以元数据模式生成，未重读参数、检查点或数组 |

样本 SHA 只描述捕获版本，不是实时最新状态；原 profile 与冻结数据集仍保留原身份。

## 轻量使用方式

在项目根显式调用，例如：

~~~powershell
python -B scripts/prepare_portable_profile.py template --metadata-only --profile events/ankara_2026/programmes/okada_mei_sp/strict_v6_source_fit_20261013.json --output configs/portable/strict_v6_source_fit_20261013.resources.json
~~~

工具将模板输出限制在 configs/portable/，解析结果限制在 Data/runtime_tools/repository_validation/portable_profiles/，拒绝覆盖不同内容的既有文件。这是当前工具的实际路径检查，不是新加的项目审批或任务安排。

--metadata-only 读取 profile、清单与存在性；参数、权重、NPZ 和检查点的 expected SHA 来自登记清单，标明此次未重新核验，所选 profile 小文件仍计算 SHA。不带该选项会读取登记资源核对身份；按实际任务选择。

换机器时显式传入项目及 Data 根。candidate_profile 只是准备材料；未登记直接地址会保留失败原因并拒绝具化，不自行猜测，也不证明完整续算或物理 guard 已通过。

## 附件来源与检查

[external_resource_preservation.json](../events/ankara_2026/external_resource_preservation.json)保存 2026-10-11 临时剪贴板原图与已有 Data 副本的身份映射；这是历史核对记录，不是每次解析都重新散列图片。

相关边界见[资源测试](../tests/test_resource_registry.py)。历史测试数量和当时通过状态不作为当前认证；修改解析器时检查受影响的根目录边界、别名、身份与元数据行为，不因此触发实际拟合或全量素材审核。
