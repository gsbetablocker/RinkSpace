# 当前共同场景与广告资产

本页保存安卡拉赛事的模型资产和取点定义；通用来源、拟合范围与广告几何见[建模与接入](../../../docs/rink_model/BUILD_SPEC.md)，整镜头判定见[验证说明](../../../docs/VALIDATION.md)。求解后的数值状态与基础资产分别绑定，当前实验见[节目入口](../programmes/okada_mei_sp/README.md)。

<a id="ice-observations"></a>
## 当前冰面取点定义

F1–F4/R1–R4 八组靶圈的 center、inner、middle、outer 四层圆，以及半椭圆蓝区取材料轮廓。半椭圆蓝区只取圆弧，不取靠球门线的闭合直边；该直边不是球门区观测实体。其余冰面直线或曲线（包括 C0、A1–A4、Lane、连接线、球门线、梯形线）均取两侧材料边缘之间的宽度中线，模型中忽略实际宽度，即使实际宽度很大。

当前模型、取点和拟合输入禁止将上述中线改用旧外/内边缘定义、别名或兼容回退，未激活条目也不得混入；旧像素不得改名伪装为中线观测。旧观测与失败记录保留供追查，不因保留而成为当前输入。

这条冰面中线规则不适用于围挡广告；广告的字形外缘、内孔、面板边与明确印刷图形保持各自材料轮廓定义，并绑定实际实例、模板及路径。取点核查与显示覆盖要求统一见[连续原片审阅](../../../docs/VALIDATION.md#共同模型与连续原片审阅)。

## 当前资产与审核状态

场景结构入口为 [field/model.json](field/model.json)，当前广告版本由其 advertisement_assets 及 SHA 确定。该文件保存 `unfitted_structure_candidate` 基础结构与初值；已求解参数位于运行的 ParameterState／数值快照，不能把基础文件当成最新拟合状态，也不能把它的状态扩展为“从未进行过真实拟合”。冰面／围挡保留 M1–M5 已确认结构关系，尺寸、相机物理和已验证米制仍需相应证据。

13 套共享模板对应 26 个广告身份，每对共享模板和尺寸，安装独立；Ad15、Ad16 未定位，其余实例有源参考。基础模型包含 68 条冰面、2 条围挡、558 条已定位广告路径，共 628 条；固定交点等 Feature 按实际输入另行编译。

| 配对 | 当前模板资产 | 模板路径数 | 资产类型 |
|---|---|---:|---|
| Ad1 / Ad14 | [assets_r5.json](ad1/assets_r5.json) | 26 | 基础数值模板 |
| Ad2 / Ad15 | [assets_r6.json](ad2/assets_r6.json) | 21 | 基础数值模板 |
| Ad3 / Ad16 | [assets_r4.json](ad3/assets_r4.json) | 19 | 基础数值模板 |
| Ad4 / Ad17 | [assets_r4.json](ad4/assets_r4.json) | 33 | 基础数值模板 |
| Ad5 / Ad18 | [assets_r2.json](ad5/assets_r2.json) | 32 | 基础数值模板 |
| Ad6 / Ad19 | [assets_r3.json](ad6/assets_r3.json) | 20 | 基础数值模板 |
| Ad7 / Ad20 | [assets_r4.json](ad7/assets_r4.json) | 42 | 基础数值模板 |
| Ad8 / Ad21 | [assets_r3.json](ad8/assets_r3.json) | 14 | 基础数值模板 |
| Ad9 / Ad22 | [assets_r3.json](ad9/assets_r3.json) | 4 | 基础数值模板 |
| Ad10 / Ad23 | [assets_r2.json](ad10/assets_r2.json) | 13 | 基础数值模板 |
| Ad11 / Ad24 | [assets_r2.json](ad11/assets_r2.json) | 7 | 基础数值模板 |
| Ad12 / Ad25 | [assets_r3.json](ad12/assets_r3.json) | 34 | 基础数值模板 |
| Ad13 / Ad26 | [assets_r4.json](ad13/assets_r4.json) | 34 | Ad26 安装方向平行围挡；共享模板保持原样 |

## 共同参数与版式

| 版本 | 结构入口 | native 布局与范围 |
|---|---|---|
| 严格名义结构 | [field/model.json](field/model.json) | 选项合计 63、layout39；共享关系由构造严格满足 |
| 当前候选的冰面容差结构 | [field/ice_tolerance_current.json](field/ice_tolerance_current.json) | 选项合计 127、layout146；在名义结构后增加 107 个相对偏差 |
| Lane 横段迁移快照 | [field/lane_transverse_current.json](field/lane_transverse_current.json) | 选项合计 63、layout39；保留容差版本之前的结构与初值 |

容差版本的尺寸、位置与方向按 `ice_tolerance` 声明相对共同名义量允许 ±1%。长度以对应名义尺寸归一化，位置采用局部参照长度，方向采用横向漂移／前进长度；迭代不会重新归零共同名义量。图元类型、共享接点、同一靶圈内部同心关系、有限支持与严格顺序保留；其他共线、平行、等距与对称关系按声明放宽。该版本已用于六镜头真实拟合及当前九镜头候选，数值状态和验收范围见[节目入口](../programmes/okada_mei_sp/README.md)。

Lane1–Lane4 各有一条连接本 Lane 上下长边的短线，四线共线且平行冰场短边，人工确认见 [lane_transverse_20261029.json](../model_constraints/lane_transverse_20261029.json)。严格名义结构通过选项 32 共享 `shared_layout[35]` 的一个 X 坐标，各线端点 Y 由既有 Lane 位置与宽度生成；容差版本的偏差另按上表声明处理。

严格名义结构的选项 16 将两端球门线构造成 `x=±d`，两个半椭圆球门区共用内向半轴 `ax` 和横向半轴 `ay`，中心为各自球门线中点；曲线只包含朝冰场内部的半椭圆弧。梯形共用球门线端与围挡端的两个半开口跨度，关于冰场中轴镜像。五个共同尺寸参与拟合；旧版两端独立参数仅用于历史回放。

[print_layout_components.json](print_layout_components.json)将同一完整印刷图案中的文字、标志、内孔、间距及解析图形保持为一个版式，不开放逐字或逐行独立变形。Ad2／3／4／6／7 相对已声明的真实底板各有四个整幅图案坐标，S01 共激活 20 个内部坐标；整体尺寸和各实例安装继续参与联合拟合。参考图稿不是实测米制母版。

`domain.parameters.ParameterState` 保存共同 float64 参数、活动列和规约。参考顶点／材料路径及源 UV、PTS、人工确认保持各自身份；绘图加密点不是自由度。十维 print_shape 是存储能力，当前六个非线性形变坐标为零且不活动。未定位实例不产生安装、源像素或拟合资格。

Ad26 文字排列平行于围挡的人工要求通过安装块 `installation/Ad26/shared_placement` 实现：展开面旋转固定为 π，活动坐标为沿围挡位置和高度，共享尺寸仍可拟合。旋转列从 native 求解方向中移除，载入非平行状态会报错；完整图案内部关系与另一安装 Ad13 保持不变。

结构确认、来源核实和几何前向成功不自动表示尺寸或完整画面通过。

## 来源与历史资产

[archive.json](archive.json)保存参数化前参考模板、导入快照、旧安装回执与撤回版本的原路径、SHA、字节数及 Data 地址。reference_template_asset 和 installation_review_ref 描述对应历史原件，不是当前运行必须重新读取的资产，也不表示当前模板已经通过原回执的画面审核。source／sources／selected_sources 索引保留各实例来源、排除理由与源时间。

原安装预览在 `Data/runs/rebuild_ads_20261025/index.html`；图像、视频、原广告图稿和大型数值源保留于 Data。早期逐字形变与后续版式迁移、失败候选及原状态保留于 `Data/runs/native_joint_s01_20261026/print_component_layout/`、`print_artwork_layout/` 等既有目录，不作为当前默认参数布局。
