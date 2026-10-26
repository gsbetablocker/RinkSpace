# 当前共同场景与广告资产

本页保存安卡拉赛事的模型资产和取点定义；通用来源、拟合范围与广告几何见[建模与接入](../../../docs/rink_model/BUILD_SPEC.md)，整镜头判定见[验证说明](../../../docs/VALIDATION.md)。以下资产状态与具体任务的执行范围分别读取，不能据此自行启动求解。

<a id="ice-observations"></a>
## 当前冰面取点定义

F1–F4/R1–R4 八组靶圈的 center、inner、middle、outer 四层圆，以及半椭圆蓝区取材料轮廓；其余冰面直线或曲线（包括 C0、A1–A4、Lane、连接线、球门线、梯形线）均取两侧材料边缘之间的宽度中线，模型中忽略实际宽度，即使实际宽度很大。

当前模型、取点和拟合输入禁止将上述中线改用旧外/内边缘定义、别名或兼容回退，未激活条目也不得混入；旧像素不得改名伪装为中线观测。旧观测与失败记录保留供追查，不因保留而成为当前输入。

这条冰面中线规则不适用于围挡广告；广告的字形外缘、内孔、面板边与明确印刷图形保持各自材料轮廓定义，并绑定实际实例、模板及路径。取点核查与显示覆盖要求统一见[连续原片审阅](../../../docs/VALIDATION.md#共同模型与连续原片审阅)。

## 当前资产与审核状态

唯一当前场景入口为 [field/model.json](field/model.json)，由其 advertisement_assets 引用及 SHA 决定当前版本。参数化前的参考模板、selection/glyph 导入快照、旧安装回执和撤回版本已归入 Data；[归档映射](archive.json)保留旧路径、原始 SHA、字节数和 Data 相对地址。reference_template_asset 及安装回执地址只指向历史原件，不是当前运行依赖。冰面/围挡保留已确认的 M1–M5 结构关系；场景仍为 unfitted_structure_candidate，结构确认不等于尺寸、相机或整镜头精度通过。广告接入未更改这些场地初值。

13 对广告保存 26 个实体身份。每对引用同一模板和尺寸块，安装位置独立；Ad15、Ad16 保留身份但未定位。其余实例均有对应真实源参考。没有精确半圈或等尺寸伪观测。

| 配对 | 当前模板资产 | 实体路径数 | 状态 |
|---|---|---:|---|
| Ad1 / Ad14 | [assets_r5.json](ad1/assets_r5.json) | 26 | 已接共同可调内部状态；未拟合 |
| Ad2 / Ad15 | [assets_r6.json](ad2/assets_r6.json) | 21 | 已接共同可调内部状态；未拟合 |
| Ad3 / Ad16 | [assets_r4.json](ad3/assets_r4.json) | 19 | 已接共同可调内部状态；未拟合 |
| Ad4 / Ad17 | [assets_r4.json](ad4/assets_r4.json) | 33 | 已接共同可调内部状态；未拟合 |
| Ad5 / Ad18 | [assets_r2.json](ad5/assets_r2.json) | 32 | 已接共同可调内部状态；未拟合 |
| Ad6 / Ad19 | [assets_r3.json](ad6/assets_r3.json) | 20 | 已接共同可调内部状态；未拟合 |
| Ad7 / Ad20 | [assets_r4.json](ad7/assets_r4.json) | 42 | 已接共同可调内部状态；未拟合 |
| Ad8 / Ad21 | [assets_r3.json](ad8/assets_r3.json) | 14 | 已接共同可调内部状态；未拟合 |
| Ad9 / Ad22 | [assets_r3.json](ad9/assets_r3.json) | 4 | 已接共同可调内部状态；未拟合 |
| Ad10 / Ad23 | [assets_r2.json](ad10/assets_r2.json) | 13 | 已接共同可调内部状态；未拟合 |
| Ad11 / Ad24 | [assets_r2.json](ad11/assets_r2.json) | 7 | 已接共同可调内部状态；未拟合 |
| Ad12 / Ad25 | [assets_r3.json](ad12/assets_r3.json) | 34 | 已接共同可调内部状态；未拟合 |
| Ad13 / Ad26 | [assets_r3.json](ad13/assets_r3.json) | 34 | 已接共同可调内部状态；未拟合 |

field/model.json 的 installation_review_ref 登记各组 D3 在 Data 中的原始回执，并保留 SHA。回执的旧地址与原相对引用基准见归档映射；其字节未重写，仍描述当时模板、场景及真实源曝光。内部参数化后的当前模板由上表及 field/model.json 引用决定，不把历史 D3 回执当成新模板已通过画面审核。诊断图片、视频、日志和数值源大数组留在 Data，原始广告图稿和节目媒体不随公开源包分发。source/sources/selected_sources 索引与 field 引用一起标明本次来源、排除理由、源时钟及历史资料。

在上述 D3 初值阶段，模板版式、尺寸和安装尚未重新拟合或验证米制；新十对等待用户目视审核。D3 图中允许看到初值残差，Ad26 仍有明显继承偏差。该阶段没有运行新求解器，也没有因图或数值检查通过而自动激活观测。小字、纹理及难对应的细节只作来源参考；后续 S01 版式修正与拟合状态见下文及节目记录。

2026-10-25 D3 待审页：D:\CodexData\RinkSpace_Data\runs\rebuild_ads_20261025\index.html。该页逐对列出官方50fps原生 PTS、纯原片裁图和同源投影，原始 UV、身份及相关组保留。完整节目仍按 [rebuild_scope.json](../programmes/okada_mei_sp/rebuild_scope.json) 登记 S01–S09，本次 D3 为真实曝光子集。

内部参数通过 `domain.parameters.ParameterState` 的共同float64布局消费，字形参考UV不再被当作不可调整的最终形状。每对共用内部组与尺寸块；孔/同字断笔同组，方形/矩形按其类型生成。未定位实例仍无位置和源像素。所有既有人工确认及原读数保留；状态接通与工程验证不代表新拟合或画面验收。

2026-10-27 广告内部自由度修正：上述初次接入中的独立字母变形已由 [明确版式定义](print_layout_components.json) 替换。当前 13 个模板将同一完整印刷图案内的文字、标志及解析矩形保持为同一版式，不再开放逐字或逐行独立变形；Ad2／3／4／6／7 相对真实底板各保留 4 个整幅图案坐标。模板及尺寸仍由每对实例共享，整块广告尺寸与独立安装继续可拟合。参考图稿并未因此被宣称为实测米制图。

当前 S01 实际观测激活的广告内部坐标由 1,326 经整词／整行阶段的 72 降为 20，另有 22 个共享尺寸、33 个安装坐标；加上 31 个相机、352 个运动方向，共注册 492 个活动方向，不等于独立可辨识维数。`parameterize_print_layout` 和参数校验器要求显式完整路径分组，并拒绝重新开放六个非线性变形坐标。模板原路径身份、顶点和来源记录保留；各次旧模型、失败数值和迁移回执在 `Data/runs/native_joint_s01_20261026/print_component_layout/`、`print_artwork_layout/`。当前模型已进入同一 native GPU 联合拟合，画面与物理参数的完整验收仍未完成；最新输入与修正范围见[节目记录](../programmes/okada_mei_sp/README.md)。
