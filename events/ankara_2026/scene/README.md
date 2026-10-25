# 当前共同场景与广告资产

唯一当前场景入口为 [field/model.json](field/model.json)，由其 advertisement_assets 引用及 SHA 决定当前版本；reference_template_asset 保留内部参数化前的数值参照，其他旧版本不作为当前运行依赖。冰面/围挡几何保持已通过工作单结果；本次新增广告没有更改场地数值。

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

field/model.json 的 installation_review_ref 登记各组 D3 回执（Ad4、Ad6、Ad7 使用 installation_d3_r2.json，其余已登记组使用各自回执）。回执保留当时模板与场景身份，以及展开图、连续围挡三维图和实际源曝光路径；内部参数化后的当前模板由上表及 field/model.json 引用决定，不把历史 D3 回执当成新模板已通过画面审核。诊断图片、视频、日志和数值源大数组留在 Data，原始广告图稿和节目媒体不随公开源包分发。source/sources/selected_sources 索引与 field 引用一起标明本次来源、排除理由、源时钟及历史资料。

模板版式、尺寸和安装均尚未重新拟合或验证米制；新十对等待用户目视审核。D3 图中允许看到初值残差，Ad26 仍有明显继承偏差。没有运行新求解器，也没有因图或数值检查通过而自动激活观测。小字、纹理及难对应的细节只作来源参考。

本机最终审核页：D:\CodexData\RinkSpace_Data\runs\rebuild_ads_20261025\index.html。该页逐对列出官方50fps原生 PTS、纯原片裁图和同源投影，原始 UV、身份及相关组保留。完整节目仍按 [rebuild_scope.json](../programmes/okada_mei_sp/rebuild_scope.json) 登记 S01–S09，本次 D3 为真实曝光子集。

内部参数通过 `domain.parameters.ParameterState` 的共同float64布局消费，字形参考UV不再被当作不可调整的最终形状。每对共用内部组与尺寸块；孔/同字断笔同组，方形/矩形按其类型生成。未定位实例仍无位置和源像素。所有既有人工确认及原读数保留；状态接通与工程验证不代表新拟合或画面验收。
