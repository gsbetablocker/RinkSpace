# 公开分发与分项许可核查

核查日期：2026-10-19。范围为当前 Git 候选源码、构建定义、必要审阅归档、模型和来源记录；不包含私有旧历史和已安装环境。项目介绍见 [README](../README.md)，维护入口见 [MAINTENANCE](../MAINTENANCE.md)。

## 当前结论

继续使用根目录的[分项许可与权利范围声明](../LICENSE.md)，不把整个仓库改标 MIT。项目自有且未另行许可的内容不额外授予使用、修改、分发、再许可或商用许可；第三方已经授予的权利继续有效。本声明不代替第三方授权。

用户已确认三份待查辅助代码由其使用 ChatGPT 生成；三张场地图由 Codex 在本项目中生成，可能包含用户标注。用户没有赛事素材的既有授权记录，并明确选择：**公开仓库不附独立比赛截图与叠加图，保留原有测量数据。** 对应图像在 Data 本地保全，公开包保留代码、参数、时钟、观测、统计量及明确的缺失说明。

## 已自行核实的事项

| 范围 | 证据与处理 |
|---|---|
| SAM2 基础权重 | 固定为 facebook/sam2.1-hiera-tiny，修订 de431c4043854a71d8101e17995dfe596bf101a5。本地下载收据、当前实际文件 SHA 和[官方固定提交](https://huggingface.co/facebook/sam2.1-hiera-tiny/commit/de431c4043854a71d8101e17995dfe596bf101a5)的 LFS 指针一致；该提交模型卡标 Apache-2.0。无需用户重复确认。 |
| 本项目 rink_vision.pt | 当前 SHA 为 26e792f2381a9bd66c484b8cfa38b13993f5c6cf11b52ef84ee22da65041f208；与训练报告 step 9200 一致。162 个编码器张量、108,876,544 字节与上述基础权重逐字节一致。208 个存储均属于模型状态，未发现额外原图缓冲区。完整来源、修改和训练时钟见[模型来源记录](../models/rink_vision.provenance.json)。保留[SAM2 原许可](../models/SAM2_LICENSE.txt)，不把完整派生权重重标 MIT。 |
| GVHMR 衍生补丁 | 对应冻结提交 6ec3ca39336c50492c0fae65fba2fb831fc7d866。保留[原许可](../environments/third_party_patches/GVHMR_LICENSE.txt)、公开补丁、修改说明和论文引用，见[第三方登记](THIRD_PARTY_ASSETS.md)。上游教育、研究、非营利与衍生修改非商业条件继续适用，不能由 MIT 覆盖。 |
| Ceres／Abseil、SciPy、oneTBB | 既有原许可证和源码署名保持原字节。Ceres 为 BSD，嵌入 Abseil 含 Apache 条件；SciPy 派生实现保留 BSD；oneTBB 保留 Apache。各自原授权不会因项目声明而被缩减。 |
| run_vitpose_h.py | 当前副本与旁项目原件 SHA 相同；历史未跟踪本身不能证明权属。用户现已确认 ChatGPT 生成来源，已补[来源记录](../calibration/program_tracking/body_tracking/source_helpers/run_vitpose_h.provenance.json)；它动态调用的 GVHMR／ViTPose 和外部检查点仍按各自条件处理。 |
| calibration companion 两份代码 | 原 ZIP SHA 和两份原始源码 SHA 均匹配，包内说明是新写辅助代码；用户确认 ChatGPT 生成。已补[provenance](../src/rinkspace/calibration_companion/provenance.json)，保留旧适配 SHA，另登记当前 SHA，未把历史记录伪装成当前字节。 |
| 外部模型与框架 | GVHMR、HMR2、ViTPose、SMPL-X、YOLO 检查点未打包进当前 Git 候选。RAFT 仅在被忽略的本地缓存；PIDNet、SMP、co-tracker 在已查实现中是参考链接，未发现相应打包权重。外部运行依赖不能误写为本仓库已分发资产。 |

关于 AI 输出，[OpenAI 官方使用条款](https://openai.com/policies/row-terms-of-use/)在用户与 OpenAI 之间、法律允许的范围内约定输出权利归用户，并转让 OpenAI 对输出可能享有的权利；这不保证输出独一无二，也不替第三方输入或依赖授予权利。因此此次记录解决代码来源缺口，没有声称所有 AI 代码必然具有排他版权或不受第三方条件约束。

## 比赛画面不进入公开包

- 排除两张机位证据 JPG，以及 review 中 23 张 native PNG、2 张 clipboard 叠加图。来源路径、SHA、PTS 和人工事实仍保留；缺失项从可读取 assets 移到 public_withheld，不让读取器假装它们仍在公开包。
- 按用户要求不再逐色值清理测量记录：观测 JSON、GZ、NPZ 中的色值采样、剖面、ROI 测量及原有源 SHA 保持，避免改变开发输入和续算契约。图片文件排除不等于这些测量记录获得第三方再许可；原片取样仍按原来源及有效条件处理。
- 三张项目生成的场地图可保留；前两张实际是同 SHA 的图。两张 original_mask 为只含 0/255 的分类掩膜，生成代码不叠广播 RGB，保留为必要观测数据。
- 18 个源数组是观测坐标、索引、方差、状态等数值数据；profile_normal 是二维方向，不是原像素。review/weights.zip 的 1,604 项全部为统计 W 数组；shot1_baseline.zip 的 2,096 项全部为观测／绑定 JSON。未因文件名误删关键数据。
- 网页不自动载入本机比赛截图；历史源码中的图像处理能力、源视频地址和 PTS 不等于仓库附带原图，不删除相应代码或来源事实。

本地保全与实际排除清单见 review/manifest.json 的 public_withheld、public_media_derivative；两张本地机位 JPG 由 .gitignore 明确排除。公开包可以离线审查代码、模型、数值定义和失败状态；**原像素复核、颜色剖面重放与整镜头视觉验收需要本地原素材，不能宣称仅凭公开包已经完成。**

[YouTube 许可说明](https://support.google.com/youtube/answer/2797468?hl=en)和[使用条款](https://www.youtube.com/static?template=terms)不使观看者自动获得服务外再分发权。两条具体视频的实际 CC 字段未能取得，不能把默认 Standard 当作这两条视频的实证；现按用户决定不分发画面，无需再要求用户为这些图片作授权保证。未代用户联系权利人。

## 模型训练与运行边界

rink_vision.pt 没有直接嵌入比赛画面，继续作为关键模型保留。它的 SAM2 基础来源和 Apache 许可已闭合；训练输入来自旧广播源 oxXu1avhrvs 和项目标注，用户未报告既有授权记录。移除原图不等于证明所有训练使用或派生模型的每种公开／商业用途都已获得授权；本次不对具体司法辖区的训练例外、输出权利或专利自由实施作法律结论。该限制如实记录，不据此篡改训练事实或删除模型。

[pyproject.toml](../pyproject.toml)的 advanced 声明 quadprog 0.1.13，其[发布者元数据](https://pypi.org/project/quadprog/0.1.13/)标 GPLv2+。当前未打包该库二进制；查到的实际 import 在环境 smoke 验证中，src/、calibration/、scripts/ 未发现生产调用。仅列依赖或本地安装不能直接推定整个仓库已按 GPL 分发；实际组合和分发时仍需满足适用条件，见 [GNU 说明](https://www.gnu.org/licenses/gpl-faq.en.html#IfLibraryIsGPL)。没有为这项核查删除依赖或改动运行环境。

## 还需要用户确认什么

代码生成来源、项目图来源和没有赛事授权记录三项均已由用户回答，现无重复待答的事实问题。项目继续采用已选分项声明，已按用户随后授权准备直接提交并推送。若未来要公开比赛画面或作出模型商用／无条件授权承诺，需另有可适用依据；本次不把“研究用途”或用户上传自动等同于第三方许可。
