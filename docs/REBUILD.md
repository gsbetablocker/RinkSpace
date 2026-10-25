# 重构实现状态与复现入口

截至 2026-10-25，共同 native 前向、完整状态导入、prepare/review、场地模型和广告 D1/D3 已有实际实现。新引擎的 GPU LM、solve/resume 和真实拟合尚未完成。本文记录当前能力；旧交付包、阶段工单和历史审核回执不是正常运行前置条件。

设计与实现边界见 [ARCHITECTURE.md](ARCHITECTURE.md)、[SOLVER.md](SOLVER.md) 和 [INTERFACES.md](INTERFACES.md)。[MIGRATION.json](MIGRATION.json) 记录迁移归属，不是删除清单；目标架构不表示相应求解接口已经实现。

## 当前实现

| 范围 | 当前能力与限制 |
|---|---|
| native ABI 2 | 共同几何、固定节点、连续围挡、静态及完整时变相机前向、预测行隐式 RS；CPU 数学库，没有 GPU LM |
| 完整状态导入 | `model import-state` 导入冻结 d78 镜头状态及实体，保留九镜头登记，不补造其他镜头相机；`import-d78` 仅生成未拟合结构初值 |
| prepare | 校验当前输入、观测合同、相关组、源 PTS 及相机定义域，生成选定镜头的 prepared；当前仍核对所读数组，不是已实现的增量缓存系统 |
| review | 同一 native 前向、同源 PTS 的完整选定镜头回放及可选同源对照；不优化，不表示九镜头已经拟合或通过审核 |
| 广告 D1 | 来源、模板、实例及逐轮廓关联导入；模板与尺寸共引用，实例安装参数独立，缺失来源不合成像素 |
| 广告 D3 | 展开面、三维连续围挡及真实曝光叠加，共用 material UV、native 路径和相机；安装值仍是未重新拟合初值 |
| solve/resume | 尚无 CLI 实现、GPU 残差/导数批处理、LM 或完整检查点续算生命周期；合同保留为实现目标 |

`core/native.py` 是薄绑定，数学在 `native/`；`domain/` 管理实体、观测和印刷资产；`session/` 编排导入、准备及展示；`review/` 提供回放并保留旧 review API。当前实体图/回放仍引用 `compat.d78_entities` 的静态采样辅助，这是尚未完全退出的兼容依赖，不会重跑历史数据生产。

## 共同场地与广告

当前共同场地位于 `events/ankara_2026/scene/field/model.json`，由该文件引用决定资产版本。本机当前为 r34，保留 13 对、26 个广告实例身份；审计包的三对广告基准不替代当前库存。当前关系包括：

- F/R 共列、配对共行，四行等距且上下外圈至长边等距；保留 center/inner/middle/outer 四层靶圈共32条圆。
- Lane 三条实体边，无左封口和已撤销的等长关系；左侧绘制支持不越过左底线，连接中线支持独立且停在 N 左侧。
- B1/B2/C 和两门线平行短边，端点接实际轮廓；A1/A4 同下行、A2/A3 同上行，四圆心上下左右对称，两门线位置仍独立。
- 蓝区为半椭圆；梯形端点和围挡上下沿使用同一连续围挡。围挡曲线坐标可跨直/弧及闭合接缝，不把点锁在旧分段上。

冰面八组靶圈与半椭圆蓝区取材料轮廓，其余冰面线只有宽度中线、模型零线宽。旧冰面边缘定义及别名不能进入当前 feature，未激活条目同样拒绝；旧像素不会改名伪装为中线观测。广告字形外缘、内孔、面板边与明确印刷图形按实际实例/模板/path/role 保持材料轮廓，经 Feature、prepare 和 prepared 载入执行同一绑定检查。绘制支持不生成源观测。

尺寸仍用自有单位，长宽可独立变化，没有固定 2:1；尚未重新拟合或验证米制。与工程设计 60×30 m 最接近的单一尺度转换不等于固定长宽比。

当前已接入的配对广告如下。模板与尺寸由参数引用共享，不增加等尺寸伪观测，也不强制两个实例相差精确半圈。

| 配对 | 当前模型内容 |
|---|---|
| Ad2 / Ad15 | `ISU-SKATING.COM` 字形、A/O 内孔及面板四边；无背景纹理。Ad15 保留身份、安装未知，不复制 Ad2 源像素 |
| Ad1 / Ad14 | 清晰文字、内孔、面板四边及二维码白色外方框；方框在展开面中为实际正方形。无 QR 小格、手机界面或细碎装饰；安装允许跨弯角 |
| Ad3 / Ad16 | 当前接受模板的六边形、`買取` 及面板轮廓；`大吉` 不在当前几何/源关联中。Ad16 保留身份、安装未知；历史全场 UV 保留来源/候选语义 |
| Ad4 / Ad17 | ANKARA、JUNIOR GRAND PRIX、可靠内孔和四边；撤回拼帧假接缝，ANKARA 使用同一真实源 chart；右侧文字经同一行变换去除继承剪切，恢复正体 |
| Ad5 / Ad18 | 40、MEIWA、明和地所及蓝绿矩形图案边；该矩形不是整块白色广告外框 |
| Ad6 / Ad19 | オリエンタルバイオ大字、分离笔画和可靠タ孔；31.000s 清晰尾部修正粗笔画/粘连；两オ模糊假孔只留历史，首オ为同字形推断；无背景纹理 |
| Ad7 / Ad20 | ISU logo、两行文字、孔和开放弧；logo 保留当前 panel 域，两行文字从原官方 canonical 轮廓恢复正体。2192 条原 logo 读数保留，上下文字不制造源行 |
| Ad8 / Ad21 | WYNDHAM、白色 W 图案及蓝块边；不恢复撤回的 Ankara 小字、商标和假 WY 连桥 |
| Ad9 / Ad22 | 与现场匹配的 can 字形及徽章大边界；不套用字形不同的官方另一版图稿 |
| Ad10 / Ad23 | SPORTOTO 八字、三个 O 孔和上下大弧；无小字和纹理 |
| Ad11 / Ad24 | 大环、盾形、新月和大星；无环绕小字、年份及冰鞋细节 |
| Ad12 / Ad25 | 盾形大边、四行市政文字及可靠内孔；无细塔、小星等内部装饰 |
| Ad13 / Ad26 | 土耳其体育部双行文字、可靠字孔和大外环；环仅线框，无碎星、运动员及细碎徽章内容 |

新增十对已完成同一 native 核心的展开面、连续围挡三维和双面真实曝光 D3；每镜头使用同一份完整冻结相机、全套使用同一最终 scene。原 UV、PTS、来源身份与相关组保留，机器候选未自动激活或升级为人工通过。Ad26 仍有明显的继承安装初值偏差，坐标等价核查后未按目视调参。当前资产入口见 [场景索引](../events/ankara_2026/scene/README.md)。

资产导入、模板展示和 D3 安装展示不自动授予观测拟合资格，也不表示已经完成广告拟合。

独立审查修订补齐：10 个标线与真实连续冰面边界固定交点（仅模型节点、无新 UV）、13 个广告孔洞父路径引用、当前场地/广告整场 review、scene 与 camera/controls 的自包含 prepare 快照，以及当前内容的 revision 绑定。D1 模板预览已在审计包的 Ad1/Ad2/Ad3 三组资产上验证，缺私有原图或读数时逐项列出；几何和安装数值未因审查而重新拟合。

审计包三对广告基准的完整 review 库存为 66 条场地路径和 92 条已定位广告路径，共 158 条；当前 r34 的完整库存为 66 条场地路径和 558 条已定位广告路径，共 624 条；工程回归检查覆盖该库存，未据此声明真实整镜头画面通过。Ad15/Ad16 保持未定位，不生成坐标。D3 的 `--frame` 是未核实图片预览；用 `--source-video` 与 `--scope` 核对现有原视频 SHA 并按整数 PTS 解码，才可声明来源曝光已核实。该声明不授予画面通过。相同视频供多个 review 镜头使用时仅校验一次 SHA。

广告字形边与内孔已通过显式 print_binding 进入通用 Feature，按广告材料轮廓规则检查，不再套用冰面中线规则。冰面标线与 board.bottom 的固定交点仍需有类型地表达合法跨表面关系，不能整体取消表面校验。现有模板与原读数保留，不将材料边缘改名成中线、不将模型采样点冒充源点。具体边界见 [INTERFACES.md](INTERFACES.md)。

[Ad2 两条 O 边界的审核记录](../events/ankara_2026/programmes/okada_mei_sp/ad2_o_review.json)已保存用户对官方单节目 PTS 320000（00:25.000）这一曝光的确认。确认只涉及两个指定既有记录的实体身份及展示读数；原 UV 不变，保留 `active_for_fit=false`，不扩展到其他曝光、特征、模板几何或相机拟合。

## 九镜头范围与原生时钟

范围为 `events/ankara_2026/programmes/okada_mei_sp/rebuild_scope.json` 的 **S01–S09**。`required_shot_ids` 保留全集，阶段选择只改变 `selected_shot_ids`；旧 `fit_joint6` 只是历史子集。

当前处理源是已登记的 ISU 官方 50fps 单节目视频，`time_base=1/12800`。处理曝光从 PTS 256256（00:20.020）至 2415104（03:08.680），右开截止为 2415360（03:08.700）；S09 物理镜头末曝光为 2485504（03:14.180），与处理截止分别保留。实际运行读取原生 PTS 索引，不用 `frame/fps` 重造时钟。

原片、截图、视频、日志、大数组和完整检查点留在外部 Data。公开源资料及离线快照边界见 [GITHUB_REVIEW.md](GITHUB_REVIEW.md)；公开源码/数值资产不意味着附带本机所有真实像素输入。

## 复现入口

在现有环境、项目根目录中构建；源码未安装时仅为本进程设置 `PYTHONPATH`：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
python scripts/build_core.py --out D:\CodexData\RinkSpace_Data\build\core
$RinkBuild = Get-Content D:\CodexData\RinkSpace_Data\build\core\build.json -Raw | ConvertFrom-Json
python -m rinkspace model preview-field --scene events/ankara_2026/scene/field/model.json --library $RinkBuild.library --out <新的Data输出目录>
```

`build.json.library` 是实际 DLL 路径。Windows/MSVC 路径已实际使用；构建不安装环境，也不启动求解。其余已实现入口为：

```text
python -m rinkspace model import-state --model FILE [--complete-state FILE] --scope FILE --library FILE --out DIR
python -m rinkspace prepare --input FILE --scope FILE --shots all|s01,s04 --out DIR
python -m rinkspace review --prepared DIR --state FILE --assets FILE --shots all|s01 --library FILE --out DIR [--compare FILE]
python -m rinkspace model import-ad-assets --input FILE --out FILE
python -m rinkspace model preview-ad-assets --assets FILE --library FILE --out DIR
python -m rinkspace model rebuild-glyphs --input FILE --library FILE --out DIR
python -m rinkspace model preview-print-installation --scene FILE --assets FILE --instance ID --camera FILE --frame FILE --pts INT --library FILE --out DIR
python -m rinkspace model preview-print-installation --scene FILE --assets FILE --instance ID --camera FILE --source-video FILE --scope FILE --pts INT --library FILE --out DIR
```

`FILE/DIR` 为调用方当前资产与新输出路径，参数详情见 [INTERFACES.md](INTERFACES.md)。字形重建需要 `advanced` 可选依赖中的 SciPy，实体图辅助函数 `domain.entity_graph.render_graph` 需要同组的 Matplotlib；它们不是 native 核心或一般几何前向的必需依赖。

同源回放、模型迁移和展示结果均明确 `fit_executed=false`；完整视频的视觉审核与新 GPU 拟合分别判断，编译、测试或模型图不能代替。

广告内部几何现已进入共同参数状态：全部模板使用共享内部组，孔/断笔与父组共同变化；尺寸、字距/组内排布、宽高和单调轮廓形变通过同一native求值。QR方框和明确矩形保持分析几何关系，配对共用模板与尺寸、安装独立。原参考模板、原片UV/PTS、相关组与人工确认均保留。prepare写自包含布局/初值，review与比较实际读取候选参数。本次只实现和验证状态与几何导数，未启动拟合，未完成生产GPU LM或完整相机因子Jacobian。
