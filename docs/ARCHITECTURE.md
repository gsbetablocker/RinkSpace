# 架构与模块职责

本文描述模块职责与依赖方向；未实现的目标边界另行标明。实际能力见 [REBUILD](REBUILD.md)，调用签名与格式见 [INTERFACES](INTERFACES.md)，数值方案见 [SOLVER](SOLVER.md)。

RinkSpace 输入有原生 PTS 的比赛原片、场景实体及观测，输出共同场地、每镜头相机、同源审阅和语义明确的运动员轨迹。赛事、节目与镜头是数据实例，其名字不得成为公共数学的分支条件。

## 程序边界

目标生产入口为一个 `rinkspace` Python 包、一个 CLI 和一个经 C ABI 调用的原生数学核心。Python 编排数据、进程和展示；C++ 控制数值生命周期，CPU 参考与 CUDA 共享几何、相机和残差定义。数值后端的选择与范围在 SOLVER 维护，当前构建能力以 REBUILD 为准。

| 模块 | 职责 |
|---|---|
| `src/rinkspace/cli.py` | 参数与路径解析、调用应用服务；不实现几何或 Jacobian |
| `domain/` | 源时钟、实体、观测、场景、机位、镜头及参数状态；不编排运行或重放历史来源 |
| `media.py`、`storage.py` | 复用原片索引、解码与 Project/Data 路径解析 |
| `observations/`（目标边界） | 原片测量、材料跟踪、实体关联；不以模型投影伪造源像素 |
| `core/native.py`、`core/gpu.py` | CPU／GPU 的薄绑定与 ABI 检查；不重写投影公式 |
| `session/` | 当前数据准备、native 模型 GPU 编译及运行编排；数据集完整 solve/resume 生命周期尚待实现 |
| `review/` | 同一核心的投影、同 PTS 对照及审阅导出；旧 review API 由包内 legacy 保留 |
| `tracking/`、`analysis/` | 按明确的身体定义、尺度与缺失状态生成轨迹和统计；具体能力另验 |
| `compat/` | 读取旧静态状态／观测并保留映射；目标是正常运行消费当前格式，现存兼容依赖见 REBUILD |
| `native/include/rinkspace/`、`native/src/` | 公共数学、C ABI 与驻留数值引擎；不读原片或识别赛事名称 |
| `native/src/core_gpu.cu`、`gpu_linearization.cuh`、`gpu_solver.cuh` | 同源前向／导数、逐点或相关目标累计、连通消元、cuDSS 与连续 LM |
| `events/<event>/` | 赛事实体、事实、安装时期、模型与节目绑定 |

依赖方向为 `CLI → session/review/tracking → domain + core + media/storage`，观测准备使用 `domain + media + core`。native 不反向依赖 Python；渲染与残差共用前向数学。同名模块与包的消费者应一并迁移，不能让两份接口各自演进。

## 数据所有权

| 对象 | 归属与边界 |
|---|---|
| Source | 原片身份、尺寸、有理时基和整数 PTS 索引；不随优化变化 |
| Scene | 同一场地／布置时期共享的实体、参数、坐标与事实；不保存逐帧相机 |
| Rig | 物理机位及安装时期、支点／杆臂和镜头族；不因型号相同而共用现场数值 |
| Shot | 源范围、Rig 引用及运动曲线；不复制场地，不跨切镜平滑连接 |
| Observation | 原 UV、PTS、实体关联、测量含义、质量及相关组；关联修订不覆盖原测量 |
| Trajectory | 身体参考定义、位置含义、单位／假设、缺失及上游模型版本 |

固定交点由父实体和已确认分支生成，没有逐帧自由 station；曲线样本的 station 留在真实支持域；跨帧同一材料点共享其位置。候选对应不冒充已确认固定点。具体类型和跨表面交点约束见 INTERFACES，材料测量含义见 [BUILD_SPEC](rink_model/BUILD_SPEC.md)及所属赛事。

旧记录与机器观测的准入按同一[来源与准入要求](rink_model/BUILD_SPEC.md#observation-eligibility)处理。等价迁移保留原状态，不因导入而激活；也不因 legacy 或非人工采集标签而一概否定。

## 当前输入、数值状态与结果

PreparedDataset 保存当前场景／相机定义、源索引、实体和观测数组及相关组。它是可独立消费的运行输入；精确 schema、数组布局和引用规则只在 INTERFACES 维护。原相关 W、观测行序与单位保留，不能压成互相独立的 sigma。

数值状态、优化器接受步与用户验收是不同对象。完整 checkpoint 需能恢复同一输入、参数布局、局部变量与优化器状态；失败 trial 或半写文件不能覆盖最后完整状态。运行行为、日志和禁止历史重放的要求统一见 [SOLVER 运行要求](SOLVER.md#run-contract)。

场景接入见 BUILD_SPEC，准确性及完整镜头判定见 [VALIDATION](VALIDATION.md)，本机文件与构建入口见 [MAINTENANCE](../MAINTENANCE.md)。镜头全集、排除范围和每次选择读取所属节目清单，不在架构文档复制时间表。

## 迁移边界

[MIGRATION.json](MIGRATION.json)保存旧模块到目标职责的映射，供修改相关模块时查阅，不是开工、删除或重跑的指令。可用的媒体、存储、原生公式和完整状态优先复用；转换保留原字节身份与映射，不通过静默简化未知字段制造“等价”。

旧初始化与新场景的差异应显式展示。未实现的数值能力明确报不可用，不用空函数、模型预览或合成演示宣称已经完成真实求解。工程验证与真实画面验收分别记录。
