# 维护与开发

公开阅读从 [README](README.md) 开始。源码、环境/构建定义、模型参数、人工事实和必要审阅归档保存在项目；原片、帧图、大数组、完整检查点、缓存、日志和诊断视频保存在配置的外部 Data。目录分工见[架构](docs/ARCHITECTURE.md)。

## 本机工具

| 工具 | 用途 |
|---|---|
| [INSTALL_LOCAL.cmd](INSTALL_LOCAL.cmd)、[START_LOCAL.cmd](START_LOCAL.cmd) | Windows 基础安装和本地服务；安装器选择 dev/acquisition，不含 advanced |
| [START_POINT_EDITOR.cmd](START_POINT_EDITOR.cmd)、[PowerShell 参数](START_POINT_EDITOR.ps1) | 指定 Dataset、Media、SourceMedia、Model 和 Port，编辑有源身份的逐帧观测 |
| [refresh_project_entry.py](scripts/refresh_project_entry.py) | 刷新 Data 中的本机结果面板，保留 CURRENT_TASK 与冻结结果绑定 |
| [START_FAST_SOLVER.ps1](START_FAST_SOLVER.ps1) | 现有样本配置的便利入口；输入、完整状态和后端须与所选 profile 相符 |
| [solver_runtime](calibration/solver_runtime/README.md)、[rinkfit](calibration/rinkfit/README.md) | 求解与后端接口 |

共享资源映射保存在 [storage.defaults.json](configs/storage.defaults.json)。本机 `configs/storage.json` 被 Git 忽略；新机器可复制 [storage.example.json](configs/storage.example.json) 并填入绝对 Data 根，或设置 `RINKSPACE_DATA_ROOT`。显式 `AssetStore(root, data_root=...)` 优先于环境变量和本机文件；未配置 Data 时，通用工作台只使用受忽略的 outputs，本机 pytest 缓存使用系统临时目录。AssetStore.project_entry() 将面板定位到 Data/workbench/START_HERE.html；Web 服务的 / 路由在该页面存在时显示它，否则使用通用工作台。审阅页按 source_datasets → results 定位到 Data，源观测和人工确认仍留原目录。

旧根页面、镜头专用快捷方式及历史交接材料在本机保留并忽略，不作为公开入口。它们可能仍被已有对话使用。刷新器可从旧根页面首次初始化 Data 面板；本机无冻结绑定时无需生成该面板，基础 CLI 和通用工作台可独立使用。

## 第一镜头原版恢复与当前广告修复

本来的Ceres时期第一镜头d78已原样找回。原始模型和完整状态JSON分别在 `events/ankara_2026/programmes/okada_mei_sp/s01_original/original_model.json`、`original_complete_state.json`，与当时原件逐字节相同；大数组、原代码包、日志与原视频仍在Data，索引为 `Data/runs/solve2_s01/recovered/restore_manifest.json`。可直接消费的原数据为 `recovered/d78_input.json`；原W与预测独立重构值和历史报告差0，不把重估候选伪称含完整PHR对偶状态。

原样新前向包 `Data/runs/solve2_s01/ceres_regression/d78_exact` 保留当时的完整相机与1252组控制。当前复用实际生成d78的原primary、完整W、有限曲线RS回算和原GPU接口，只做S01（官方原片20.02–44.96秒，原生PTS256256–575488，共1248曝光）。静态广告修复基准中的原相机、冰面和围挡保持原值；38个非广告参数块和77条材料曲线的保护SHA为 `160b6fa248719480e8684aad8ef8382f3a9f6f9b2f44e0ce7e4e54f452c317fd`。不复用后来改坏冰线的暖链，也不续算已被真实画面否定的新求解层。

当前完整广告源目录为 `Data/runs/solve2_s01/ads_original/source_original/ready_model`。已将既有Ad2–Ad12细节实际接回原位置；Ad3修既有源图转回原相机后的大字偏移；Ad4修中央图案、ANKARA和JUNIOR GRAND PRIX两组大字；Ad5修周年标志轮廓和内部排布；Ad6修源坐标转换遗漏，并用31.00秒清晰材料替换タ／ル／バ模糊合并轮廓。Ad4右边与Ad6左边两条无观测的demo辅助线已退出实际场景，改为原片蓝底／红底实体边，保留运动模糊范围且没有新增测量或权重。这些静态修正保持整块位置和原相机、冰线、围挡不变。Ad7仅用24个真实固定角点的小修经34.92与39.96秒画面局部复核；原24角工作模型保留；57条实际逐点核实的串片错绑改正后，用326个清晰源跑原W得到 `Ad7_rebound57/solve01/full2/model.json`，已按两次经过的完整大字画面局部复核，作为当前Ad7工作位置。所有原UV/PTS/E/σ/W保留，未核实旧映射不因全图启动而复活。

Ad6原数值六步虽然cost下降，但真实整字画面退步，已全部拒绝并保留在 `Ad6_orig/continue01`；当前保持原安装位置。此前新层159步、Ad12新层996步及Ad12原W局部尝试均保留历史状态，不代表当前活动模型或运行。以上修正已合入 `source_original/native_ready/model.json`，全部659条真实材料路径，原相机、冰线和围挡保持d78原值，原块仅Ad7采用经过局部图审的工作值。完整第一镜头原primary/W/full2已经实际启动：`Data/runs/solve2_s01/full_original/solve01/main_restart01.log`，只含步数与cost的文本为同级 `step_cost.log`。完整物理图包括相机、冰面、围挡、全部广告对象和5个跨帧共享冰线位置，使用原GPU接口及20CPU线程；当前运行阶段以s01_original/index.json和实际日志为准，尚未通过完整求解与整镜头验收。用户要求此镜头通过后直接接第二镜头，并继续第一、第二镜头连续求解。高清模型全部保留，逐曝光按真实清晰度使用细节，模糊画面不硬造细边。后续镜头未解锁。

完整原求解已真实完成第一步：`full_original/solve01/i00001_retry01/solve_retry01/full2/model.json`，cost 从570502.4647123809到429628.77703457524，277564活动源全部严格RS有效，897个Query回算完成。该完整步自由调整相机、冰线、围挡与广告；没有把单广告冻结掩码冒充完整求解，原d78基准仍原样保留。这份同模型1248帧已按序实际核查：26.46秒、29.16秒C0蓝圆比原baseline分别偏开约33px、17px，故该数值候选已明确视觉拒绝，未替换原结果。根因之一是旧转换将C0改作宽度中线并整类退役，令3018条原outer缺映射；现按真实圈实体外边恢复2993条及197整原W组/4条原独立来源（rank975），25条交点替代继续退出。相机、冰线和全部物理值已回到原baseline，当前数据在 `full_original/solve03/input_c0_hole02`；等待剩余具体广告错归属修正后跑下一完整步，尚未通过整镜头验收。主日志为 `Data/runs/solve2_s01/full_original/solve01/main_restart06.log`，易读步数与cost为同目录 `step_cost.log`。Ad5的具体错字/拖影点和Ad4误挂到踢脚条的点按稳定ID修正，改变W源范围后的下一scope成本单独重建，撤点造成的下降不计为优化步。

此前控制台按用户明确要求已经撤掉：8787服务已停止，节目 `control_surface.enabled=false`，入口会直接退出，不再自动打开或展示。正常求解直接消费当前自足输入、模型、配置和完整状态；不得重放历史生产或资格链。每次实际启动求解须提供本次主日志绝对路径，给用户的易读日志仅含步数与cost。以下实现与运行说明全部属于此前停用过程的历史记录。

## 此前新求解器与控制台记录（已停用）

按用户当前要求，旧联合求解器与旧子代理已停止；新设计和实现位于 [calibration/solve2](calibration/solve2/)。当前只做S01（官方原片20.02–44.96秒、原生PTS256256–575488，共1248曝光），S02–S09硬锁。目标是冰线、围挡、广告与网格在完整真实镜头中贴合，并以未参与调参的真实证据验证；数值下降、测试通过、正常退出和局部抽帧均不是验收。

本机控制页为 http://127.0.0.1:8787 ，启动命令为 `.venv/Scripts/python.exe -B calibration/solve2/main.py --port 8787`，控制服务主日志为 `D:/CodexData/RinkSpace_Data/runs/solve2_control/server.log`。控制服务与数值进程分开；常驻俯视图始终显示同一实际模型的全部冰线、广告编号、13个固定材料位置与三个机位。固定点可点击查看共享定义；5个C上未知世界位置由feature各自全曝光共用，8个Ad7角点由源图实体角和共同安装派生，不把未知位置标成已知数值。主机位光心随选定原生PTS由固定云台与杆臂推导，另两机位仍明确为历史光心初值参考，不能冒充已拟合安装位置。

[模块与数据流](calibration/solve2/architecture.json)、[事件约束台账](events/ankara_2026/solve2/constraints.json)、[数据契约](calibration/solve2/contracts.json)、[节目绑定](events/ankara_2026/programmes/okada_mei_sp/solve2/input.json)构成当前接口。页面区分用户事实、原片证据、数学定义、工程预算、未批准假设及是否实际绑定运行；同一模型与参数SHA在监控中可核对。参数通过 `/api/model/parameters` 原子生成新状态，旧revision冲突返回409；原片取点面板可查看精确源曝光、UV、实体关联及退出原因，并按稳定源ID可恢复撤点。撤出完整W组中一个成员时整组退出，保留原矩阵与所有源数据；普通恢复不能绕过实体身份未核实的基础退出。

本次恢复参照是Ceres时期第一镜头已较贴的冰线效果，对应d78完整物理初值 `Data/runs/solve2_s01/d78_init`；单独保留的历史demo平面数据 `Data/runs/solve2_s01/baseline_init` 不代替这次参照。当前共同候选是节目绑定的 `Data/runs/solve2_s01/print_chart_candidate`，模型修订11；实际接受步发布到同包generation和current指针，不能用静态初值SHA冒充当前参数。广告以有限真实图形和每对象共享静态拼版放到连续展开围挡，允许跨直线与圆弧；不使用每帧广告变形吸收相机误差。Ad7高清字形和固定角点已接入，Ad8未核实小字1320条、Ad5具体错关联82条、Ad2具体错关联8条、Ad4具体串字关联50条退出当前拟合，原UV/PTS/ID及失败记录保留。

S01原始355554行在 `Data/runs/solve2_s01/observations/manifest.json` 自足交付，当前准备数据通过 `preparation/delivery.json` 指向g12。正常运行直接消费当前数据、模型、配置与完整数值状态，不打开旧packet、不重放历史来源链。原W统计永远作用于预测与原UV之差；原Query的Q(alpha)单列精确等式；工程像素预算E不是σ，每个活动源均受E约束。第一阶段最小化一个全局共享预算b，第二阶段固定b=1细化原W统计。S01内按冰线围挡、固定相机广告、全源联合的数值顺序推进，不改变实体许可，也不能跳过最终全部约束与全镜头验收。

新Ceres2.2批量稀疏桥、显式PHR约束外层、CUDA相机核及连续围挡核已经实际接入。完整g8残差与雅可比一次约3.92秒，冰线围挡阶段约0.90秒；这是整体实测，不拿单个GPU核时延冒充求解速度。相机182017行比对、围挡286034行与原数学逐点比对及方向导数检验保存在 `Data/runs/solve2_s01/model_core` 和 `optimizer_preflight_g8_gpu`。

真实S01冰线围挡求解已运行并发布实际接受步；完整运行、失败和状态迁移保留在 `Data/runs/solve2_s01/warm_geometry`。本次数值工作固定g8几何输入，完整广告准备升级不触动它。圆线17格初值落错盆地已用同一有限分支的可靠多初值修正；形式RS行辅助根与合法station/row联合恢复分别受真实PHR/E不恶化检查保护，原像素、实体和物理参数不变。仍无合法联合根的点保原等式，不能夹边、删除真实灰圈或伪称完成。

已用真实试步定位：大b下原W未参与的阶段，以及原W本身的低秩空方向，均可能让旧数值度量沿整条曲线或弱物理方向跳飞。新信赖域用完整二维像素投影的非零支持RMS构造数值单位，每inner固定，并保存实际尺度、参数SHA与尺度SHA；不使用W/σ/E作度量权重，不添加先验、不冻结变量、不减少自由度。临时固定b=5494.858的原W暖阶段已根据真实画面和同物理圆的反例明确结束：F3两个真实极点仍约107/92像素偏，原W可单靠沿圈站点下降；不称数值收敛或几何通过。旧完整态留在 `objective_history/statistical_level_set_before_return`，现保持x/λ/ρ与同fullUV度量，回到原共享min-b可行性收紧，随后必须在原b=1细化统计。最新真实启动主日志为 `D:/CodexData/RinkSpace_Data/runs/solve2_s01/warm_geometry/return_feasibility.log`，实际状态以控制页监控和current_run.json为准。

Windows监控读取曾令MoveFileEx/os.replace拒绝替换current_run.json，已实际保存失败并修复。公共 [atomic_io.py](calibration/solve2/rinksolve/atomic_io.py) 使用共享删除读取和经本机验证的ReplaceFileW原子发布，并处理短暂共享冲突；完整checkpoint使用同一模块。真实Windows四项回归覆盖保持旧快照、旧读者短冲突恢复、永久只读错误保旧并报告、坏JSON契约。非关键监控发布冲突会记录告警并重试，不中断Ceres；模型和完整checkpoint的永久错误仍显式保状态处理。

当前S01有活动广告源的是Ad2–Ad12。只有Ad7的曲线实际接入官方高清模板；其余素材候选存在，字体与有限轮廓仍逐对象核对，不能将素材文件存在称为已接入投影。Ad7的完整typed源组件→内容拼版→既有安装关系已在detached候选中补齐，所有几何和参数不变，尚未切换当前模型。全S01时间范围的原片抽帧审阅仍见F3灰圈与广告偏移；未入拟合的门板固定角点仅作一次位置锚定及其它曝光条件验证。全部1248曝光已有其它拟合源，不能把这些时刻冒称时间留出。尚未通过整镜头验收，不生成无明确改善的整段视频，不解锁后续镜头。

当前控制面板右侧源时间滑杆下可打开“查看此帧原片与当前网格”。预览按需读取一次冻结的完整当前模型，全部标线、广告边界、编号与网格全开，可切原片并放大；图内及元数据保留该图模型/参数修订与SHA，数值后续接受步不会偷换此图身份。总览有新状态时界面明确提示刷新，单帧不作为全段验收。线上PNG与同一冻结参数的离线完整叠图已作实际原片逐字节对照，报告 `D:/CodexData/RinkSpace_Data/runs/solve2_control/projection_preview_verification.json`；错误原生PTS明确拒绝。

原全读出单窗口充分证书曾在同一r538末帧只认证90个画面内根，导致网格大片消失。新证书对完整原读出用四个闭时间子区间，逐区间取真实样条导数上界并还原原tau/高度系数；窗口仅向外微重叠处理浮点接缝，物理读出、数值根和几何均不变。原数值UV/row位级回归、正负零tau导数界、全部1248时刻窗口覆盖通过，9444个画面内根获证，仍无证的957点保留失败。此修复恢复显示能力，不是冰线贴合改善；报告 `Data/runs/solve2_s01/current_review/rs_domain_audit/regression.json`。

Ad3实际SVG内容拼版及13952源行的detached准备候选在 `Data/runs/solve2_s01/ad3_binding/prepared`，未发布当前。13893绑定继承原资格，59未定源退出；12个rank0相关源family保留完整成员与空矩阵，其余1142条有效工程预算不被连坐撤出，正秩W闭包为0。正式接入须由factory保留incoming latest参数，冻结r436只是本次审查见证。

## 旧六镜头运行记录（已停止，仅历史追查）

摄像机已确认事实：用户向摄制团队询得并明确确认，长边主机位、长边副机位、低位端部三台均为 **Sony PXW-Z190**。当前对应 `main→S01/S04`、`secondary→S02/S06`、`end→S03/S05`；三个独立固定三脚架安装、非零相机局部杆臂、姿态驱动投影中心属于已有安装前提。来源见 `events/ankara_2026/model_constraints/camera_hardware/user_camera_mount_and_height_20261006_v4.json` 与 `configs/camera_hardware/okada_sp_installations_z190_v1.json`。型号已确认；厂家未给出的现场读出时长、畸变标定和开关状态不可混同为“型号未知”。上述来源供工程追查，普通运行不打开历史证据。

当前输入为 `D:/CodexData/RinkSpace_Data/runs/okada_joint6_1021b/current_input_g6a/manifest.json`（SHA `cdfa8c2bfd1c00da3b64ffd5e82ded9892b8a1b1f564c8682b1e0ff96ef1ae51`），原始域4,307,952行，当前拟合代表815,177行。在g5上撤回14,450条缺少独立F2圈层实体证据的dense56读数（其中9,704条为拟合代表），追加30条原生材料顺序闭合的连续围挡读数；旧UV/PTS/ID、Query/W及全部原始记录保留。初始状态来自r14g5完整candidate，token `8d9dd38e4ca7510b`；模型、旧全raw辅助前缀、路径、cell、运动与全局乘子/罚因子精确保留。新增数据使用显式当前目标迁移，只重绑来源身份，不清空对偶值。149个当前资产与30条真实49样本剖面独立核验见 `geometry_error_1023/g6a_independent_check.json`。

普通启动直接运行 exe，只消费当前数据、配置、模型、完整检查点与运行库。原隔离验收已取消，`file_guard_io.exe`、`--current-reference`、历史资格/冷验证回执均不进入普通启动。运行库路径在当前启动器中明确给出，后续启动无需打开构建recipe或源码集合。

当前运行入口为 `launch_r15bg6a.py`，配置 `fit_current_r15bg6a_config.json`，主日志 `D:/CodexData/RinkSpace_Data/runs/okada_joint6_1021b/fit_current_r15bg6a.log`。原生程序修复条件支撑秩被当作长期硬冻结的问题，保留原模型顶点自由度；新增当前专用PHR迁移接口，普通入口继续拒绝历史导入键。此前 `fit_current_r15g6a` 在加载前因旧迁移键被拒并退出，失败日志保留。真实六镜头仍未通过物理、画面或正式投影验收；固定开发对照 `geometry_error_1023/r14trace_fixed/index.html` 覆盖六镜头15个原始曝光，仍有S06板边共同上偏与S03广告错位。

运行所需文件：

- 当前原生程序 `native_current_1023/artifacts/48abced1f835022d/rinkfit_joint_fit.exe`，实际Ceres/glog/gflags/CUDA运行库及配置绑定PTX。
- `current_input_g6a/manifest.json`、`packet.json`、`source_model.json`、`initial_checkpoint.json`、`source_append.json`、`frame_roles.json`、`source_identity.json`、`numeric_domain.json`和这些当前文档引用的 `a/` 数组。Query、W、运动约束、原UV/PTS/ID及完整辅助状态保留。
- 历史诊断运行 `fit_current_r11c` 已停止并保存完整候选（没有接受任何实际更新步），主日志 `D:/CodexData/RinkSpace_Data/runs/okada_joint6_1021b/fit_current_r11c.log`；配置 `fit_current_r11b_config.json`，启动身份见 `fit_current_r11c_launch.json`。五次试步均被拒绝、100次试点评估（此前缺失败因子记录，不能称全部有限）；尚未修复下降方向问题，未通过图像、物理或正式投影验收。
- r10c坏候选及完整状态保留，token `bfb5b9946a84d7f0`，只作开发基线。r10d未启动。r11因签发器顶层计数遗漏、r11b因新支撑工作集局部槽越界，均在迭代前退出；对应日志和产物保留，两处根因已修。

历史首步失败诊断 `fit_current_r12_failure` 已结束（退出码0，仅诊断，无实际接受更新），主日志 `D:/CodexData/RinkSpace_Data/runs/okada_joint6_1021b/fit_current_r12_failure.log`；首提议、基点及参数顺序已完整保存。复用保存状态，仅离线复算关联运动行和三个光学域，确认：完整试步先触发S06运动投影无效；缩到1e-6后该运动行已有效，但main、secondary的原Brown域仍失败，secondary缩到1e-12仍不合法。证据 `domain_fix_test/first_failure_local.json`。该诊断定位了闭光学域可行方向问题，后续已接入工作集；不以增加回溯、放宽域或盲目续算代替。`fit_current_r12_direction` 为此前诊断，保留。两者均未通过质量验收；每次新启动先提供主日志绝对地址。

继续已有结果时，`production_config.json` 的 `current_checkpoint_v1` 绑定该结果实际保存的 `candidate/current_checkpoint.json`（文件及 SHA），CLI 模型使用同一 candidate 的 `model.json`；该 current checkpoint 已包含对应原生检查点与全 raw 辅助量，不另建一套恢复身份。当前仅说明 observed 求解入口，trueRS/formal 转段及最终贴合尚未完成。

Python 编排仍可调用 `rinkfit_joint.dense_joint_run.prepare_run(spec)` 和 `execute(spec)`；spec 明确给出当前 model、packet、native、configuration 的文件及 SHA、输出目录与 CUDA 缓存。缺少 `current_input_v1` 时常规入口报输入尚未准备；历史全链重放仅保留为显式离线审计工具。当前入口保留数据结构、有限值、实体许可、资格、完整 W 与状态一致性检查，数值下降或程序退出不代表六镜头最终贴合。

## 检查

~~~console
python -B scripts/check_onedrive_paths.py
python -B scripts/check_github_readiness.py --data-root <外部Data绝对路径> --output <Data内报告绝对路径> --review-only
~~~

路径检查包含被忽略的环境和缓存。公开检查复用已有大小、LFS、凭证候选及本机生成物检查；检查报告写入 Data。修改功能时选择相关测试，基础范围见 README，高级依赖见[环境说明](environments/README.md)。

旧地址的精确文件别名见[路径映射](configs/path_aliases.json)，可选 ResourceRegistry 的限制见[资源接口](docs/PORTABLE_RESOURCES.md)。公开派生文件与原运行字节的区别见[数据身份](docs/GITHUB_REVIEW.md#数据身份与缺失材料)。Git 数据库继续留在本机 Data；本机维护记录不进入公开阅读路径。

公开前按 [LICENSE](LICENSE.md) 与[许可核查](docs/LICENSING_REVIEW.md)核对实际内容。公开树和归档不含比赛视频、截图或叠加图；来源链接与 PTS 可公开，原始素材只留本机。
