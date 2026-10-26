# 实现接口：现有入口与目标合同

版本：2026-10-26。native ABI 2 的完整几何/时变相机前向（含隐式 RS）、prepare/review 及下述 D1/D3 入口已实现；同一 resident handle 已接通连续 GPU LM、安全停止及继续；数据集命令和磁盘完整 solve/resume 生命周期仍是目标合同。CPU 回放不表示 GPU 拟合。算法与参数设计见 [SOLVER.md](SOLVER.md)，当前实现状态见 [REBUILD.md](REBUILD.md)。

## 1. 现在确实能调用什么

在项目根、已有 Python 环境中执行；源码未安装时仅为当前进程设置 `PYTHONPATH=<项目>/src`。

```text
python scripts/build_core.py --out DIR [--cmake EXISTING_CMAKE]
python -m rinkspace model import-d78 --model FILE --relations FILE --out FILE
python -m rinkspace model preview --scene FILE --library DLL_OR_SO --out DIR
python -m rinkspace model import-state --model FILE [--complete-state FILE] --scope FILE --source-binding FILE [--shot ID] --library FILE --out DIR
python -m rinkspace model preview-curling --scene FILE --library FILE --out DIR
python -m rinkspace model preview-field --scene FILE --library FILE --out DIR
```

构建目录在源码树外。CUDA 默认指定 `--cuda-architectures native`，可显式选择目标架构；`--cuda-fast-compile 0|min|mid|max` 仅选择编译器优化开销档位，默认 0，实际选择写入 build.json，不启用 fast-math。构建输出 `build.log/build.json`，后者的 `library` 是实际库路径；默认 CPU 构建为 `cuda_built=false`、`nonlinear_solver_implemented=false`；显式 CUDA + cuDSS 构建两者为 true，磁盘完整续算仍未实现。`import-d78` 输出 `rinkspace.scene.seed.v1`、状态为 `unfitted_structure_candidate`，不是保持原投影的完整 d78 状态；`import-state` 才导入冻结镜头的完整状态与实体供回放，同时保留九镜头登记，不补造其他镜头的相机。各模型预览输出图、几何和回执，不拟合。已有 `doctor/probe/index` 等命令维持原语义。

`import-state` 默认要求显式 `--source-binding`，认证原 observations、media/input_manifest、视频和原时间原点；同源子窗口不重置 controls 时间原点。显式 `--unverified-replay` 只生成未验证回放状态，不能作为 prepared fit 相机。

F10 的完整生产求解／磁盘续算未实现；F30 已保留候选资格适配及读取接线，但未完成真实节目闭环验收。两项实际改动和测试边界见 [REBUILD](REBUILD.md) 的 F10／F30 记录。

Python 入口为 `rinkspace.core.native.NativeCore(library_path)`；方法签名见 [core/native.py](../src/rinkspace/core/native.py)，C ABI 唯一定义见 [core_api.h](../native/include/rinkspace/core_api.h)。ABI 2 使用 float64、行优先、显式长度和调用者分配输出。同一 ABI 号不保证旧库有所有新增符号，绑定需检查实际能力。

`camera[22]=R[9],C[3],fx,fy,cx,cy,skew,k1,k2,p1,p2,k3` 只代表静态曝光；`board[6]=half_length,half_width,rx,ry,exponent,height`。运动相机经 project_motion 使用完整状态及隐式预测行 RS，不使用静态 camera 数组替代。

RS 先认证整个源读出区间的深度、光学域及行映射收缩界；整段充分上界过于保守时，用中点的解析相机坐标速度及区间加速度范数上界分别包住三个方向的速度和位移，并以覆盖该区间的二分子区间重新认证，最多细分 8 层。每一叶区间都须通过原条件，导数比例始终使用完整 `tau/(height-1)`；任何未认证区间仍返回状态 9，真实光学域失败仍拒绝。CPU 与 GPU 共用这一路径，不改变行根方程、观测、物理参数或条件阈值。

单点包装失败抛 NativeError；批投影返回逐点状态，失败 UV 为 NaN，调用方逐项处理，不能钳回有效像素或删除失败请求。状态由头文件和 rs_status_message 维护。前向接口不更新参数；连续 LM 使用 `rs_g4_solve`，薄绑定与当前模型入口见 [native 调用说明](../native/README.md#当前模型的-gpu-求解入口)。

D3 已实现 `model preview-print-installation --scene FILE --assets FILE --instance ID --camera FILE --frame FILE --pts INT --library FILE --out DIR`。消费当前自包含模板/参数与已迁入相机，输出同一实例的 `unfolded.png/three_d.png/source.png/source_full.png/geometry.json/result.json`，不重新导入历史来源、不拟合。`domain.print_assets.initialize_instance` 只显式初始化共用 log 尺寸块及所选实例独立安装块，拒绝覆盖不同现值；未定位实例求三维坐标会明确失败。`sample_instance` 对参考材料路径及形变分界加密，通过共同 `evaluate_print_path` 分派到native形变、方形、矩形或不形变路径，展开变换和BoardCourse共用同一C++实现。安装参数为 `[s_origin,z_origin,rotation,width,height]`，不按直/弧分段限制安装。字形、内孔、开放支持和原观测保持不变；采样不生成观测。当前ABI2使用新增能力前检查对应symbol；版本号相同不表示旧DLL具备所有新函数。Ad2 的 source-UV 方向通过安装旋转 π 表达，没有再翻转原顶点。

**D3 来源核实：** `--frame` 仅用于任意图片的未核实预览，输出明确 `native_source_verified=false`，不宣称其对应相机源曝光。需要核实原曝光时使用下面的互斥入口；它核对 scope 中既有原视频 SHA、源尺寸/时基/镜头区间，并精确解码请求 PTS，保存 `decoded_source.png`。相邻曝光不能代替缺失的 PTS，来源核实也不代表视觉拟合通过。

```text
python -m rinkspace model preview-print-installation --scene FILE --assets FILE --instance ID --camera FILE --source-video FILE --scope FILE --pts INT --library FILE --out DIR
```

`--assets` 实际内容必须与 scene 当前登记的资产 SHA 相同；多个引用的实例 ID 不得重叠。孔洞父引用采用完整当前 path ID，不能在显示时静默退回自身。`domain.print_assets.sample_instance(..., stations_by_path={path_id: stations})` 支持指定共同物质采样位置以比较不同尺寸候选，仍在 native 求值，不插值像素。

`compat.joint_camera.import_joint_camera` 可只读取所需镜头的完整保存相机。显式 `control_coordinates` 按块与 offset 读取已保存的三元组，不用失效的别名块，也不重新拟合时间轨迹。已知 wide Brown hinge 仅在全部已分配 log-f 控制值均不低于激活边界时等价导入基础前向，并保存原定义与整个 cubic control hull 上不激活的依据；仍有作用或未知的扩展明确拒绝。该旧导入接口的限制仍保留。当前 `session.gpu_geometry.complete_camera_rig` 会将已经保存的原始扩展显式打包为 rig43：选定 log-f 结点加四个平方 hinge 系数；native GPU、隐式预测行导数、联合更新和审核前向均使用完整模型，不能把旧的不激活证明当作继续省略扩展的许可。

### 原片连续取点入口

`python -m rinkspace observe read --request FILE --out DIR` 调用已有 `SourceRinkReader`，使用配置中登记的现有 GPU 环境与正式权重；请求与实际验证见[节目记录](../events/ankara_2026/programmes/okada_mei_sp/source_reader_audit.json)。请求中的 sequences 使用原生整数 PTS；首帧可读取已有人工接受包，后续帧用双向光流有效性传递身份、导航点和法向，再从当前 RGB 读取材料。实际切镜重置记忆，导航不是新观测。

公共入口已接入当前 native 实体语义、既有线／圈联合读数和 `SourceFeedbackRead` 的实际使用回执。围挡正面读取蓝带上沿与黄带下沿；只见顶面时使用朝冰边缘，顶面、外侧设备和人物不能获得广告身份。现有官方 S06 顶面证据只在源 SHA、整数 PTS、时基及原生 RGB 哈希全部一致时保留原 UV 与相关身份；新环境的剖面复读另存诊断，不能覆盖它。其它画面缺证据时保留未知。

该入口不调用求解、不改模型几何，也不把已读取知识条目冒充已使用。大数组复用现有无损 NPZ 存储，输出包含逐帧读数、知识消费回执、实际模型版本及 native 引用。旧 `observe audit` 对应整个已撤销的 O 工作单，不能继续作为正式取点入口。当前验证包含 S01 连续帧、S02 切镜和 S06 近侧顶面；完整目标仍是 S01–S09。

## 2. 完整节目范围与源时钟〔prepare/review 已接入〕

源与范围读取所属节目 scope，当前样本见[节目入口](../events/ankara_2026/programmes/okada_mei_sp/README.md#源与范围)。公共代码不硬编码镜头数、帧率或时基。物理边界与请求处理区间分别保存，处理终点不是切镜。

区间统一 `[first_pts,stop_pts_exclusive)`，曝光身份为 `(source_id,int64 pts)`，读取实际 PTS 索引。计算秒先减参考 PTS，再乘有理 time_base；JSON 保存整数分子/分母，不用 frame/fps 重造时钟。

已实现的 prepared/review 及未来 run 合同均保存 `required_shot_ids`、`selected_shot_ids`、每镜头处理区间。子集输出 `scope_status=subset`；全集也只有覆盖所有必需曝光才可标 `scope_status=complete`。这表示覆盖完整，**不表示拟合或用户审核通过**。缺镜头/缺曝光列出 ID 与范围，不能只保存 selected 数量后写“完整节目”。

prepared/run 的覆盖指源 PTS 登记及相机状态定义域覆盖，不要求每帧都有人工点或活动观测；review 的覆盖还要求相应输出实际包含这些曝光。不得借完整范围要求重新逐帧人工标注。

## 3. 统一 CLI 与编排接口

D1 已实现的来源适配接口：`compat.ad_assets.import_mixed_ad_template(path,current_source)` 返回来源、模板、实例、逐轮廓观测候选、别名及原始provenance；`append_source_atlas(assets,path,source,frame_sha256=...)` 把已有原生字形关联同名模板，不生成缺失的material坐标。复制别名不再登记测量；所有新迁入候选不因导入获得拟合或人审资格。`model import-ad-assets --input FILE --out FILE` 与 `model preview-ad-assets --assets FILE --library FILE --out DIR` 已接唯一CLI。它们只导入/查看，不调用优化器；输出已存在时明确拒绝覆盖。

D1 预览消费统一模板和原有逐曝光记录，审计包已验证 Ad1、Ad2、Ad3 的预览，保留外轮廓、内孔及已声明解析图形。缺图、缺原像素、缺直接观测关联会列入输出，不借模板投影补齐。不同曝光必须使用逐 observation 的 PTS 及匹配的 source.exposures 记录，不能把 registry 顶层首曝光套给所有行。

使用现有 `src/rinkspace/cli.py`。`FILE/DIR` 为路径参数，`--shots` 接受 `all` 或逗号分隔的清单 ID；未知/重复 ID 报错。当前已实现：

```text
python -m rinkspace prepare --input FILE --scope FILE --shots all|s01,s04 --out DIR [--library FILE]
python -m rinkspace review --prepared DIR --state FILE --assets FILE --shots all|s01 --library FILE --out DIR [--compare FILE]
```

以下是尚未实现的目标命令，当前 CLI 不接受它们：

```text
python -m rinkspace solve --prepared DIR --state FILE --options FILE --library FILE --out DIR
python -m rinkspace resume --run DIR --library FILE
```

| 命令 | 状态 | 输入与输出 |
|---|---|---|
| `prepare` | 已实现 | 当前输入清单与源范围 → 自包含 prepared 目录；校验所读数组与当前合同，不重跑历史生产链。增量依赖缓存尚未实现 |
| `review` | 已实现 | 同一 prepared/状态、资产映射与可选对照状态 → 同 PTS 的 `review.json`、静帧和对应完整镜头视频；不优化、不平滑输出 |
| `solve` | 目标 | prepared、初值、RunOptions、已编译库 → 新 run；GPU 不可用报错，不回落为 CPU 优化 |
| `resume` | 目标 | 原 run 的输入引用与最近完整 checkpoint → 同一 run 追加日志及新 checkpoint；不重新 prepare、不改变目标 |

当前工具拒绝覆盖已有输出，`review` 只能选择 prepared 所含镜头；`--compare` 检查相同源曝光和实体库存，缺项明示，不静默取交集。求解目标合同中，恢复通过 `resume`；数据/目标/布局变更形成新 prepared 和新 run，可显式迁移兼容初值。

已实现的 Python 薄编排接口：

```python
session.prepare.prepare(input_path, scope_path, shot_ids, out_dir, *, core=None, local_charts=None) -> PreparedReceipt
review.render.render(prepared_dir, state_path, assets_path, shot_ids,
                     library_path, out_dir, compare_path=None) -> ReviewReceipt
```

`review/` 已替换旧 `review.py`，旧 API 通过包内 `legacy.py` 保留。`session.solve.solve(...) -> RunReceipt` 与 `session.resume.resume(...) -> RunReceipt` 仍是目标签名，相应模块尚未实现。

`PreparedReceipt` 返回 `manifest_path,dataset_id,revision,selected_shot_ids,scope_status`；`ReviewReceipt` 返回 review 对象及 `review_path`，路径为绝对路径。目标 `RunReceipt` 返回下面定义的 result 对象及 `result_path/log_path`。当前 CLI 捕获失败并返回非零；后文的结构化求解错误和完整退出码映射仍是目标，不能视为已实现。

`domain.scene_assets.snapshot_scene` 保存当前 scene 及其显式广告数值资产；`domain.runtime_rigs.snapshot_runtime` 保存 rigs/shots 及其显式 camera/controls 依赖，重写包内相对路径，保留数值与原 NPY 字节。prepared revision 绑定这些实际当前内容、features、观测元数据、查询和配置；不只依赖调用方手写 revision 标签。不读这些文档中的历史 origin 或重新生成来源。

`review` 支持 `rinkspace.field_model.v1`，沿统一 scene 资产引用显示已定位广告，未定位实例进入 missing 清单。已有连续围挡上下边不会再次补画。源视频每次 review 按唯一 source 校验一次已登记 SHA；同源多镜头不重复扫描该视频。预览字体优先使用已有系统字体，可用 `RINKSPACE_CJK_FONT` 指定现有 CJK 字体文件，不在渲染时下载。

`model rebuild-glyphs --input FILE --library FILE --out DIR` 已实现，其 SciPy 依赖属于 `advanced` extra；`domain.entity_graph.render_graph` 的 Matplotlib 同属 `advanced`。一般 native 几何/回放不需要这两个辅助依赖。

### 冰面线与围挡底边的固定交点

现有 `Feature(kind="fixed_junction", surface="ice", measurement="material_point")` 可显式声明 `junction_relation="ice_board_bottom"`。两个父 Feature 必须分别属于 `ice` 和 `board`，后者明确标识 `board.bottom`；父实体通过 `path_id`（省略时为 Feature ID）绑定当前 scene 路径。普通交点未声明此关联时继续要求同表面。关联字段只用于固定交点，不能放在普通曲线或材料点上。

`branch="0"/"1"` 按有向冰面直线的两次围挡交点排序，排序在有限支持检查之前确定；缺少该分支不能自动换到另一交点。`domain.junctions.evaluate_ice_board_junction` 使用当前 scene 与同一 native 几何选项解析父实体。native `rs_ice_board_junction` 检查实际连续围挡、有限冰面线支持、分支及双方的 `z=0`，拒绝顶边、非零或未知深度、延长线交点、相切和重合。固定交点无逐帧 station，观测 `kind=0, local_index=-1, support=[0,0]`。

包含此关联的通用准备入口使用 `prepare ... --library 当前库`；Python 为 `prepare(..., core=NativeCore(...))`，当前拟合输入重载为 `load_prepared(..., core=...)`。准备与重载共用上述校验，缺 native 能力时明确失败，不以保存过的通过记录替代当前几何检查。原 UV、PTS、父实体身份与人工确认保留，不由数学交点新增观测。

### 广告内部几何的共同可调状态

当前安卡拉场景使用 `scene/print_layout_components.json` 的完整印刷图案分组，同一图案的文字行、标志、轮廓、内孔和间距共用一个正轴仿射变换。S01 仅 Ad2／3／4／6／7 的完整图案相对独立底板保留各 4 个坐标，共 20 个；Ad5 的解析矩形属于 MEIWA 图案本体，与其余印刷内容保持同一参考版式。整体尺寸和安装继续拟合。`print_shape` 仍有 10 个存储坐标，但六个非线性形变坐标为零且不活动，不能重开逐字、逐行或标志之间的独立形变。下述十维族是引擎的存储能力，不表示当前全部开放；参考图稿也未冒充实测印刷母版。

实拍确认的彩色底板可显式声明 `PrintTemplate.material_domain={kind:physical_unit_panel,boundary_path_ids:[四边],evidence:{...}}`。它要求已有四边和单位面板坐标规约，不由文字名称或包围盒推断。原生十二整数 `ScenePath` 的 print 类型以 `code=1` 表示此关系，`code=0` 不声明包含；其它值拒绝。GPU 用同一图案求值和导数检查有限路径的四边包含条件，与围挡高度和相机边界共同修正步长，真实 trial 再验证。这里没有像素容许误差、观测权重或新自由度；底板尺寸与安装、印刷组件整体变换仍参与联合拟合，无观测路径不加入当前求解约束。

模板、实例及版本库存见[场景索引](../events/ankara_2026/scene/README.md#当前资产与审核状态)；原材料UV数组是不可变参考几何，原片UV/PTS、观测相关组和人工确认不变。每个字形/明确图形的组内位置、独立宽高和非整体仿射的双轴单调轮廓形变进入 `print_shape`（10个存储坐标）；内孔与父轮廓、同字断开笔画使用同一组。单调映射保留组内拓扑和正体横竖直边，绘图加密点不是独立自由度，也不为每帧复制形状。

`print_square` 的3个坐标为展开面方框中心UV与log边长/面板高，仍由native生成真正方形。`print_rectangle` 的4个坐标为中心UV与log宽高，四边共角且保持直角；Ad5原四边种子不共角，转为已确认矩形的参数初值时存在小位置差，旧参考数组和原片读数保留，未作拟合。面板单位矩形固定为模板坐标规约，其物理宽高/安装仍可调；没有面板的模板选一个显式参考组，用整体尺寸/安装承担该组的平移和尺度，避免重复活动自由度。这是坐标规约，不是固定现场位置或尺寸的观测。

`domain.parameters.compile_scene_parameters(scene,scene_path, *, local_charts=None, optimize_scene=False)` 返回不可变 `ParameterState`：同一个float64向量、全局稳定block ID/offset、活动局部坐标、冻结/规约坐标与明确的null未定位块。默认保持场地几何冻结；显式 `optimize_scene=True` 激活共享冰场／围挡几何并保留长度规约及已声明的依赖坐标。尺寸、独立安装和内部形状共用此布局。`with_blocks`产生独立数值候选，拒绝修改冻结坐标或替未定位实例发明值。活动坐标表示可优化能力，不自动赋予观测信息、质量或统计模型。

`SceneParameterModel.dependencies(instance_id+'::'+template_path_id)` 编译真实共享依赖；`evaluate_print(...,state=...,jacobian=True)` 调用同一native并返回SZ/XYZ、按block排列的几何Jacobian、全局活动列和station导数。`jacobians_by_block[id]` 的末轴对应该block全部存储坐标；用于累计的 `active_jacobians_by_block[id]` 经统一展开矩阵映射到局部活动坐标（v1 为 `independent_local_indices` 选列，v2 可为混合基底），末轴与 `parameter_columns[id]` 的全局列号严格一一对应。冻结block的活动导数末轴与活动列号长度均为0，不能把完整存储导数直接与活动列zip。尺寸导数已经从物理宽高链到log尺寸。`sample_instance(...,parameter_state=state)`与完整审核共用它；审核比较重新建立共同采样时也继续消费同一状态，不退回静态模板。

显式局部图由 `compile_shape_local_chart` 编译，`load_shape_local_charts` 读取，经 `SceneParameterModel(..., local_charts=...)` 或 `prepare(..., local_charts=...)` 接入现有参数布局。`rinkspace.parameter_layout.v2` 为每个 block 保存 `ambient_size`、打包的 `tangent_offset/tangent_size`、chart revision 与 `d_ambient_d_local`。`FactorPlan` 复用该布局生成依赖和列映射；prepare 保存并在 reload 时核对同一计划。v1 的存储列编号保持兼容，v2 的活动列是紧凑局部列；两种列空间不能混用。

`ParameterState.local_values/retract_block/retract` 通过同一展开更新候选，`migrate_parameter_state(..., station_bindings=..., core=...)` 对旧状态做矩匹配，并用既有 native 单调材料映射在原边上重新编码辅助 station。源 UV、整数 PTS、相关组及确认字段只透传；无法保持物理点时明确报错。D1、D3 与完整 review 均接受同一保存重载的状态，候选仍为未审核。

当前显式选择见 `events/ankara_2026/scene/ad7/local_charts.json`：Ad7/Ad20 共享的 top T/E/T/L 与 bottom T 五组，局部维数分别 7/9/7/6/7。全部存储值仍为 1946，活动维数从 1847 降为 1833；仅选 L 时为 1843。其它组保持原布局。此选择基于严格轴对齐与完整转角约束，不把法向数值零方向当作冻结清单，也不表示已执行 GPU 联合拟合。

D1 的 `session.ad_review.render(..., parameter_state=state)` 和 `domain.print_assets.sample_template_paths(...)` 复用上述求值分派，显示当前内部形变及尺寸。材料图通过新增 `rs_sample_print_deformed_material_path` / `rs_sample_print_analytic_material_path` 求值，不传围挡，不输出XYZ，不替未定位实例发明安装。Python材料模式为 `board=None, jacobian=False`；旧DLL缺少能力时明确失败。D3按scene的 `advertisement_assets[].instance_ids` 精确选取实例，资产文件中其余配对实例不自动进入当前场景。

新增native ABI2能力为形变路径和typed方形/矩形求值，返回材料UV、展开SZ、XYZ及解析导数；旧DLL不支持时明确失败。这里实现的是几何导数，不能当作完整时变相机因子Jacobian、GPU累计或LM求解。生产求解生命周期仍按下文目标合同处理；几何接口成功不是拟合执行记录。

## 4. 持久输入与输出〔当前合同与求解目标〕

下述持久合同采用具体 schema 版本、稳定 ID 和相应 revision；字段以所属合同为准。JSON 以 UTF-8 写入，禁止 NaN/Infinity；数组是 C 连续、显式小端 `.npy`，禁用 pickle。文件引用相对所属清单，解析后不得逃出所属资产根。大数组、原片、日志、检查点和视频放 Data；已登记资产路径优先复用，通过 `storage.py` 解析，不搬原片。

### 当前输入与 prepared

`rinkspace.input.v1` 必填：`dataset_id/revision, scene_ref, rigs_ref, shots_ref, sources_ref, observations_ref, features_ref, noise_groups_ref, provenance_ref`。引用绑定当前静态文件/数组及其版本；适配器一次性产生它，不把旧 Python 源码路径注册为运行依赖。未知实体/相机族显式失败。

当前 `rinkspace.prepared.v1` 的 `manifest.json` 另保存 `parameter_layout_ref/initial_state_ref/parameter_dependencies_ref`；初值状态是独立可读的共同数值快照，不是假优化结果。它保存上述身份、`scope_ref/required_shot_ids/selected_shot_ids`、源/实体/机位/镜头/噪声组索引表、`arrays`、`model_configuration`、`dependency_revision`。`arrays` 每项含 `path,dtype,shape,sha256`；当前 prepare 核对所读数组。源记录含 SHA、宽高、time_base、整数 PTS 索引引用和外部资产 ID；镜头含源、rig、处理/物理区间和样条参考时间。未来正常 solve/resume 只检查当前输入的必要一致性，不重扫未变的历史资产或重放来源生产链。

观测数 N，行序固定；以下数组最小合同：

| 数组 | dtype / shape | 含义 |
|---|---|---|
| `uv` | float64 `[N,2]` | 原始源像素，永不被模型投影覆盖 |
| `pts` | int64 `[N]` | 源整数曝光 |
| `source_index,shot_index,feature_index` | int32 `[N]` 各一份 | 索引上述稳定 ID 表 |
| `kind,active,split` | uint8 `[N]` 各一份 | kind=0 固定点/1 曲线；active=0/1；split=0 拟合/1 调参选择/2 最终检查 |
| `noise_group_index` | int32 `[N]` | -1 为独立协方差，否则引用相关组 |
| `covariance` | float64 `[N,3]` | 独立行的 xx,xy,yy；相关行全零且仅消费其组 W |
| `support` | float64 `[N,2]` | 有限曲线真实参数域；周期围挡行全零仅占位、不得作为边界；固定点全零且不注册 station |
| `support_kind` | uint8 `[N]`（连续围挡必需） | 0=真实有限域；1=整圈未缠绕弧长，不受分段或闭合接缝限制 |
| `local_index` | int64 `[N]` | 固定点 -1；曲线对应局部状态索引，共享材料点跨帧共用 |

冰面测量语义由[赛事取点定义](../events/ankara_2026/scene/README.md#ice-observations)维护。当前 Feature 对旧外/内边缘语义及兼容别名执行拒绝检查，包括未激活条目；prepare 与重载共用分类型规则。实体不因测量方式重命名，广告轮廓不套用冰面中线。

广告曲线的 `measurement` 为 `contour`，保留模板原有字形外缘、内孔（含完整父轮廓 ID）、面板边与明确图形的角色；开放材料边界弧同样按原路径支持范围取轮廓。`domain.print_assets.resolve_print_curve(assets, instance_id, template_path_id)` 返回不可变 `PrintCurveBinding`。`Feature` 用 `surface="board"`、`kind="finite_curve"` 和该 `print_binding`；`label` 为绑定的模板路径 ID 或完整实例路径 ID，不能用广告绑定覆盖冰面标签。

JSON feature 的 `path_id` 为 `binding.path_id`（`instance_id::template_path_id`；若 `id` 就是此值，可省略 `path_id`），`print_binding` 为 `binding.as_record()`，保存 `instance_id/template_id/template_path_id/role/parent_contour_key`。prepare 与 prepared 载入均从 scene 实际登记、SHA 匹配的资产及所选实例重新解析该绑定；逐行元数据复用同一解析结果，不凭广告编号、任意 board 曲线或自称的 role 放行。不改源 UV、PTS、观测身份或相关组，不把轮廓改名为中线。共享尺寸仍引用资产参数块；未定位的配对实例可保留模型身份，但不能因此激活拟合观测。材料绑定不代替[来源准入](rink_model/BUILD_SPEC.md#observation-eligibility)、可见性或统计模型，也不表示已启动拟合。

跨表面固定交点字段与校验见[冰面线与围挡底边的固定交点](#冰面线与围挡底边的固定交点)。

围挡上下沿使用 native kind7，以未缠绕全局弧长 s 求值；接缝不是 bounds。旧分段到全局坐标只由 native 做一次坐标转换，原分段作为来源记录，不限制新的 station。

`observation_ids/provenance/measurement/quality/origin` 按同一行序保留；身份未知、拒绝、合成或语义不符不能因数组存在而激活。关联更正产生新 revision，不覆盖原记录。

prepare与拟合重载共用逐行核对器：JSONL显式声明的源UV、整数PTS、source/shot、时基、观测ID、相关组ID和active必须与对应数值行及索引表一致；原始provenance内部的历史记录不被改写。显式pending/rejected/synthetic或非实测证据不能仅靠翻转数组活动位进入拟合。此核对不证明取点准确，也不扩大已有人工确认范围。Feature的 `finite_curve` 对应数值kind=1，`material_point/fixed_junction` 对应kind=0。

相关组保存有序观测索引 `int64[m]`、白化矩阵 `float64[r,2m]`、有效秩 r、源协方差/信息表示引用和转换依据；W 的列序明确为 `u0,v0,u1,v1,...`。输入可以是已有协方差或信息矩阵，由准备阶段等价编译，不丢相关性。尚未支持等价转换的 nuisance 组显式报错，不把它当独立行。相关组/材料轨迹/同源衍生证据的闭包不得跨 split。有限支持和分支由 native 校验，不以 JSON 字符串代替。

prepare与拟合重载共同检查packed members为int64一维、weights为float64一维、组ID唯一、offset/count/rank为非负整数、范围和成员互指一致；不会把浮点成员行号静默取整。已有rank=0的工程约束组保持零统计权重，不在此重估噪声或重新审核来源。

### 当前状态导入与审核

完整状态导入已输出 `rinkspace.state.v1`，包含 `state_id,scene_revision,prepared_revision,layout_revision,parameter_blocks,values_ref,local_values_ref,shots,origin,user_approval`。参数/局部值为 float64 `[P]`/`[L]`；block 保存参数布局信息。当前冻结 d78 回放布局继续按只读历史语义保留。新的 `layout_family=shared_scene_parameters.v1` 状态实际读取 `parameter_layout_ref` 和有dtype/shape/SHA的 `values_ref`，完整审核不得忽略这些值；prepare 输出 `initial_state.json`。数值候选保存到新的JSON/NPY，不覆盖初值。已有辅助 station 可显式等价迁移并以带 SHA 的 `station_bindings_ref` 保存；源观测字段不变，没有辅助 station 时不凭空创建。这里没有 LM RunState。初值和候选为 `user_approval=not_reviewed`，不更改源观测既有人工确认。

已实现的 `review.json` 保存 `state_id,prepared_revision,source_ids,required_shot_ids,selected_shot_ids,scope_status,covered_pts,missing_pts,entity_inventory,missing_entities,artifacts,comparison,user_approval`。artifacts 含绝对路径与源 PTS/时间范围；回放使用已有标线、围挡、广告边界/编号。输出明确 `fit_executed=false,cuda_executed=false`；数值或覆盖 complete 不改用户验收。

### G1 只读 GPU 评估〔已实现的有界能力〕

`core.gpu.ResidentResidualProblem` 薄绑定调用 `rs_g1_create/evaluate/destroy`。create 复制当前 board/layout、完整物理相机、控制点、实体码、冻结 station、源时间与原 UV 到 native 自有内存及 GPU；evaluate("cpu"/"cuda") 返回每行 XYZ、UV、深度、预测行、RS 根误差、收缩界、原 UV 残差和状态；close 释放同一 handle。该评估方法不更新；只有显式附加 G3 才允许 native 候选接受／拒绝。

G1 实际实体范围是 shared curling contour（native code 100–131）；输入准备仍按当前 native 的 contour 语义核对实体身份。其他码保留行并返回 unsupported=11；非有限值、有限支撑及投影失败保留错误／NaN，不删行。设备端几何由同源 native 函数生成，不上传 CPU 投影冒充 GPU 运算。原观测的相关组、W、UV、PTS、人工记录和 station 来源保留，G1 不白化；原始平方和仅作诊断，不是统计目标。

`scripts/evaluate_g1.py` 接受显式 scene、camera、window、prepared、local-state、library、cpu-reference、out 路径，输出问题绑定、原始数值、逐帧记录、原片对照及时间／内存。冷热调用和 H2D／kernel／D2H 分开记录；CUDA 首次事件跨度可能含驱动延迟加载。自有 GPU 缓冲字节是准确值，cudaMemGetInfo 差值为设备全局变化，不能称作进程独占峰值。此 G1 前向不表示下方 solve/checkpoint 聚合生命周期已完成。

### G2 局部导数、白化与块累计〔已实现的有界能力〕

同一 `ResidentResidualProblem` 通过 `configure_linearization`／`rs_g2_attach` 附加活动方向、原 station 索引与数值尺度、完整相关组成员和原 W。活动方向是原相机控制点坐标上的列映射，不重造逐帧自由姿态。固定点映射为 -1，不注册 station；共享 station 的实体与初值必须一致。native 校验全部成员唯一覆盖、组秩／尺寸／有限值，并按相关组和共享 station 的连接关系建立局部连通分量。

`linearize("cpu"/"cuda")` 复用同一几何、完整物理相机、Brown 投影与隐式 RS。局部方向导数在已认证预测行根上使用隐函数关系，不采用观测行，不在生产内做全局有限差分。GPU 先算原残差及局部 J，再用完整 W 白化；按组的白化范数与有效秩计算 Huber 权重，并在连通分量内累计所有交叉项。任意行非法会使本次线性化报错，不删行后返回成功。

布局按 `directions` 的非零列与完整声明 RS 读出域的样条支撑并集打包。每条观测只保存实际依赖列，每个局部连通分量保存这些列的并集；固定点组不人为合并。保留系统按实际结构生成上三角 CSR 和可复用 scatter 地址。值偏移用 64 位，行／station／列身份及 CSR 索引用 int32。`layout["component_columns"]` 是每个分量在 `layout["columns"]` 中的起点／数量，分量 H/g 的前部采用这份列顺序，后部采用其局部 station 顺序。

默认返回四个 double 摘要（32 字节 D2H）及分阶段时间，不返回 J。`export_blocks`、`probe_rows`、`probe_group` 是显式诊断接口，分别导出 H/g、选定行导数、单组白化结果，独立累计下载字节。`export_sparse_blocks()` 返回打包分量 H/g 和保留上三角 CSR 值；`export_blocks()` 仅为显式小规模诊断把保留矩阵展开为 q×q，分量仍按各自列映射解释。CPU 同公式对照、旧 DLL 有限差分和 NumPy 原 W 校验仅供工程验证。

该入口尚未实现阻尼、生产消元、cuDSS 或参数更新。实际 G2 窗口采用 pan／tilt／log(f) 三个常量修正与已有 station；时间斜率因条件性差而冻结，场地、广告、安装、畸变和读出固定。原 197 个相关组与 73 个独立观测完整保留；局部秩亏、参考规约、损失配置和无先验的限制见节目 `g2.json`。原片扰动图只临时改变复制的控制值，不写回初值或宣称改善。

### 完整安装、共享场景与源剖面因子

`session.gpu_geometry.NativeSceneGPU` 只消费显式选定的当前 scene 和共同 ParameterState；观测参数没有替换场景的入口。`resident(..., scene_directions=...)` 附加该状态中实际的共享几何、尺寸、安装、轮廓形变与解析图形。`configure_camera_parameters` 的基础 27 维／含原有广角扩展 31 维物理局部图与运动、场景和 station 使用同一个保留系统；完整参考姿态、转轴、固定安装支点、杆臂、轴间距、入口瞳、未知光学与读出均可由方向矩阵激活。长度和参考坐标规约必须显式保留，不等于固定光心。

`NativeSceneGPU.resident` 的数值路径表仅包含 `path_ids` 实际引用的路径，联合场景与冻结场景采用相同范围。共同参数按这些路径的依赖连接；没有读点的其他显示路径不能以旧安装初值使当前候选失败。完整场景库存和参数关系仍保留，数值路径表不是删改全场模型，也不是未观测实例的验收。

固定双轴机构中的 `P=rig[15:18]` 是两轴最短连线在摇摄轴上的端点；`Q=P+a*h` 是该连线在俯仰轴上的端点，`h=unit(n×Rᵀb)`，`a=rig[21]` 为有符号的两轴最近距离。`Q` 随摇摄移动，但同一次纯俯仰时不动；`C=Q+Rᵀ(d+e_z·pupil(z))`。`h` 同时垂直于两轴，摇摄轴竖直时 P、Q 等高，故 `a` 不是云台底部到俯仰轴的高度。P 未绑定到某个安装螺孔；禁止再把一个独立轴向高度作为与 P 同时自由的变量，因为两者完全抵消。可视化外壳及底座相对 P 的未测绘位置，不得成为另一组透视拟合参数。给定 P 高度只计一次，不需要固定光心或固定杆臂来消除此冗余。

`configure_camera_envelope(..., frame_time_bounds=(begin, end))` 可显式选择相机时钟下的曝光中点区间，必须包含 resident 中的全部源曝光；不传时仍检查整个已分配样条。native 按每个候选实际的 `abs(readout)/2` 扩展至完整传感器行时间，认证所有与该区间相交的完整样条单元，首尾各最多保守扩展一个存储单元。该选项排除未使用的外推存储，不裁掉镜头或滚动快门范围，不改变安装／姿态／入口瞳等数值限制。C ABI 的 21 个 limits 值是在原 19 个值之后附加 `(time-first_control_time)/control_dt` 表示的两个曝光中点界限；Python 使用秒并负责转换，native 再检查源曝光覆盖。

预测像素不被图像矩形截断。`project_motion` 在画外继续求同一 `v=pi_y(X,t(v))`，并对延伸区间认证深度、光学域和唯一根；全局快门直接返回该时刻的有限投影。`configure_camera_parameters(max_readout=...)` 同时为每行保留曝光中点前后 `max_readout/2` 的数值时间支持，投影与稀疏列编译使用同一区间。固定 rig 的稀疏调用可在编译前使用 `configure_projection_time_support(half_seconds=...)`（C ABI `rs_gpu_set_projection_time_support`）显式声明更窄的计算区间；必须包含完整允许读出。没有声明时使用完整样条支持，列计划也包含整个支持，可能更密集。时间支持不是读点误差范围或新的物理测量；越界明确失败，不能把跨结点后漏掉的导数当作零。渲染裁剪与拟合残差分开，已删除把画面边缘加入 station／相机活动约束、强行拉回及停滞梯度判断的旧处理。

相机包络的候选修正使用当前已分解的联合 Schur 矩阵 S：对约束梯度 a 解 S w=a，再沿 w 恢复该边界，随后按真实几何重新检查其他边界、原始目标和预测下降。不会以逐参数裁剪或单独的对角阻尼度量替代该耦合方向；这不改变物理限制或观测权重。重复非线性包络修正仍须实际候选检查，不声称是一般约束问题的精确全局解。

共同场景的有观测印刷路径，在 native 内按其全部有限节点、形变分段节点或解析端点检查 `0 <= z <= board_height`，并以候选处解析导数和当前联合 Schur 矩阵修正违反边界的方向。相机与印刷边界在同一次候选恢复中交替重检，随后仍按原像素目标与实际施加步接受／拒绝；没有改变边界、删观测或增加误差罚项。检查节点只缓存不可变路径结构，当前共同几何每次重新计算。无观测路径不在本 resident 的数值路径表内。

安装域也参与上述联合修正：支点正高度、非零杆臂、超椭圆围挡外侧和已声明一侧使用同一当前几何及解析导数，支点与围挡参数共同调整。负／正 Y 侧的外侧半空间已蕴含围挡外部，不重复施加同义曲面条件；最终 `valid_installation` 检查保留。没有把数值提议越界直接当作必须拒绝整步的理由，也不放宽实际安装域。

GPU 分量正规矩阵内部仅存对称上三角；公共线性化诊断的 `H_value_count` 与显式导出块仍表示原来的密集逻辑布局，不能直接把该计数乘 8 当作实际设备分配。实际分配见 `device_bytes`；显式请求密集诊断块仍需要对应的主机内存。

`configure_profiles` 附加原始有理三次源查询曲线、alpha 区间及几何 station 区间。`exact` 在 native 内求投影曲线与源查询的交点，使用相同 UV/W 并对交点作隐式微分；alpha 不进入全局未知量，station 不得同时注册为自由量。`resolve_profile_roots` 总是检查严格交点。显式 `feasibility_restoration` 仅允许初值阶段暂用同名曲线 station 和原 UV/W 恢复定义域，不能将它的目标冒充最终严格查询似然。

候选可行性恢复保留同一原始目标：各 station 独立投影到真实有限支持；画面边缘用同一预测行方程恢复。共享 Lane／球门关系、已声明杆臂范围及有限查询边界通过联合候选修正处理，不用单个局部边界缩小所有参数的步长。查询外延只用于定位越过的原始 alpha 边界，不进入接受目标；最终候选必须通过原始有限区间、预测行唯一性及物理有效性检查，预测下降量使用实际施加的修正。

### G3 一次原生 LM 候选〔已实现的有界能力〕

局部有限端点和传感器边界按同一次联合步计算乘子，批量释放乘子符号错误的边界，再统一消元／求解。对重复工作集使用设备端逐项精确比较与常量数量的快照检测循环。共享参数的物理投影并不是完整约束二次问题的精确解，投影后重新释放局部端点可能再次触发同一端点。出现这种循环且局部块均为独立标量时，在已经投影的共享增量条件下，直接最小化每个局部阻尼二次项，并同时满足原有限支持及线性化传感器行区间，再由原生非线性域、原目标下降及增益比决定是否接受；共享参数与局部变量均未在后续拟合中冻结。成功候选返回 `reason_code=5`（`accepted_decrease_after_conditional_local_completion`），不声称求得完整约束二次问题的最优解。若局部块耦合、条件区间不可行或共享增量越界，仍返回 `reason_code=4`（`repeated_active_set_retry_with_damping`），保持已接受状态并增加阻尼重试。循环不计为分解失败或数值收敛，也不改变点的目标、边界或活动参数。`RINKSPACE_TRACE_LM_ACTIVESET` 可显式开启内部阶段诊断，默认不输出。

`configure_solver`／`rs_g3_attach` 给同一 resident handle 附加原 station 支撑交集、明确的 LM 配置和启动时可用容量。`lm_step`／`rs_g3_step` 只生成一个候选：必要时调用同一 G2 线性化，在显式缩放坐标中默认采用 D=I；可选 `jacobi` 使用完整联合 Hessian 对角及浮点数值下限，`jacobi_image_station` 补入 station 的原像素 Jacobian 范数，`jacobi_image` 使用 `D=max(1,diag(H),diag(J_image^T J_image))`。最后一种保留配置的坐标单位，抑制源 W 的切线零空间与近饱和轮廓坐标造成过大数值步长。它们只改变阻尼，不增加先验、不修改源 W、目标或参数活动性。先加入阻尼，再按完整连通分量求解局部块：标量块直接除法，较大块按实际维度分桶调用 cuSOLVER 批量 Cholesky；工作区不按全局最大局部维度填充，右端列数取本分量的实际相机依赖。保留变量的 CSR 上三角按 SPD 交给 cuDSS；禁用 hybrid execute/memory，不使用 CPU 数值求解后备。

初始 lambda 为显式 `initial_lambda_relative × max(diag(H))`（可显式指定 lambda）；默认配置为 1e-3、eta=1e-4、相对线性残差容限 1e-8，均保存进本次结果。边界下降分量的活动集由 native 处理；耦合后的步若在已有边界朝外，固定相应局部维度并重新消元／求解，保持同一 lambda 和线性化。内部边界求解不是另一个 LM 候选。可行步长后按实际增量计算未加阻尼的二次预测下降，候选重新计算原完整 W 和原 Huber 总目标，不能使用旧 IRLS 权重替代实际候选目标。

cuSOLVER、cuDSS 主机返回码与设备数值状态分别核对；检查 `Sδ-rhs`，拒绝非有限或不正定结果。源行失效使整个候选拒绝，不删行。只有实际下降且增益比超过 eta 才接受；Nielsen 规则更新 lambda/nu。拒绝保持 accepted state 和原线性化，之后若显式再调用则按新阻尼重做局部消元，复用符号分析。接受后使旧线性化失效。真实数据的一次候选及后来连续尝试的失败／停止均按原记录保留；连续原生循环与百万观测工程验证见 REBUILD，磁盘完整续算仍未交付。

正常返回摘要、时序及状态，完整 J 不回传；`export_state(trial=False/True)` 与 `export_step_system(sparse=True)` 是显式交付／工程诊断；后者返回保留 CSR，默认 `sparse=False` 为小规模检查显式展开 q×q。CPU 前向在显式请求时同步已接受 native 状态，正常候选无需下载整个状态。`scripts/evaluate_g3.py` 读取 G1 数值输入和 G2 已保存的 plan，不重跑历史权重生产；结果目录已有 decision 时拒绝重复执行，`--review-existing` 仅完成已保存候选的报告和原片对照，不是求解恢复。

本机 cuDSS 复用项目已有 0.8.0.10 / CUDA 13 包，现归 `environments/cudss/0.8-cu13`，逐文件身份见 `environments/cudss-local.json`。CUDA 13 系统运行库从 `CUDA_PATH/bin/x64` 读取，cuDSS DLL 从本项目 bin 读取，不依赖其它项目目录。数值库调用依据其随包头文件与示例；另参见 [cuDSS 状态与内存接口](https://docs.nvidia.com/cuda/cudss/latest/types.html)及 [cuSOLVER 批量分解](https://docs.nvidia.com/cuda/cusolver/index.html)。

### 连续原生 LM〔已实现〕

`solve(gradient_tolerance=1e-7, step_tolerance=1e-6, relative_cost_tolerance=1e-10)` 调用同一 native 连续控制器，没有隐含次数／时间上限。每个接受状态重新线性化，只有原目标的投影梯度通过容差才报告数值收敛；边界截断或大阻尼导致步长／下降小不单独触发成功。浮点停滞和调用者停止分别返回非收敛状态。`request_stop()` 或回调返回 `False` 在安全点停止，同一 handle 再次 `solve()` 保留状态和阻尼继续。结果的 `linearization_wall_ms` 与 `candidate_wall_ms` 分别累计线性化和候选阶段，互不重叠；进度事件 `wall_ms` 包含该次必要的线性化。

### 求解运行与检查点〔目标，未实现〕

`rinkspace.run_options.v1` 必填：`backend=cuda_cudss,active_block_ids,precision,loss,motion_prior,measured_priors,lm,memory_budget`。LM 算法见 SOLVER 第 7 节；实际容差、损失参数与预算由实现及选定配置定义，完整保存进 run，不将旧设计数值冒充当前默认值。精度未验证不能自动切换；CPU 仅数学回放/小问题对照，不是生产 solve 后备。

run 输出：`run.json`（输入引用、实际选项、scope）、`main.log`、`events.jsonl`、`result.json`、`checkpoints/<sequence>/`。checkpoint 必须含完整 accepted State、局部状态、LM 的 lambda/nu/迭代位置、ActiveSet、布局/输入版本及格式版本；临时写入成功后原子登记为 latest，半写目录不能续算。GPU 缓存可重建；源数据不重造。

`result.json` 必填：`run_id,status,reason_code,last_checkpoint,selected_shot_ids,scope_status,objective_terms,timings,memory,fit_executed,cuda_executed,user_approval`。没有实际迭代不能写 `fit_executed=true`。

## 5. 唯一 native 聚合生命周期〔待实现〕

Python 仅编排准备/路径/进程；C++ 的公开数值生命周期如下。`step` 是 `solve` 内部动作，不再作为另一套公开入口：

```cpp
CompiledProblem compile_problem(const PreparedProblemView&, const ActiveSet&,
                                const BackendOptions&);
RunResult solve(CompiledProblem&, State&, RunState&,
                const RunOptions&, RunControl&);
ProjectionBatch project(const CompiledModel&, const State&,
                        ArrayView<ProjectionRequest>, ProjectionWorkspace&);
Checkpoint snapshot(const CompiledProblem&, const State&, const RunState&);
void request_stop(RunControl&);
```

`CompiledProblem` 持有同一 `CompiledModel`、因子布局和 GPU 缓冲；渲染复用其模型视图，不复制投影公式。只读回放可编译模型部分而不建因子/求解器，但未知字段仍报错。PreparedProblemView 借用的只读内存须存活到问题销毁；State/RunState 的当前有效值由 native 拥有，Python 不在迭代间直接改数组。输出投影/快照是显式导出的副本。

`ProjectionRequest` 含 `source_id,shot_id,pts,feature_id,coordinate_kind,coordinate`；坐标含义由实体类型决定，固定交点不接受逐帧 station。输出逐项 UV、深度/可见性与状态；失败保留该请求和错误，不删除行。`project` 与残差共用 geometry/projection 数学及 RS 模式。

`RunControl` 提供线程安全停止标志和只读事件回调。停止在安全点处理，保留最后完整 accepted State 并 snapshot；trial 不覆盖它。resume 从同一快照恢复 RunState 后调用同一个 solve，无第二个求解算法。回调不能更改观测、目标或参数；UI 不回调 GPU 内层。

扩展 C ABI 沿用 `core_api.h/core_api.cpp` 与 `core/native.py` 薄绑定；C++ 异常在 ABI 边界转换为错误状态，不跨语言传播。聚合结构定义和 ABI 升级随实际实现落地。现有 ABI 2 的完整几何/运动前向不包含上述求解聚合生命周期。

## 6. 求解生命周期的状态、错误和反馈〔目标，未实现〕

创建 run 后、加载大数组/启动 GPU 前，立即打印 `main.log` 的绝对路径。事件行至少含 `schema,run_id,sequence,utc_time,kind,payload`；kind 为 `started/phase/iteration/checkpoint/stopped/error`。iteration 报 accepted/rejected、目标分项、耗时、内存和原因；不得仅刷无意义心跳。

结果 status 为 `numerical_stop`、`paused`、`budget_exhausted`、`failed`。reason_code 区分 `gradient_small/step_and_cost_small/user_stop/iteration_budget/invalid_input/unsupported_capability/domain_failure/unobservable/numerical_failure/resource_exhausted`。预算耗尽不是收敛；错误含具体实体/参数 ID、字段与输入位置；不能把无法识别的实体跳过后返回成功。

目标 CLI 退出码：0=请求正常完成（solve 的数值停止仍不代表视觉通过）；2=参数/合同错误；3=能力/依赖不可用；4=数值/资源失败；5=预算耗尽；130=收到停止并安全保存。当前已实现 CLI 尚未完整映射上述求解退出码。native 逐点状态与进程退出码不混用。目标求解失败写结构化 result/error；没有结果时 stderr 写明原因和已有日志路径，文件系统错误不打印成功回执。

F30 候选资格适配保留在 SourceRinkReader 的 material_reader 路径，默认关闭。只有所选读取配置显式设置布尔值 `enable_candidate_admission: true` 才启用；输出 `native_material.candidate_admission_enabled` 记录实际状态。默认读取仍保留已有命名导引精读、F15 同帧事务和 F16 遮罩修复。该开关不代表真实节目闭环已经验收。
