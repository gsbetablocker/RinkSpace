# 实现接口：现有入口与目标合同

本页分别说明已实现接口与尚未实现的聚合合同。native ABI 2 已提供共同几何、完整时变相机、prepare／review 和驻留连续 GPU LM；统一数据集 solve/resume CLI 与完整优化器磁盘恢复仍未实现。算法见 [SOLVER](SOLVER.md)，能力见 [REBUILD](REBUILD.md)。

## 1. 可调用入口

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

当前可调用驻留求解器；完整数据集命令与跨进程优化器恢复仍缺失。来源读取具备候选适配与原片回读接线，完整真实节目自动取点尚未验证。

Python 入口为 `rinkspace.core.native.NativeCore(library_path)`；方法签名见 [core/native.py](../src/rinkspace/core/native.py)，C ABI 唯一定义见 [core_api.h](../native/include/rinkspace/core_api.h)。ABI 2 使用 float64、行优先、显式长度和调用者分配输出。同一 ABI 号不保证旧库有所有新增符号，绑定需检查实际能力。

`camera[22]=R[9],C[3],fx,fy,cx,cy,skew,k1,k2,p1,p2,k3` 只代表静态曝光；`board[6]=half_length,half_width,rx,ry,exponent,height`。运动相机经 project_motion 使用完整状态及隐式预测行 RS，不使用静态 camera 数组替代。

RS 前向认证深度、光学域及预测行唯一根，必要时对子区间继续认证；未认证返回状态 9。CPU／GPU 使用相同方程与判据，有限画外预测不因图像边缘而失败，完整时间支持仍须有效。

单点包装失败抛 NativeError；批投影返回逐点状态，失败 UV 为 NaN，调用方逐项处理，不能钳回有效像素或删除失败请求。状态由头文件和 rs_status_message 维护。前向接口不更新参数；连续 LM 使用 `rs_g4_solve`，薄绑定与当前模型入口见 [native 调用说明](../native/README.md#当前模型的-gpu-求解入口)。

广告安装预览已实现 `model preview-print-installation --scene FILE --assets FILE --instance ID --camera FILE --frame FILE --pts INT --library FILE --out DIR`。消费当前自包含模板/参数与已迁入相机，输出同一实例的 `unfolded.png/three_d.png/source.png/source_full.png/geometry.json/result.json`，不重新导入历史来源、不拟合。`domain.print_assets.initialize_instance` 只显式初始化共用 log 尺寸块及所选实例独立安装块，拒绝覆盖不同现值；未定位实例求三维坐标会明确失败。`sample_instance` 对参考材料路径及形变分界加密，通过共同 `evaluate_print_path` 分派到native形变、方形、矩形或不形变路径，展开变换和BoardCourse共用同一C++实现。安装参数为 `[s_origin,z_origin,rotation,width,height]`，不按直/弧分段限制安装。字形、内孔、开放支持和原观测保持不变；采样不生成观测。当前ABI2使用新增能力前检查对应symbol；版本号相同不表示旧DLL具备所有新函数。Ad2 的 source-UV 方向通过安装旋转 π 表达，没有再翻转原顶点。

**安装预览来源核实：** `--frame` 仅用于任意图片的未核实预览，输出明确 `native_source_verified=false`，不宣称其对应相机源曝光。需要核实原曝光时使用下面的互斥入口；它核对 scope 中既有原视频 SHA、源尺寸/时基/镜头区间，并精确解码请求 PTS，保存 `decoded_source.png`。相邻曝光不能代替缺失的 PTS，来源核实也不代表视觉拟合通过。

```text
python -m rinkspace model preview-print-installation --scene FILE --assets FILE --instance ID --camera FILE --source-video FILE --scope FILE --pts INT --library FILE --out DIR
```

`--assets` 实际内容必须与 scene 当前登记的资产 SHA 相同；多个引用的实例 ID 不得重叠。孔洞父引用采用完整当前 path ID，不能在显示时静默退回自身。`domain.print_assets.sample_instance(..., stations_by_path={path_id: stations})` 支持指定共同物质采样位置以比较不同尺寸候选，仍在 native 求值，不插值像素。

`compat.joint_camera.import_joint_camera` 读取所需镜头的完整保存相机，按已声明的 control_coordinates 块和 offset 读取控制值，不重新拟合时间轨迹。旧导入器仅支持已证明不激活的 wide Brown 扩展，不支持时明确拒绝。当前 `domain.camera.complete_camera_rig` 可显式打包完整 rig43 扩展，GPU、隐式导数、联合更新和审核前向使用同一模型。

### 原片连续取点入口

`python -m rinkspace observe read --request FILE --out DIR` 调用已有 `SourceRinkReader`，使用配置中登记的现有 GPU 环境与正式权重；请求与实际验证见[节目记录](../events/ankara_2026/programmes/okada_mei_sp/source_reader_audit.json)。请求中的 sequences 使用原生整数 PTS；首帧可读取已有人工接受包，后续帧用双向光流有效性传递身份、导航点和法向，再从当前 RGB 读取材料。实际切镜重置记忆，导航不是新观测。

公共入口已接入当前 native 实体语义、既有线／圈联合读数和 `SourceFeedbackRead` 的实际使用回执。围挡正面读取蓝带上沿与黄带下沿；只见顶面时使用朝冰边缘，顶面、外侧设备和人物不能获得广告身份。现有官方 S06 顶面证据只在源 SHA、整数 PTS、时基及原生 RGB 哈希全部一致时保留原 UV 与相关身份；新环境的剖面复读另存诊断，不能覆盖它。其它画面缺证据时保留未知。

读取入口不调用求解或修改模型；输出保留逐帧读数、知识消费回执、模型版本与 native 引用。`enable_candidate_admission: true` 才启用候选准入，默认关闭，输出 `native_material.candidate_admission_enabled` 记录实际选择。旧 `observe audit` 已撤销。真实检查范围以 source_reader_audit.json 为准，局部检查不表示全 S01–S09 自动取点完成。

## 2. 完整节目范围与源时钟〔prepare/review 已接入〕

源与范围读取所属节目 scope，当前样本见[节目入口](../events/ankara_2026/programmes/okada_mei_sp/README.md#源与范围)。公共代码不硬编码镜头数、帧率或时基。物理边界与请求处理区间分别保存，处理终点不是切镜。

区间统一 `[first_pts,stop_pts_exclusive)`，曝光身份为 `(source_id,int64 pts)`，读取实际 PTS 索引。计算秒先减参考 PTS，再乘有理 time_base；JSON 保存整数分子/分母，不用 frame/fps 重造时钟。

已实现的 prepared/review 及未来 run 合同均保存 `required_shot_ids`、`selected_shot_ids`、每镜头处理区间。子集输出 `scope_status=subset`；全集也只有覆盖所有必需曝光才可标 `scope_status=complete`。这表示覆盖完整，**不表示拟合或用户审核通过**。缺镜头/缺曝光列出 ID 与范围，不能只保存 selected 数量后写“完整节目”。

prepared/run 的覆盖指源 PTS 登记及相机状态定义域覆盖，不要求每帧都有人工点或活动观测；review 的覆盖还要求相应输出实际包含这些曝光。不得借完整范围要求重新逐帧人工标注。

## 3. 统一 CLI 与编排接口

广告来源适配接口：`compat.ad_assets.import_mixed_ad_template(path,current_source)` 返回来源、模板、实例、逐轮廓观测候选、别名及原始provenance；`append_source_atlas(assets,path,source,frame_sha256=...)` 把已有原生字形关联同名模板，不生成缺失的material坐标。复制别名不再登记测量；所有新迁入候选不因导入获得拟合或人审资格。`model import-ad-assets --input FILE --out FILE` 与 `model preview-ad-assets --assets FILE --library FILE --out DIR` 已接唯一CLI。它们只导入/查看，不调用优化器；输出已存在时明确拒绝覆盖。

广告资产预览保留外轮廓、内孔与已声明解析图形；缺图、缺原像素、缺直接关联在输出中明示，不借模板投影补齐。不同曝光使用逐 observation 的 PTS 与相匹配的 source.exposures，不能用 registry 首曝光代替所有行。

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

`review.render` 是当前回放包入口，兼容 API 由包内 `legacy.py` 保留。`session.solve.solve(...) -> RunReceipt` 与 `session.resume.resume(...) -> RunReceipt` 仍为目标签名，相应模块尚未实现。

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

模板、实例和版本见[场景索引](../events/ankara_2026/scene/README.md#当前资产与审核状态)。材料 UV 是不可变参考几何，数值参数通过共同状态改变当前形状；原片 UV／PTS、来源和人工确认分别保留。引擎支持十维 print_shape 存储族，当前完整图案配置只开放已声明的整体仿射坐标，内孔与父轮廓共用所属图案变换，采样加密不增加自由度。

`print_square` 的三个坐标为展开面方框中心 UV 与 log 边长／面板高，由 native 生成真正方形；`print_rectangle` 的四个坐标为中心 UV 与 log 宽高，四边共角且保持直角。单位面板矩形是模板坐标规约，物理宽高与安装仍可调；无底板时选择显式参考组，避免其平移／尺度与整体安装／尺寸重复活动。坐标规约不是现场测量。

`domain.parameters.compile_scene_parameters(scene,scene_path, *, local_charts=None, optimize_scene=False)` 返回不可变 `ParameterState`：同一个float64向量、全局稳定block ID/offset、活动局部坐标、冻结/规约坐标与明确的null未定位块。默认保持场地几何冻结；显式 `optimize_scene=True` 激活共享冰场／围挡几何并保留长度规约及已声明的依赖坐标。尺寸、独立安装和内部形状共用此布局。`with_blocks`产生独立数值候选，拒绝修改冻结坐标或替未定位实例发明值。活动坐标表示可优化能力，不自动赋予观测信息、质量或统计模型。

`SceneParameterModel.dependencies(instance_id+'::'+template_path_id)` 编译真实共享依赖；`evaluate_print(...,state=...,jacobian=True)` 调用同一native并返回SZ/XYZ、按block排列的几何Jacobian、全局活动列和station导数。`jacobians_by_block[id]` 的末轴对应该block全部存储坐标；用于累计的 `active_jacobians_by_block[id]` 经统一展开矩阵映射到局部活动坐标（v1 为 `independent_local_indices` 选列，v2 可为混合基底），末轴与 `parameter_columns[id]` 的全局列号严格一一对应。冻结block的活动导数末轴与活动列号长度均为0，不能把完整存储导数直接与活动列zip。尺寸导数已经从物理宽高链到log尺寸。`sample_instance(...,parameter_state=state)`与完整审核共用它；审核比较重新建立共同采样时也继续消费同一状态，不退回静态模板。

显式局部图由 `compile_shape_local_chart` 编译，`load_shape_local_charts` 读取，经 `SceneParameterModel(..., local_charts=...)` 或 `prepare(..., local_charts=...)` 接入现有参数布局。`rinkspace.parameter_layout.v2` 为每个 block 保存 `ambient_size`、打包的 `tangent_offset/tangent_size`、chart revision 与 `d_ambient_d_local`。`FactorPlan` 复用该布局生成依赖和列映射；prepare 保存并在 reload 时核对同一计划。v1 的存储列编号保持兼容，v2 的活动列是紧凑局部列；两种列空间不能混用。

`ParameterState.local_values/retract_block/retract` 通过同一展开更新候选，`migrate_parameter_state(..., station_bindings=..., core=...)` 对旧状态做矩匹配，并用既有 native 单调材料映射在原边上重新编码辅助 station。源 UV、整数 PTS、相关组及确认字段只透传；无法保持物理点时明确报错。广告资产预览、安装预览与完整 review 均接受同一保存重载的状态，候选仍为未审核。

局部 chart 选择属于具体模型配置，以当前场景引用为准。历史单字／局部图的维数和冻结清单不作为当前完整印刷图案的默认设置；活动维数也不等于独立可辨识维数。

广告资产预览的 `session.ad_review.render(..., parameter_state=state)` 和 `domain.print_assets.sample_template_paths(...)` 复用上述求值分派，显示当前内部形变及尺寸。材料图通过新增 `rs_sample_print_deformed_material_path` / `rs_sample_print_analytic_material_path` 求值，不传围挡，不输出XYZ，不替未定位实例发明安装。Python材料模式为 `board=None, jacobian=False`；旧DLL缺少能力时明确失败。安装预览按 scene的 `advertisement_assets[].instance_ids` 精确选取实例，资产文件中其余配对实例不自动进入当前场景。

形变路径、typed 方形／矩形接口返回材料 UV、展开 SZ、XYZ 与几何解析导数；旧 DLL 缺符号时明确失败。完整相机因子、GPU 累计与 LM 使用下文驻留接口，单独调用几何函数不表示已执行拟合。

## 4. 持久输入与输出〔当前合同与求解目标〕

下述持久合同采用具体 schema 版本、稳定 ID 和相应 revision；字段以所属合同为准。JSON 以 UTF-8 写入，禁止 NaN/Infinity；数组是 C 连续、显式小端 `.npy`，禁用 pickle。文件引用相对所属清单，解析后不得逃出所属资产根。大数组、原片、日志、检查点和视频放 Data；已登记资产路径优先复用，通过 `storage.py` 解析，不搬原片。

### 当前输入与 prepared

`rinkspace.input.v1` 必填：`dataset_id/revision, scene_ref, rigs_ref, shots_ref, sources_ref, observations_ref, features_ref, noise_groups_ref, provenance_ref`。引用绑定当前静态文件/数组及其版本；适配器一次性产生它，不把旧 Python 源码路径注册为运行依赖。未知实体/相机族显式失败。

当前 `rinkspace.prepared.v1` 的 `manifest.json` 另保存 `parameter_layout_ref/initial_state_ref/parameter_dependencies_ref`；初值状态是独立可读的共同数值快照，不是假优化结果。它保存上述身份、`scope_ref/required_shot_ids/selected_shot_ids`、源/实体/机位/镜头/噪声组索引表、`arrays`、`model_configuration`、`dependency_revision`。`arrays` 每项含 `path,dtype,shape,sha256`；当前 prepare 核对所读数组。源记录含 SHA、宽高、time_base、整数 PTS 索引引用和外部资产 ID；镜头含源、rig、处理/物理区间和样条参考时间。未来正常 solve/resume 只检查当前输入的必要一致性，不重扫未变的历史资产或重放来源生产链。

以下是通用 prepared 的最小数组合同。节目原始七字段 NPZ 是直接驻留适配器的输入，不等同于该 prepared 格式；其精确 dtype、shape 与 SHA 由节目 manifest 登记。prepared 中的统计字段仅供显式相关模式使用，不自动进入原始像素目标。观测数 N，行序固定：

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

### 驻留评估与当前场景〔已实现〕

`core.gpu.ResidentResidualProblem` 持有 native 自有模型、rig、运动、观测和设备缓冲；`evaluate("cpu"|"cuda")` 只读返回逐行 XYZ、UV、深度、预测行、RS 根诊断、残差和状态。`close()` 释放 handle。CPU／GPU 使用同源实体与投影，不上传 CPU 投影冒充 GPU 运算；非法行保留错误状态和 NaN。

`session.gpu_geometry.NativeSceneGPU(scene_path, core)` 编译当前场景，支持冰面、连续围挡、固定点与已定位广告。数值路径表仅包含本次 `path_ids` 引用的路径，完整场景库存仍保留；无观测实例不因此获得验收，未定位实例不生成安装。

`resident(*, camera, controls, path_ids, stations, pts, observed, scene_directions=None, scene_lower=None, scene_upper=None, motion_times=None, rigs=None, camera_ids=None)` 返回同一驻留问题。整数 PTS 先相减再转换秒；`motion_times` 是显式平移后的秒坐标，不能缩放时间。多机位 `rigs` 与逐行 `camera_ids` 保持 rig 归属；`scene_directions` 按共同 ParameterState 的活动列连接场景，未传时场景冻结。

### 完整相机、运动与共享场景〔已实现〕

`configure_scene_parameters(values=..., records=..., data=..., directions=..., lower=..., upper=..., charts=...)` 接入共同几何、尺寸、安装与当前版式。`configure_camera_parameters(directions=..., max_readout=..., lower=..., upper=...)` 接入完整相机方向：基础 rig38 每机位 27 行，广角扩展 rig43 每机位 31 行；多机位按 rig 顺序堆叠为 `[rig_count * chart_size, retained_columns]`。物理范围按 rig 打包；长度与参考角规约显式保留。

基础 chart 包含完整参考姿态、两转轴、支点、杆臂、轴间距、随变焦入瞳与光学、读出时间。广角 Brown 扩展保留所选 log-f 结点及其下方平方 hinge 的四个系数。`domain.camera.complete_camera_rig` 打包当前完整定义；兼容导入器不支持的旧扩展必须明确失败，不能省略后冒称等价。

双轴机构中 `P=rig[15:18]` 是两轴最短连线在摇摄轴上的端点，`Q=P+a*h` 是俯仰轴端点，`h=unit(n×Rᵀb)`、`a=rig[21]` 为有符号最近距离；`C=Q+Rᵀ(d+e_z·pupil(z))`。a 不是云台底座高度，P 未绑定某个螺孔；与 P 完全抵消的独立轴向高度不能同时开放。

`ShotMotionTimeline` 为各镜头分配独立运动块，只平移存储时钟，保持源 PTS、秒尺度和读出时长。共享 rig 不要求切镜间运动连续。首段参考角规约与后续自由参考角按实际方向矩阵保存。

`continue_exposure_halos` 可将每段首尾三次曲线延续到读出 halo；初始 controls 与 `directions(continue_endpoint_cubics=True)` 必须一致使用。它保持首至末帧中心区间的轨迹与前两阶导数，保留内部运动空间，不跨切镜延续，也不在输出端平滑。默认表示与实际运行选择分别保存。

`configure_camera_envelope` 可选择 `frame_time_bounds=(begin,end)` 或互斥的 `frame_time_intervals=[...]`。区间必须覆盖 resident 中的全部源曝光，native 按候选实际读出扩展完整传感器行时间。速度／加速度／jerk 的 pan、tilt、log(f) 三元边界通过各三次区间解析极值检查；`None`／正无穷表示未设限。实际物理范围由运行配置给出。

`project_motion` 对有限画外预测继续求同一隐式滚动快门方程，并认证深度、光学域、时间支持与唯一根。图像矩形不约束 station 或相机，渲染才裁切。`max_readout` 为稀疏列编译保留曝光中点前后相应支持；固定 rig 可在编译前用 `configure_projection_time_support(half_seconds=...)` 明确计算区间，必须包含完整允许读出。越界明确失败，不遗漏跨结点导数。

### 两种线性化目标〔已实现〕

| Python／C ABI | 配置与目标 |
|---|---|
| `configure_pixel_linearization(directions=..., local_ids=..., local_scales=...)`／`rs_g2_attach_pixels` | 直接累计 `0.5 * sum(du² + dv²)`；不接收 W、协方差、sigma 或 Huber，无隐含降权 |
| `configure_linearization(directions=..., local_ids=..., local_scales=..., group_offsets=..., members=..., ranks=..., weight_offsets=..., weights=..., huber_delta=...)`／`rs_g2_attach` | 显式相关组 W 白化及按有效秩的组级 Huber；保留全部交叉项 |

运动 `directions` 为 float64 `[control_count,3,q]`；`local_ids` 为 int32 `[N]`，固定点为 -1，曲线点引用局部 station；`local_scales` 为 float64 `[L]`。缩放改变求解坐标，不改变像素误差。共享材料点须由相同局部身份表达，实体与初值一致；不能拆成每帧独立 station。

`linearize("cpu"|"cuda")` 在认证的预测行根上对同一完整几何与相机作隐式微分。相关模式再应用 W 和组级 Huber；逐点模式直接累计两个像素残差。任意行非法则本次线性化失败，不删行后返回成功。G2 线性化本身不更新状态，附加求解器后才执行候选。

布局按真实方向非零列、声明读出域的样条支持与局部连通关系编译。局部块保留全部相关连接，保留系统为上三角 CSR；值偏移用 64 位，行／station／列和 CSR 索引用 int32。默认仅返回四个 double 摘要及时间，不回传完整 J。

`export_blocks()` 为小规模诊断展开保留稠密矩阵；`export_sparse_blocks()` 返回打包分量 H/g 与保留 CSR；`probe_rows/probe_group` 显式导出所选导数或白化诊断。分量按 `component_columns` 的列映射解释；逻辑密集计数不能当作实际设备分配，设备分配见 `device_bytes`。

`configure_profiles(row_ids=..., numerator_u=..., numerator_v=..., denominator=..., alpha_bounds=..., station_bounds=..., initial_alpha=..., mode="exact")` 附加原始有理三次源查询。native 求有限查询与投影曲线交点并隐式求导，alpha 不进入全局未知量，同一 station 不得另注册为自由量。`resolve_profile_roots` 检查严格交点。显式 `feasibility_restoration` 仅用于恢复有效域，其临时目标不能冒充最终严格查询目标。

### 联合约束与 LM 候选〔已实现〕

`configure_solver(lower=..., upper=..., device_budget_bytes=..., host_budget_bytes=..., ...)`／`rs_g3_attach` 传入真实 station 支撑交集、LM 配置和可用容量。`lm_step()`／`rs_g3_step` 执行一个候选，连续入口见下文。

阻尼在局部消元之前加入。默认 D=I，显式可选 `jacobi`、`jacobi_image_station`、`jacobi_image`；它们改变阻尼度量，不改变源 W、目标、物理先验或参数活动性。独立标量块直接求解，较大连通块按实际维数使用 cuSOLVER 批量 Cholesky；保留上三角 CSR 主系统交给同 stream 的 cuDSS。小型约束对偶工作面可通过 `RINKSPACE_SMALL_DUAL_BACKEND=cpu|gpu` 显式选择辅助后端，实际选择由运行记录保存；它不替换 cuDSS 主系统。

初始 lambda 为显式 `initial_lambda_relative * max(diag(H))`，也可指定 lambda。当前默认相对值 1e-3、接受阈值 eta=1e-4、相对线性残差容限 1e-8；实际配置保存到运行结果。拒绝复用原线性化但按新阻尼重新消元，接受后旧线性化失效。

有限 station 端点、rig／scene 范围、安装、印刷包含关系与连续相机包络参与同一候选修正。共享约束沿当前联合 Schur 度量恢复可行性，再核对原非线性域；不能单独裁剪参数后沿用未修正步的预测下降。图像边缘不属于该约束集合。

有观测印刷路径按实际有限节点、形变分段或解析端点检查围挡高度及已声明底板包含关系。安装检查包括支点正高度、非零杆臂、围挡外侧及已声明一侧；重复的同义外侧条件不重复计入。共享几何与安装共同求导，未观测路径不凭旧初值约束本次问题。

独立原始像素／标量 station 使用原界内的条件局部最小值与联合二次 merit 搜索，接受分支 `reason_code=6`。相关／多局部量分支可返回条件局部补全接受 `reason_code=5` 或重复活动集增阻尼重试 `reason_code=4`。这些状态不证明一般约束非线性问题的精确最优。

候选使用实际施加增量计算未加阻尼的预测下降，并重算所选完整目标：原始像素平方和，或显式相关 W／Huber。源行非法、非有限数、不正定或线性残差失败均明确拒绝。只有真实下降和增益比通过才接受，再按 Nielsen 规则更新 lambda／nu。

独立原始像素模式可显式选择 `profile_trial_stations=True`，在候选判定前对每个曲线点作原支持内全域搜索和下降修正；默认关闭，不用于 W／Huber 因子或共享 station。ABI options 第 8 项为 0/1。原 UV、PTS、实体与权重不变，搜索采样不生成观测。

`export_state(trial=False|True)` 显式导出完整 rig、控制点、station 与共享场景；`export_step_system(sparse=True)` 导出 CSR，默认稠密模式仅供小规模检查。库和运行依赖见[环境说明](../environments/README.md)。窗口脚本及历史结果的范围见[节目记录](../events/ankara_2026/programmes/okada_mei_sp/README.md#原生工程对照)。

### 连续原生 LM〔已实现〕

`solve(gradient_tolerance=1e-7, step_tolerance=1e-6, relative_cost_tolerance=1e-10, ...)` 调用 native 连续控制器，无隐含次数／时间上限。每个接受状态重线性化，只有原目标投影梯度通过容差才报告数值收敛；小步、边界截断或大阻尼不单独触发成功。上一接受步的预测与实际下降同时低于相对容差时返回 `cost_progress_stagnation_without_convergence`，属于非收敛停止。

`request_stop()` 或进度回调返回 `False` 在安全边界停止；同一 handle 再调用 `solve()` 保留接受状态、lambda 和 nu。回调异常先安全停止再抛出。当前 `export_state` 不包含完整优化器内部状态，不是磁盘严格恢复。

`projected_gradient_inf` 使用原目标梯度与当前活动约束 Jacobian 作联合切锥投影，覆盖物理 rig／scene、安装、印刷、连续包络和局部支持。投影与 KKT 数值残差都须通过，数值失败不能报告收敛。`rs_g4_stationarity_version()=2` 时 `constraint_projected_gradient_inf` 明示该语义；版本 1 仅覆盖可证明单列对应的盒范围，旧库按实际能力标识。

`stationarity_diagnostics(include_constraints=True)` 导出原梯度、投影方向、约束身份／归一化 Jacobian／乘子、可行性与互补性。`full_nonlinear_kkt_checked` 表示已执行所声明物理域的一阶检查，数值结论另看 `numerically_converged`，不能推成全局最优、相机物理或图像验收。

`refine_curve_stations(mode="global"|"local")` 在保持实体及原有限支持的条件下细化独立曲线对应，默认全域搜索；每点仅在自身像素距离下降时更新，固定点与共享材料关系不改。`evaluate_errors()` 返回每行原像素误差，失效为无穷；`evaluate_rows(rows)` 下载所选完整前向。

`linearization_wall_ms` 与 `candidate_wall_ms` 分别累计两个阶段；`callback_wall_ms` 已包含在 native `wall_ms`。显式 evaluate、state、probe 与回调下载计入诊断传输，`instrumented_d2h_bytes_during_solve` 不含 CUDA／cuDSS 内部传输。时序与内存口径不能当作准确性结论。

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
