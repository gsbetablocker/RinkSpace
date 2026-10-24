# 实现接口：现有入口与目标合同

版本：2026-10-24。native ABI 2 的完整几何/时变相机前向（含隐式 RS）、prepare/review 及下述 D1/D3 入口已实现；GPU LM、solve/resume 仍是目标合同，尚无对应命令或可调用生命周期。CPU 回放不表示 GPU 拟合。算法与参数设计见 [SOLVER.md](SOLVER.md)，当前实现状态见 [REBUILD.md](REBUILD.md)。

## 1. 现在确实能调用什么

在项目根、已有 Python 环境中执行；源码未安装时仅为当前进程设置 `PYTHONPATH=<项目>/src`。

```text
python scripts/build_core.py --out DIR [--cmake EXISTING_CMAKE]
python -m rinkspace model import-d78 --model FILE --relations FILE --out FILE
python -m rinkspace model preview --scene FILE --library DLL_OR_SO --out DIR
python -m rinkspace model import-state --model FILE [--complete-state FILE] --scope FILE --library FILE --out DIR
python -m rinkspace model preview-curling --scene FILE --library FILE --out DIR
python -m rinkspace model preview-field --scene FILE --library FILE --out DIR
```

构建目录在源码树外。构建输出 `build.log/build.json`，后者的 `library` 是实际库路径；当前 `cuda_built=false`、`nonlinear_solver_implemented=false`。`import-d78` 输出 `rinkspace.scene.seed.v1`、状态为 `unfitted_structure_candidate`，不是保持原投影的完整 d78 状态；`import-state` 才导入冻结镜头的完整状态与实体供回放，同时保留九镜头登记，不补造其他镜头的相机。各模型预览输出图、几何和回执，不拟合。已有 `doctor/probe/index` 等命令维持原语义。

Python 入口为 `rinkspace.core.native.NativeCore(library_path)`，实际方法如下；数学只在 native 实现：

```python
centres(parameters, rows)                  # XYZ 列表
ring(parameters, rows, house, ring, theta) # XYZ
lanes(parameters, rows, left_support)     # 每行三段、每段两个 XYZ
centre_circle(radius, theta)              # XYZ
line_line(a, b)                           # XYZ
line_circle(line, circle, branch)         # XYZ
project(camera, xyz)                      # (UV 列表, 逐点状态列表)
project_jacobian(camera, xyz)             # (UV, 行优先 2x3 世界点导数)
pivot_pose(rotation, pivot, lever, axis_world, axis_spacing) # R[9], C[3]
board_perimeter(board)                    # 周长
board_point(board, station, z)            # XYZ + 单位切向
board_points(board, stations, heights)    # XYZ 列表
ad_polyline(board, s0, width, z0, height, samples=24) # 四条展开边
sample_path(board, kind, parameters, vertices, stations) # 同一实体前向 XYZ
sample_print_chart(parameters, vertices, stations)       # D3 同一 kind5 的未绕回 (s,z)
print_square_vertices(parameters, center_uv, side_over_panel_height) # 展开面实际正方形
project_motion(rig, controls, first_control_time, control_dt, image_height, frame_times, xyz)
```

C ABI 的完整、唯一签名在 `native/include/rinkspace/core_api.h`，绑定在 `src/rinkspace/core/native.py`。当前 ABI 2 使用 float64、行优先、显式数组长度和调用者分配输出；导出几何、实体/印刷路径及静态/运动投影。批投影必须逐点检查状态，具体状态以头文件及 `rs_status_message` 为准。单点包装失败抛 `NativeError`；`project` 保留逐点状态，失败点为 NaN，不能钳回有效像素。

`camera[22]=R[9],C[3],fx,fy,cx,cy,skew,k1,k2,p1,p2,k3`，只代表静态曝光。`board[6]=half_length,half_width,rx,ry,exponent,height`。其他参数顺序以同一头文件为准。当前扩展库为 ABI 2；已有 `project_motion` 完整时变投影（含隐式预测行 RS），没有 LM 或 `rs_solve`。

D3 已实现 `model preview-print-installation --scene FILE --assets FILE --instance ID --camera FILE --frame FILE --pts INT --library FILE --out DIR`。消费当前自包含模板/参数与已迁入相机，输出同一实例的 `unfolded.png/three_d.png/source.png/source_full.png/geometry.json/result.json`，不重新导入历史来源、不拟合。`domain.print_assets.initialize_instance` 只显式初始化共用 log 尺寸块及所选实例独立安装块，拒绝覆盖不同现值；未定位实例求三维坐标会明确失败。`sample_instance` 对原 material_uv 沿边加密，调用同一 native kind5 和 `rs_sample_print_chart`，其展开变换共用一份 C++ 实现。kind5 参数为 `[s_origin,z_origin,rotation,width,height]`，不按直/弧分段限制安装。字形、内孔、开放支持和原观测保持不变；采样不生成观测。缺少新 chart 能力的旧 DLL 明确报错。Ad2 的 source-UV 方向通过安装旋转 π 表达，没有再翻转原顶点。

## 2. 完整节目范围与源时钟〔prepare/review 已接入〕

范围文件为 `events/ankara_2026/programmes/okada_mei_sp/rebuild_scope.json`。完整节目必须覆盖 **s01–s09 九镜头**，显示为 S01–S09；旧六镜头只是子集。公共代码读取清单，不硬编码 9。

清单中的物理镜头边界、用户要求处理的区间分别保存；处理终点不是新的切镜。本源已登记 `time_base=1/12800`，相邻 50fps 曝光间隔 256 PTS；代码必须读取实际索引，不能用 `frame/fps` 重造时钟。区间统一 `[first_pts,stop_pts_exclusive)`，源曝光身份为 `(source_id, int64 pts)`。计算秒先减当前镜头参考 PTS，再乘有理 time_base；JSON 保存整数分子/分母。

已实现的 prepared/review 及未来 run 合同均保存 `required_shot_ids`、`selected_shot_ids`、每镜头处理区间。子集输出 `scope_status=subset`；全集也只有覆盖所有必需曝光才可标 `scope_status=complete`。这表示覆盖完整，**不表示拟合或用户审核通过**。缺镜头/缺曝光列出 ID 与范围，不能只保存 selected 数量后写“完整节目”。

prepared/run 的覆盖指源 PTS 登记及相机状态定义域覆盖，不要求每帧都有人工点或活动观测；review 的覆盖还要求相应输出实际包含这些曝光。不得借完整范围要求重新逐帧人工标注。

## 3. 统一 CLI 与编排接口

D1 已实现的来源适配接口：`compat.ad_assets.import_mixed_ad_template(path,current_source)` 返回来源、模板、实例、逐轮廓观测候选、别名及原始provenance；`append_source_atlas(assets,path,source,frame_sha256=...)` 把已有原生字形关联同名模板，不生成缺失的material坐标。复制别名不再登记测量；所有新迁入候选不因导入获得拟合或人审资格。`model import-ad-assets --input FILE --out FILE` 与 `model preview-ad-assets --assets FILE --library FILE --out DIR` 已接唯一CLI。它们只导入/查看，不调用优化器；输出已存在时明确拒绝覆盖。

使用现有 `src/rinkspace/cli.py`。`FILE/DIR` 为路径参数，`--shots` 接受 `all` 或逗号分隔的清单 ID；未知/重复 ID 报错。当前已实现：

```text
python -m rinkspace prepare --input FILE --scope FILE --shots all|s01,s04 --out DIR
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
session.prepare.prepare(input_path, scope_path, shot_ids, out_dir) -> PreparedReceipt
review.render.render(prepared_dir, state_path, assets_path, shot_ids,
                     library_path, out_dir, compare_path=None) -> ReviewReceipt
```

`review/` 已替换旧 `review.py`，旧 API 通过包内 `legacy.py` 保留。`session.solve.solve(...) -> RunReceipt` 与 `session.resume.resume(...) -> RunReceipt` 仍是目标签名，相应模块尚未实现。

`PreparedReceipt` 返回 `manifest_path,dataset_id,revision,selected_shot_ids,scope_status`；`ReviewReceipt` 返回 review 对象及 `review_path`，路径为绝对路径。目标 `RunReceipt` 返回下面定义的 result 对象及 `result_path/log_path`。当前 CLI 捕获失败并返回非零；后文的结构化求解错误和完整退出码映射仍是目标，不能视为已实现。

`model rebuild-glyphs --input FILE --library FILE --out DIR` 已实现，其 SciPy 依赖属于 `advanced` extra；`domain.entity_graph.render_graph` 的 Matplotlib 同属 `advanced`。一般 native 几何/回放不需要这两个辅助依赖。

## 4. 持久输入与输出〔当前合同与求解目标〕

所有对象有 `schema`（具体版本）、稳定 ID 和 revision。JSON 以 UTF-8 写入，禁止 NaN/Infinity；数组是 C 连续、显式小端 `.npy`，禁用 pickle。文件引用相对所属清单，解析后不得逃出所属资产根。大数组、原片、日志、检查点和视频放 Data；已登记资产路径优先复用，通过 `storage.py` 解析，不搬原片。

### 当前输入与 prepared

`rinkspace.input.v1` 必填：`dataset_id/revision, scene_ref, rigs_ref, shots_ref, sources_ref, observations_ref, features_ref, noise_groups_ref, provenance_ref`。引用绑定当前静态文件/数组及其版本；适配器一次性产生它，不把旧 Python 源码路径注册为运行依赖。未知实体/相机族显式失败。

当前 `rinkspace.prepared.v1` 的 `manifest.json` 保存上述身份、`scope_ref/required_shot_ids/selected_shot_ids`、源/实体/机位/镜头/噪声组索引表、`arrays`、`model_configuration`、`dependency_revision`。`arrays` 每项含 `path,dtype,shape,sha256`；当前 prepare 核对所读数组。源记录含 SHA、宽高、time_base、整数 PTS 索引引用和外部资产 ID；镜头含源、rig、处理/物理区间和样条参考时间。未来正常 solve/resume 只检查当前输入的必要一致性，不重扫未变的历史资产或重放来源生产链。

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

当前测量定义只有：八组靶圈四层圆及半椭圆蓝区取轮廓，其他直曲线取材料宽度中线（模型零线宽）。靶圈实体名字保持 center/inner/middle/outer；取点方式不能替代实体名字。当前 feature 不接受任何旧外/内边缘定义或兼容别名，未激活条目也拒绝；prepare 与拟合读取入口执行同一规则。

围挡上下沿使用 native kind7，以未缠绕全局弧长 s 求值；接缝不是 bounds。旧分段到全局坐标只由 native 做一次坐标转换，原分段作为来源记录，不限制新的 station。

`observation_ids/provenance/measurement/quality/origin` 按同一行序保留；身份未知、拒绝、合成或语义不符不能因数组存在而激活。关联更正产生新 revision，不覆盖原记录。

相关组保存有序观测索引 `int64[m]`、白化矩阵 `float64[r,2m]`、有效秩 r、源协方差/信息表示引用和转换依据；W 的列序明确为 `u0,v0,u1,v1,...`。输入可以是已有协方差或信息矩阵，由准备阶段等价编译，不丢相关性。尚未支持等价转换的 nuisance 组显式报错，不把它当独立行。相关组/材料轨迹/同源衍生证据的闭包不得跨 split。有限支持和分支由 native 校验，不以 JSON 字符串代替。

### 当前状态导入与审核

完整状态导入已输出 `rinkspace.state.v1`，包含 `state_id,scene_revision,prepared_revision,layout_revision,parameter_blocks,values_ref,local_values_ref,shots,origin,user_approval`。参数/局部值为 float64 `[P]`/`[L]`；block 保存参数布局信息。当前冻结 d78 回放布局不等于已经实现新优化器的活动布局。初值和候选为 `user_approval=not_reviewed`。

已实现的 `review.json` 保存 `state_id,prepared_revision,source_ids,required_shot_ids,selected_shot_ids,scope_status,covered_pts,missing_pts,entity_inventory,missing_entities,artifacts,comparison,user_approval`。artifacts 含绝对路径与源 PTS/时间范围；回放使用已有标线、围挡、广告边界/编号。输出明确 `fit_executed=false,cuda_executed=false`；数值或覆盖 complete 不改用户验收。

### 求解运行与检查点〔目标，未实现〕

`rinkspace.run_options.v1` 必填：`backend=cuda_cudss,active_block_ids,precision,loss,motion_prior,measured_priors,lm,memory_budget`。具体 LM 默认值采用 SOLVER 第 7 节；所有默认值展开保存进 run，不能藏在临时脚本。精度未验证不能自动切换；CPU 仅数学回放/小问题对照，不是生产 solve 后备。

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
