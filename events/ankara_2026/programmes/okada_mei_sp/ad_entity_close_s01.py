"""Close source evidence, make review samples, and seal the checked handoff."""
from pathlib import Path
import sys,json,gzip,collections,shutil,zipfile,time
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_entity_registration_s01 as r
b=r.b;WORK=r.OUT;OUT=b.OUT/'increments/002'
NAMES=['verified_ads.npz','measurement_summary.json','pointwise_checks.jsonl.gz','coverage_intervals.jsonl.gz']

def publish():
    history=WORK/('final_candidates_before_stroke_resolution' if (WORK/'final_source_candidates').exists() else 'final_source_candidates');history.mkdir(exist_ok=True)
    if any((history/n).exists() for n in NAMES):raise ValueError('Candidate history already exists')
    raw=dict(np.load(WORK/'verified_ads.npz'));z=dict(np.load(WORK/'observability_ads.npz'))
    index={int(v):i for i,v in enumerate(raw['source_rows'])};rows=np.array([index[int(v)] for v in z['source_rows']])
    for k in raw:
        if not np.array_equal(raw[k][rows],z[k]):raise ValueError('Visibility audit changed a measured value: '+k)
    if len(set(z['observation_ids']))!=len(z['pts']) or len(set(z['source_rows']))!=len(z['pts']):raise ValueError('Duplicate source identity')
    counts=collections.Counter();reasons=collections.defaultdict(collections.Counter)
    with gzip.open(WORK/'observability_checked_points.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            c=json.loads(line)
            if 'path_id' not in c:continue
            key=(c['pts'],c['path_id'])
            if c['decision']!='not_admitted':counts[key]+=1
            else:reasons[key][c['reason']]+=1
    if sum(counts.values())!=len(z['pts']):raise ValueError('Point evidence/dataset mismatch')
    for name in NAMES:
        src=(WORK/name).resolve();dst=(history/name).resolve()
        if not src.is_relative_to(WORK.resolve()) or not dst.is_relative_to(WORK.resolve()):raise ValueError('History path outside workspace output')
        src.rename(dst)
    (WORK/'observability_ads.npz').rename(WORK/'verified_ads.npz')
    (WORK/'observability_checked_points.jsonl.gz').rename(WORK/'pointwise_checks.jsonl.gz')
    summary=b.read(WORK/'observability_summary.json')
    summary.update(candidate_rows_before_local_visibility=len(raw['pts']),source_scope=dict(first_pts=283136,last_pts=575488,time_base='1/12800'),old_rows_not_overwritten=True)
    b.write(WORK/'measurement_summary.json',summary)
    with gzip.open(history/'coverage_intervals.jsonl.gz','rt',encoding='utf8') as f,gzip.open(WORK/'coverage_intervals.jsonl.gz','wt',encoding='utf8') as out:
        for line in f:
            c=json.loads(line);key=(c['pts'],c['path_id']);c['candidate_included_points']=c['included_points'];c['included_points']=counts[key];c['local_visibility_exclusions']=dict(reasons[key]);out.write(json.dumps(c,ensure_ascii=False)+'\n')
    mapping=b.read(WORK/'verified_mapping.json');mapping.update(identity_charts_sha256=b.sha(WORK/'source_identity_charts.json'),actual_local_observability_checked=True,point_input_sha256=b.sha(WORK/'verified_ads.npz'));b.write(WORK/'verified_mapping.json',mapping)
    registry=b.read(WORK/'native_entity_registry.json');missing=[]
    for p in registry['paths']:
        name=p['native_id']
        if not summary['by_path'].get(name):
            why=collections.Counter()
            for (pt,n),rs in reasons.items():
                if n==name:why.update(rs)
            missing.append(dict(native_id=name,accepted_rows=0,candidate_rejection_reasons=dict(why)))
    b.write(WORK/'coverage_gaps.json',dict(unobserved_retained_native_entities=missing,known_visible_artwork_without_current_native_entity=[dict(ad='Ad3',artwork='大吉 two right-hand characters',evidence='Current Ad3 retains logo and 買取 source paths only; no named current native contour for 大吉',source_example_pts=320000),dict(ad='Ad9',artwork='Fine ornamental and script details outside the retained red field/CAN curves',evidence='Current native Ad9 registry retains only four named contours')],no_new_native_geometry_invented=True,before_22_120_excluded_by_user=True,complete_visible_artwork_coverage_claimed=False))
    b.write(WORK/'observability_rules.json',dict(unit='individual current visible material contour, with fine glyph structure checked separately from panel identity',native_geometry_used_for_identity=False,source_reference_motion_blur_is_navigation_only=True,fine_shape_admission=dict(unblurred_correlation_min=.66,horizontal_structure_correlation_min=.14,vertical_structure_correlation_min=.25,current_luminance_std_min=3.,navigation_translation_checked_px=[-2,0,2]),thresholds='Engineering screening criteria checked against current source examples; not calibrated confidence or proof of human acceptance',colored_outer_panel_check='Actual source profile crosses colored paint and background; matching a gradient inside the same paint is insufficient',fine_Ad7_requires_independent_original_artwork=True,large_logo_and_border_partial_arcs_allowed=True,opposing_source_paint_interfaces_require_observed_blur_support_narrower_than_their_separation=True,foreground_pixels_excluded=True,user_counterexample=dict(source_pts=507136,source_seconds=39.62,required='No unresolved Ad7 small-letter points; keep independently observable panel/ISU arcs')))
    print(json.dumps(dict(stage='publish',rows=len(z['pts']),candidate_rows=len(raw['pts']),missing_native_entities=len(missing))),flush=True)

def qa():
    z=dict(np.load(WORK/'verified_ads.npz'));paths=np.asarray(b.read(WORK/'verified_mapping.json')['path_ids']);review=WORK/'qa_final';review.mkdir(exist_ok=True)
    times=[22.12,23.0,25.0,26.46,28.0,28.8,30.0,32.0,34.4,35.2,36.4,37.0,37.84,39.0,39.62,40.0,41.5,43.2,44.0,44.96]
    records=[]
    for t in times:
        pt=int(round(t*12800));im=r.current(pt)[0].copy();ix=np.flatnonzero(z['pts']==pt)
        for j in ix:cv2.circle(im,tuple(np.rint(z['uv'][j]).astype(int)),1,(0,255,0),-1)
        cv2.putText(im,f'SOURCE {t:.3f}s | PTS {pt} | ad points {len(ix)}',(20,240),0,.6,(0,255,255),1,cv2.LINE_AA)
        path=review/f'points_{pt}.jpg';cv2.imwrite(str(path),im[220:760]);records.append(dict(pts=pt,path=str(path),sha256=b.sha(path),points=len(ix)))
    for i in range(0,len(records),4):
        tiles=[cv2.imread(x['path']) for x in records[i:i+4]];sheet=np.vstack(tiles);cv2.imwrite(str(review/f'sheet_{i//4}.jpg'),sheet)
    userpts=507136;adpaths=paths[z['path_indices']];small=np.array([bool(__import__('re').match(r'print_instance/Ad7::Ad7\.(top|bottom)_\d+',p)) for p in adpaths]);n=int(np.count_nonzero((z['pts']==userpts)&small))
    if n:raise ValueError('Unresolved small text remains in the user example')
    b.write(WORK/'visual_review_pending.json',dict(samples=records,user_example_Ad7_small_text_points=n,source_scope=[22.12,44.96],all_rows_programmatically_rechecked=True,every_point_manually_viewed=False,human_visual_acceptance=False))
    print(json.dumps(dict(samples=len(records),user_example_small_text_points=n)),flush=True)

def seal():
    required=[OUT/'validation.json',OUT/'combined_input/fit_input_validation.json',OUT/'review/video.json',OUT/'evidence/visual_review.json']
    for p in required:
        if not p.is_file():raise ValueError('Missing required delivery evidence: '+str(p))
    valid=b.read(required[0]);native=b.read(required[1]);video=b.read(required[2]);manifest=b.read(OUT/'increment_manifest.json')
    if not valid['passed'] or not native['passed'] or native['rows']!=valid['rows'] or video['displayed_ad_rows']!=valid['ad_rows']:raise ValueError('Delivery checks disagree')
    if not (native['input_sha256']==valid['point_input_sha256']==video['point_input_sha256']):raise ValueError('Review/native/input identity mismatch')
    allocation=b.read(OUT/'combined_input/native_solver_allocation.json')
    summary=b.read(OUT/'evidence/measurement_summary.json');gaps=b.read(OUT/'evidence/coverage_gaps.json');sampling=b.read(OUT/'evidence/sampling_report.json')
    text=f"""# S01 广告点增量 002 交接

范围为原片 22.120–44.960 秒，PTS 283136–575488，时基 1/12800，共 1143 个原始曝光帧。时间没有重置成新的源时间。

本包以冻结的不完整替换包为基线，替代增量 001。不要先叠加 001 再叠加 002。可直接测试 combined_input，也可用 apply_increment.py 从冻结基线重建。不会改写基线、001、原片或当前 native 模型。

## 本次数据

- 广告点：{valid['ad_rows']}；区间内原有非广告点：{valid['unchanged_nonad_rows']}；合并输入：{valid['rows']}。
- 密集已核验池 {sampling['dense_verified_rows']} 点，去冗余后保留 {sampling['selected_rows']} 点；23,375 个有观测的实体/帧组合和 441 个固定材料角全部保留。冗余点仍存于 evidence/reference_dense_verified_pool.npz，只供证据追溯，不是拟合输入。未改变损失权重，也未声称数值信息矩阵完全等价。
- 原测量保留其 UV、PTS 和稳定 ID。新增点来自该 PTS 的原始 RGB 材料交界。
- 每点记录源图哈希、具体参考轮廓、当前 native 路径、材料方向和局部检查。没有用未拟合 native 的投影残差来证明身份。
- 对细字检查当前帧的二维笔画结构。模板能识别广告、模糊后能匹配整组，均不意味着细字可以取点。
- 用户指出的源 39.620 秒样例中，Ad7 两行小字为 0 点；能辨认的 ISU 和广告边界局部保留。
- fixed material corner 与 curve station 分开处理；曲线 station 只在已确认的同一 native 曲线上初始化。

## 实际验证

当前 native GPU 已加载并线性化全部 {valid['rows']} 点，invalid rows = 0，目标为完整、未加权像素平方和。没有运行优化步骤。全局场景、相机、轨迹数值保留；非广告观测和 station 在新时间范围内逐字段不变。初始化残差不代表拟合准确性。

validation.json 记录基线不变、身份唯一、前景排除和错误基线/重复应用拦截。combined_input/fit_input_validation.json 记录实际 native 库哈希和全量输入检查。该检查直接使用交付 station，未重新优化 station，线性化前后的逐点求值完全一致。

**运行限制：全量优化器工作区分配未通过。** 两次实际启动检查分别在 LM 工作区和 native 曲线搜索缓存分配处遇到显存不足；记录与原始失败日志在 combined_input/native_solver_allocation.json 及其引用日志中。当时本机有另一个拟合进程运行，未打断它。线性化自身占用 14,234,112,928 字节；成功加载、求值与线性化不证明当前显存条件能启动完整求解。没有通过继续删点、缩减参数或改变目标绕过此问题，未执行任何拟合步骤。

## 使用

直接输入目录：
{OUT/'combined_input'}

恢复数值状态：
{OUT/'combined_input/rebound_numeric_state.npz'}

若从基线重建：
python apply_increment.py --baseline "{b.OUT/'native_bundle'}" --increment "{OUT}" --output "<新的空输出目录>" --state "{b.OUT/'native_bundle/rebound_numeric_state.npz'}"

apply_increment.py 要求基线 point_observations.npz 的 SHA256 为 {manifest['baseline_point_input_sha256']}，并验证包内输入文件哈希。模型、相机或控制文件改变后需重新核验对应 authority，不能把本次检查结果直接套用。

## 视频与可见性限制

review/S01_ads_checked_increment002.mp4 只显示当帧广告观测，不显示拟合投影、人体点、冰面点或轨迹。小圆点为曲线观测，方框为固定材料角。画面保留原始 1920×1080，底部另加源时间/PTS 信息。用 PotPlayer 播放。视频文件时间从 0 开始，底部和 frame_index.jsonl 保留真正源时间。

当前仍有 {len(gaps['unobserved_retained_native_entities'])} 个 native 细部轮廓没有通过可辨识性检查；逐项见 evidence/coverage_gaps.json。可见但 native 尚无对应实体的画面内容不能硬绑到其他曲线，例如 Ad3 的“大吉”两字。此包不声称覆盖画面所有印刷细节。

全部点做了程序化来源、实体和局部可辨识性检查，并检查了代表帧图像；没有声称逐点人工目检或用户已验收。此包是经过当前输入检查的拟合数据，不是已经达到准确性结论的拟合结果。
"""
    (OUT/'HANDOFF.md').write_text(text,encoding='utf8')
    files=[OUT/'HANDOFF.md',OUT/'increment_manifest.json',OUT/'apply_increment.py',OUT/'delta_operations.jsonl.gz',OUT/'source_integrity.json',OUT/'validation.json']
    files+=sorted((OUT/'evidence').glob('*'))
    files+=[OUT/'ads_only'/n for n in ['point_observations.npz','mapping.json','initialized_points.npz','native_entities.json','station_initialization.json','initialized_evaluation.json']]
    files+=[OUT/'combined_input'/n for n in ['point_observations.npz','mapping.json','initialized_points.npz','point_input.json','rebound_numeric_state.npz','rebound_numeric_state.json','fit_input_validation.json','activity_pixels.json','solve.jsonl','native_solver_allocation.json','native_validation_console.log','native_validation_current_console.log','native_input_linearization_console.log']]
    files+=[OUT/'review'/n for n in ['S01_ads_checked_increment002.mp4','video.json','frame_index.jsonl','encoded_review.json']]
    files+=sorted((OUT/'validation_implementation').glob('*'))
    for p in files:
        if not p.is_file():raise ValueError('Missing sealed file: '+str(p))
    package=OUT/'S01_ads_checked_increment002.zip'
    if package.exists() or (OUT/'delivery_manifest.json').exists():raise ValueError('Do not rewrite a sealed delivery')
    delivery=dict(schema='rinkspace.checked_ad_delivery.v2',source_scope=manifest['source_scope'],current_input_sha256=valid['point_input_sha256'],ad_rows=valid['ad_rows'],all_native_rows=valid['rows'],native_input_evaluation_linearization_passed=True,full_solver_allocation_passed=allocation['full_solver_allocation_passed'],optimizer_steps=0,baseline_unchanged=True,supersedes_increment001=True,human_visual_acceptance=False,files=[dict(path=p.relative_to(OUT).as_posix(),sha256=b.sha(p),bytes=p.stat().st_size) for p in files])
    b.write(OUT/'delivery_manifest.json',delivery);files.append(OUT/'delivery_manifest.json')
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for p in files:archive.write(p,p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(package) as archive:
        if archive.testzip() is not None:raise ValueError('ZIP CRC verification failed')
        entries={p.relative_to(OUT).as_posix():b.sha(p) for p in files}
        import hashlib
        for name,digest in entries.items():
            h=hashlib.sha256()
            with archive.open(name) as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            if h.hexdigest()!=digest:raise ValueError('ZIP member hash mismatch: '+name)
    report=dict(package=str(package),sha256=b.sha(package),bytes=package.stat().st_size,verified_members=len(files),video=str(OUT/'review/S01_ads_checked_increment002.mp4'),video_sha256=video['sha256'])
    b.write(OUT/'package_digest.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':{'publish':publish,'qa':qa,'seal':seal}[sys.argv[1]]()
