"""Build the selected private offline body review without fitting or publishing."""
from pathlib import Path
from collections import Counter
import argparse,ast,csv,datetime,hashlib,importlib.metadata,io,json,platform,re,subprocess,sys,zipfile
import numpy as np


def digest(data):return hashlib.sha256(data).hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def norm(value):return re.sub('/+','/',str(value).replace(chr(92),'/')).casefold()
def imports(nodes):
    for n in nodes:
        if isinstance(n,(ast.Import,ast.ImportFrom)):yield n
        elif not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            for _,v in ast.iter_fields(n):
                if isinstance(v,list):yield from imports([x for x in v if isinstance(x,ast.stmt)])


def source_closure(project):
    pending=list((project/'calibration/program_tracking/body_tracking').glob('rebuild*.py'))+list((project/'calibration/program_tracking/body_tracking').glob('footwear*.py'));seen=set()
    def find(parts):
        for prefix in [project,project/'src']:
            for f in [prefix.joinpath(*parts).with_suffix('.py'),prefix.joinpath(*parts)/'__init__.py']:
                if f.is_file():return f
    while pending:
        p=pending.pop()
        if p in seen:continue
        seen.add(p);rel=p.relative_to(project/'src') if p.is_relative_to(project/'src') else p.relative_to(project);mod=list(rel.with_suffix('').parts)
        for n in imports(ast.parse(p.read_text(encoding='utf-8-sig')).body):
            targets=[x.name.split('.') for x in n.names] if isinstance(n,ast.Import) else [(mod[:-n.level] if n.level else [])+(n.module.split('.') if n.module else [])]
            for parts in targets:
                f=find(parts)
                if f:pending.append(f)
    for p in list(seen):
        for parent in p.parents:
            if parent==project:break
            init=parent/'__init__.py'
            if init.is_file():seen.add(init)
    return sorted(seen)


class Bundle:
    def __init__(self,file):
        self.zip=zipfile.ZipFile(file,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9);self.files={};self.resources={};self.arrays={};self.transformed=[];self.dedup={}
    def add(self,name,data,origin=None,kind='record',transform=None,deduplicate=False):
        data=bytes(data);sha=digest(data);key=(kind,sha);actual=self.dedup.get(key,name) if deduplicate else name
        if actual not in self.files:
            self.zip.writestr(actual,data);self.files[actual]=dict(path=actual,bytes=len(data),sha256=sha,kind=kind,origins=[])
            if transform:self.files[actual]['transform']=transform
            if deduplicate:self.dedup[key]=actual
        elif self.files[actual]['sha256']!=sha:raise ValueError('Conflicting member '+actual)
        if origin:
            self.files[actual]['origins'].append(str(origin));self.resources[norm(origin)]=actual
        if actual.endswith('.npz') and actual not in self.arrays:
            with np.load(io.BytesIO(data),allow_pickle=False) as z:
                self.arrays[actual]={k:dict(dtype=str(z[k].dtype),shape=list(z[k].shape),uncompressed_bytes=z[k].nbytes,raw_array_sha256=digest(z[k].tobytes(order='C'))) for k in z.files}
        return actual
    def file(self,name,path,kind='record',deduplicate=False):return self.add(name,Path(path).read_bytes(),Path(path).resolve(),kind,deduplicate=deduplicate)
    def json(self,name,value,kind='record'):return self.add(name,json.dumps(value,ensure_ascii=False,indent=2).encode('utf8'),kind=kind)
    def slim(self,name,path,keep):
        path=Path(path);original=path.read_bytes();z=load(path);selected={k:z[k] for k in keep if k in z};out=io.BytesIO();np.savez_compressed(out,**selected)
        transform=dict(operation='array_subset_without_value_changes',original_sha256=digest(original),original_bytes=len(original),retained_arrays=list(selected),omitted_arrays=[k for k in z if k not in selected])
        actual=self.add(name,out.getvalue(),path.resolve(),'numeric audit subset',transform);self.transformed.append(dict(member=actual,**transform));return actual
    def events(self,name,path):
        path=Path(path);raw=path.read_bytes();kept=[];evaluations=0
        for line in raw.decode('utf8').splitlines():
            row=json.loads(line)
            if row.get('event')=='evaluation':evaluations+=1
            else:kept.append(line)
        transform=dict(operation='retain_start_accepted_complete_events',original_sha256=digest(raw),original_bytes=len(raw),removed_finite_difference_evaluation_lines=evaluations,retained_event_lines=len(kept))
        actual=self.add(name,('\n'.join(kept)+'\n').encode('utf8'),path.resolve(),'optimizer event subset',transform);self.transformed.append(dict(member=actual,**transform));return actual


def build(project,root,config_path,output):
    project,root,config_path,output=map(lambda p:Path(p).resolve(),[project,root,config_path,output]);cfg=json.loads(config_path.read_text(encoding='utf8'));programme=config_path.parent
    output.mkdir(exist_ok=True);archive=output/(cfg['package_basename']+'.zip')
    if archive.exists():raise FileExistsError('Preserve existing review '+str(archive))
    pending=archive.with_suffix('.pending.zip');b=Bundle(pending);meta=json.loads((root/'input.json').read_text(encoding='utf8'));p=load(root/'input.npz');body_comparison=json.loads((root/'comparison.json').read_text(encoding='utf8'));foot_binding=json.loads((programme/'footwear_s01_comparison_20261029.json').read_text(encoding='utf8'))
    index=dict(schema='rinkspace.portable-body-review-index.v1',event=cfg['event'],programme=cfg['programme'],shot=cfg['shot'],source=cfg['source'],body_source_indices=cfg['body_source_indices'],stages=[],body_results={},datasets={})
    stage_rows=[]
    def copy(relative,destination=None,kind='record',dedup=False):return b.file(destination or 'data/'+relative,root/relative,kind,dedup)
    index['body_input']=copy('input.npz','data/body/input.npz','frozen input');copy('input.json','data/body/input.json');copy('camera_snapshot.npz','data/body/camera_snapshot.npz','frozen camera')
    for name in ['compact_smplx.npz','compact_smplx.json','contact_evidence.json','archive_receipt.json','verification.json','feet_review_geometry.json','review_images.json']:
        copy(name,'data/body/'+name)
    index['body_comparison']=copy('comparison.json','data/body/comparison.json')
    for mode in cfg['body_modes']:
        prefix='data/body/results/'+mode;z=load(root/mode/'result.npz');report=json.loads((root/mode/'result.json').read_text(encoding='utf8'));config=json.loads((root/mode/'config.json').read_text(encoding='utf8'));threshold=config['tolerance']['gtol']
        stage=dict(id='body_'+mode,label=mode,result=copy(mode+'/result.npz',prefix+'/result.npz','saved complete result'),report=copy(mode+'/result.json',prefix+'/result.json'),config=copy(mode+'/config.json',prefix+'/config.json'),events=copy(mode+'/fit_progress.log',prefix+'/fit_progress.log'),phase='main_body',gtol=threshold,numerically_converged=report['optimality']<=threshold)
        copy(mode+'/exposure_motion_diagnostic.npz',prefix+'/exposure_motion_diagnostic.npz');index['body_results'][mode]=stage['result'];index['stages'].append(stage)
        stage_rows.append(dict(stage=stage['id'],phase='main_body',cost=report['cost'],optimality=report['optimality'],gtol=threshold,numerically_converged=stage['numerically_converged'],termination=report['message'],nfev=report['nfev'],njev=report['njev'],elapsed_seconds=report['elapsed_seconds']))
    # Earlier failed assumptions are audit history, not additional accepted runs.
    for name in ['input.json','input.npz','comparison.json','contact_evidence_after_visibility_review.json']:
        copy('exposure_frozen/'+name,'history/exposure_frozen/'+name)
    for mode in cfg['body_modes']:
        src='exposure_frozen/'+mode;prefix='history/'+src;report=json.loads((root/src/'result.json').read_text(encoding='utf8'));config=json.loads((root/src/'config.json').read_text(encoding='utf8'));threshold=config['tolerance']['gtol']
        stage=dict(id='earlier_frozen_'+mode,phase='historical_body_approximation',result=b.slim(prefix+'/result_audit.npz',root/src/'result.npz',cfg['historical_body_npz_keep']),report=copy(src+'/result.json',prefix+'/result.json'),config=copy(src+'/config.json',prefix+'/config.json'),events=copy(src+'/fit_progress.log',prefix+'/fit_progress.log'),gtol=threshold,numerically_converged=report['optimality']<=threshold)
        index['stages'].append(stage);stage_rows.append(dict(stage=stage['id'],phase=stage['phase'],cost=report['cost'],optimality=report['optimality'],gtol=threshold,numerically_converged=stage['numerically_converged'],termination=report['message'],nfev=report['nfev'],njev=report['njev'],elapsed_seconds=report['elapsed_seconds']))
    for failed in ['dynamic_export_failure','dynamic_export_failure_2']:
        stage=dict(id=failed,phase='failed_export',status='no_final_result',exception_text_preserved=False)
        for name in ['config.json','fit_progress.log','last_accepted.npz']:
            member=copy(failed+'/'+name,'history/'+failed+'/'+name)
            stage[{'config.json':'config','fit_progress.log':'events','last_accepted.npz':'checkpoint'}[name]]=member
        index['stages'].append(stage);stage_rows.append(dict(stage=failed,phase='failed_export',termination='no final result or exception transcript; preserved checkpoint/log only',numerically_converged=False))
    # All source observation records and unknown states; duplicate JSON kept once.
    catalog={}
    for folder,label in [('footwear_v2','sparse'),('footwear_v2/dense','dense'),('footwear_v2/pose_transfer','pose_white'),('footwear_v2/semantic','semantic'),('footwear_v2/semantic_transfer','pose_semantic')]:
        dataset={}
        for name in ['admission.json','check_admission.json','source_index.json','model_selection_freeze.json','white_boot_material.json','white_boot_material_before_size_check.json','admission_before_edge_profiles.json','comparison.json','blade_check.json']:
            if not (root/folder/name).exists():continue
            dataset[name]=copy(folder+'/'+name,'data/footwear/observations/'+label+'/'+name,'source observation or audit',True)
        index['datasets'][folder]=dataset
        rows=json.loads((root/folder/'source_index.json').read_text(encoding='utf8'))['rows']
        for row in rows:
            if row['status']!='source_crop_available':continue
            key=str(row['pts']);entry=catalog.setdefault(key,dict(pts=row['pts'],source_seconds=row['source_seconds'],crop_xyxy=row['crop_xyxy'],original_png_sha256=row['crop_sha256'],decoded_native_crop_rgb_sha256=row['decoded_native_crop_rgb_sha256'],original_crop_path=row['crop'],roles=[]))
            if entry['crop_xyxy']!=row['crop_xyxy'] or entry['original_png_sha256']!=row['crop_sha256']:raise ValueError('Conflicting crop geometry '+key)
            entry['roles'].append(dict(dataset=label,role=row['role']))
    index['source_crop_catalog']=b.json('media/source_crop_catalog.json',list(catalog.values()))
    # Small original SAM/matte numeric candidates are retained, including rejected candidates.
    for f in sorted((root/'footwear_v2/mask_candidates').glob('*')):
        if f.suffix in ['.npz','.json']:b.file('data/footwear/mask_candidates/'+f.name,f,'source segmentation candidate')
    for f in sorted((root/'footwear_v2/dense/white_boot_mattes').glob('*.npz')):b.file('data/footwear/white_boot_mattes/'+f.name,f,'source white material candidate')
    for stagecfg in cfg['footwear_stages']:
        folder=stagecfg['directory'];prefix='data/footwear/results/'+stagecfg['id'];z=load(root/folder/'result.npz');report=json.loads((root/folder/'result.json').read_text(encoding='utf8'))
        np.testing.assert_array_equal(z['P_world_m'],load(root/'pose/result.npz')['P_world_m']);np.testing.assert_array_equal(z['Q_world'],load(root/'pose/result.npz')['Q_world'])
        stage=dict(id=stagecfg['id'],label_zh=stagecfg['label_zh'],phase='footwear',observation_dataset=stagecfg['observation_dataset'],review='programme/'+stagecfg['review'],result=b.slim(prefix+'/result_audit.npz',root/folder/'result.npz',cfg['footwear_npz_keep']),report=copy(folder+'/result.json',prefix+'/result.json'),config=copy(folder+'/config.json',prefix+'/config.json'),events=b.events(prefix+'/optimizer_events.jsonl',root/folder/'fit_progress.jsonl'),gtol=1e-4,numerically_converged=report['optimality']<=1e-4,pelvis_and_Q_equal_B=True,selected_development_candidate=stagecfg['selected'])
        for f in sorted((root/folder).glob('source_*.py')):b.file(prefix+'/'+f.name,f,'exact recovered solver-at-run source')
        for f in sorted((root/folder).glob('source_review_at_run.json')):b.file(prefix+'/'+f.name,f,'objective at numerical run')
        index['stages'].append(stage);stage_rows.append(dict(stage=stage['id'],phase='footwear',cost=report['cost'],optimality=report['optimality'],gtol=1e-4,numerically_converged=stage['numerically_converged'],termination=report.get('termination_message',report.get('termination')),nfev=report['nfev'],njev=report['njev'],elapsed_seconds=report['elapsed_seconds']))
        if stage['id']=='shoe_semantic':index['footwear_shared_fit_result']=stage['result']
        if stagecfg['selected']:index['footwear_final_result']=stage['result']
    for f in (root/'footwear_v2/source_v1').glob('*.py'):b.file('history/footwear_source_v1/'+f.name,f,'earlier source snapshot')
    for name in cfg['programme_files']:
        member=b.file('programme/'+name,programme/name,'programme binding')
        if name=='footwear_s01_comparison_20261029.json':index['footwear_binding']=member
        if name=='footwear_s01_parameters_20261029.npz':index['footwear_parameters']=member
        if name=='footwear_s01_semantic_20261029.json':index['footwear_fit_review']=member
        if name=='footwear_s01_geometry_check_20261029.json':index['footwear_check_review']=member
    b.file('programme/'+config_path.name,config_path,'package selection')
    # Provenance of the reused source observations, limited to this shot.
    rawobs=load(root/'source_cache/observations.npz');mask=rawobs['shot_index']==1;n=len(rawobs['pts']);subset={k:(v[mask] if v.ndim and len(v)==n else v) for k,v in rawobs.items()};buf=io.BytesIO();np.savez_compressed(buf,**subset)
    b.add('data/source/observations_s01.npz',buf.getvalue(),root/'source_cache/observations.npz','source subset',dict(operation='select shot_index == 1',source_sha256=digest((root/'source_cache/observations.npz').read_bytes()),source_rows=n,selected_rows=int(mask.sum())))
    for name in ['raw_shot_01.npz','rest_stature_audit.json','neutral_shared_beta_rest_mesh.npz']:copy('source_cache/'+name,'data/source/'+name)
    oldprior=load(root/'source_cache/body_prior.npz');np.testing.assert_array_equal(oldprior['shared_betas'],p['shared_betas'])
    b.json('data/source/body_prior_usage.json',dict(original_path=str(root/'source_cache/body_prior.npz'),original_sha256=digest((root/'source_cache/body_prior.npz').read_bytes()),shared_betas=p['shared_betas'].tolist(),other_arrays_not_consumed_by_rebuild_prepare=True,network_world_translation_consumed=False,exclusion='full-programme offsets are unused and omitted; raw shot articulation and compact model included'))
    ledger_path=next(Path(k) for k in meta['frozen_inputs'] if k.endswith('native60_to_official50.jsonl'));ledger_raw=ledger_path.read_bytes();ledger=[]
    for line in ledger_raw.decode('utf8').splitlines():
        row=json.loads(line);pts=row.get('accepted_official_pts')
        if row.get('shot_index')==1 and pts is not None and cfg['source']['first_pts']<=pts<=cfg['source']['last_pts']:ledger.append(row)
    b.add('data/source/exposure_correspondence_s01.jsonl',('\n'.join(json.dumps(r,ensure_ascii=False) for r in ledger)+'\n').encode('utf8'),ledger_path,'source correspondence subset',dict(source_sha256=digest(ledger_raw),operation='all confirmed shot 1 mappings inside selected official source range',rows=len(ledger)))
    index['source_clip']=copy('offline_review/source_s01.mp4','media/source_s01.mp4','source bitstream slice');index['source_clip_receipt']=copy('offline_review/source_clip_receipt.json','media/source_clip_receipt.json','source identity receipt')
    for pts in cfg['key_crop_pts']:copy('footwear_v2/source_crops/'+str(pts)+'.png','media/key_crops/'+str(pts)+'.png','unchanged source crop')
    index['figures']={}
    for rel in cfg['images']:index['figures'][rel]=copy(rel,'figures/'+rel,'saved diagnostic image')
    copy('offline_review/prepare_source_clip.py','tools/source_clip_preparation_at_author.py','source provenance helper')
    # Use the frozen build source, not the neighbour's evolving working tree.
    buildpath=Path(meta['native_library']).parent/'build.json';build=json.loads(buildpath.read_text(encoding='utf8'));b.file('native/build_identity.json',buildpath)
    native_root=Path(build['frozen_source']);verified=[]
    for row in build['source_files']:
        f=native_root/row['path'];raw=f.read_bytes();assert digest(raw)==row['sha256'];b.add('native/frozen_source/'+row['path'],raw,f,'exact native build source');verified.append(row['path'])
    b.json('native/build_source_verification.json',dict(all_recorded_source_hashes_match=True,source_files=len(verified),dll_sha256=build['sha256'],dll_included=False))
    run_root=Path(meta['run_root']);b.file('data/body/frozen_joint_manifest.json',run_root/'joint_manifest.json')
    field=Path(meta['scene_path']);field_sha=digest(field.read_bytes());expected=next(value for key,value in meta['frozen_inputs'].items() if norm(key)==norm(field))
    assert field_sha==expected;b.file('data/body/frozen_field_model.json',field,'frozen field geometry')
    closure=source_closure(project)
    for f in closure:b.file('repository/'+f.relative_to(project).as_posix(),f,'Python source at packaging')
    for name in ['test_rebuild_exposure_body.py','test_footwear_geometry.py']:b.file('repository/tests/'+name,project/'tests'/name,'test source')
    for name in ['README.md','LICENSE.md','AGENTS.md','MAINTENANCE.md','pyproject.toml','docs/VALIDATION.md','docs/SOLVER.md','docs/REBUILD.md','docs/GITHUB_REVIEW.md','docs/LICENSING_REVIEW.md','environments/README.md','models/SAM2_LICENSE.txt','environments/third_party_patches/GVHMR_LICENSE.txt']:
        if (project/name).exists():b.file('repository/'+name,project/name,'project or rights context')
    b.file('tools/build_body_review.py',Path(__file__),'package builder');b.file('AUDIT.py',project/'scripts/audit_body_review.py','offline audit tool')
    environment={}
    for name in ['numpy','scipy','opencv-python','av','torch','transformers','pillow','matplotlib','imageio-ffmpeg','smplx']:
        try:environment[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:environment[name]='not installed in packaging interpreter'
    b.json('ENVIRONMENT.json',dict(python=sys.version,platform=platform.platform(),packages=environment,numerical_device='CPU; no new GPU inference or neighbour solve',large_weights_or_environment_included=False,source_closure_policy='module-import dependency chain; optional historical rendering/inference function dependencies remain external'))
    b.json('TRANSFORMS.json',b.transformed);b.json('ARRAYS.json',b.arrays)
    b.json('RESOURCE_MAP.json',dict(normalization='replace backslashes with slash, collapse repeated slash, casefold',by_original_path=b.resources,omitted_crops='Use media/source_crop_catalog.json and tools/RESTORE_CROPS.py with media/source_s01.mp4'))
    index['stage_rows']=stage_rows;index['review_questions_zh']=cfg['review_questions_zh'];index['claim_status']=dict(accepted_3d_trajectory=False,final_independent_holdout=False,gradient_converged_main_body=False,gradient_converged_footwear=False,camera_updates=False,contact_to_root_enabled=False)
    b.json('REVIEW_INDEX.json',index)
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=['stage','phase','cost','optimality','gtol','numerically_converged','termination','nfev','njev','elapsed_seconds']);writer.writeheader();writer.writerows(stage_rows);b.add('tables/stage_outcomes.csv',buf.getvalue().encode('utf-8-sig'),kind='derived stage outcomes')
    buf=io.StringIO();w=csv.writer(buf);w.writerow(['source_pts','source_seconds','lower_body_inferred','A_X','A_Y','A_Z','B_X','B_Y','B_Z','C_X','C_Y','C_Z','B_vs_A_XY_conditional_m','C_vs_A_XY_conditional_m'])
    z={m:load(root/m/'result.npz') for m in cfg['body_modes']}
    for i,pts in enumerate(p['pts']):w.writerow([int(pts),float(pts)*cfg['source']['time_base_num']/cfg['source']['time_base_den'],bool(p['lower_body_inferred'][i]),*z['fixed']['P_world_m'][i],*z['pose']['P_world_m'][i],*z['pose_contact']['P_world_m'][i],float(np.linalg.norm(z['pose']['G_world_m'][i]-z['fixed']['G_world_m'][i])),float(np.linalg.norm(z['pose_contact']['G_world_m'][i]-z['fixed']['G_world_m'][i]))])
    b.add('tables/pelvis_trajectory_all_exposures.csv',buf.getvalue().encode('utf-8-sig'),kind='derived frame trajectory')
    # Recovery uses original source RGB and the recorded integer crop rectangle.
    restore='''from pathlib import Path
import argparse,hashlib,json
import av,cv2
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--pts',nargs='*',type=int);a=ap.parse_args()
rows=json.loads((a.root/'media/source_crop_catalog.json').read_text(encoding='utf8'));rows={r['pts']:r for r in rows if not a.pts or r['pts'] in a.pts};a.output.mkdir(parents=True,exist_ok=False);done=[]
with av.open(str(a.root/'media/source_s01.mp4')) as c:
 s=c.streams.video[0];s.codec_context.thread_count=2
 for f in c.decode(s):
  if f.pts not in rows:continue
  r=rows[f.pts];x0,y0,x1,y1=r['crop_xyxy'];image=f.to_ndarray(format='bgr24')[y0:y1,x0:x1].copy();assert hashlib.sha256(image[:,:,::-1].tobytes()).hexdigest()==r['decoded_native_crop_rgb_sha256'];target=a.output/(str(f.pts)+'.png');assert cv2.imwrite(str(target),image);done.append(f.pts)
assert set(done)==set(rows);print(json.dumps(dict(recovered_source_crops=len(done),all_original_RGB_hashes_match=True)))
'''
    b.add('tools/RESTORE_CROPS.py',restore.encode('utf8'),kind='portable source crop recovery')
    # The short human entry point names the negative results and denominator.
    mainrows=[]
    for mode,label in [('fixed','A 固定局部姿态'),('pose','B 局部姿态修正'),('pose_contact','C 姿态＋粗接冰')]:
        m=body_comparison['models'][mode];mainrows.append('| '+label+' | '+f"{m['source_body_reprojection_px']['median']:.3f} / {m['source_body_reprojection_px']['p95']:.3f}"+' | '+f"{m['diagnostic_window_body_reprojection_px']['median']:.3f} / {m['diagnostic_window_body_reprojection_px']['p95']:.3f}"+' | '+f"{m['ground_displacement_vs_fixed_conditional_m']['median']:.3f} / {m['ground_displacement_vs_fixed_conditional_m']['maximum']:.3f}"+' |')
    shoe_rows=[]
    for label,key in [('原 B','previous_B'),('最终鞋体开发候选','calibrated_observed_pose')]:
        m=foot_binding['models'][key];a=m['fitted_blade_curves']['raw_pixel_distance'];d=m['development_blade_curves']['raw_pixel_distance'];shoe_rows.append(f"| {label} | {a['median']:.3f} / {a['p95']:.3f} | {d['median']:.3f} / {d['p95']:.3f} |")
    questions='\n'.join(str(i+1)+'. '+q for i,q in enumerate(cfg['review_questions_zh']))
    readme=f'''# S01 人体定位整轮实验：PRO 离线审核包

这是私有审核快照，覆盖从当前冻结相机上的 A/B/C 人体对照，到鞋刀观测和几何改造的完整开发链。主要结果是：局部姿态修正带来小幅身体重投影收益；原接冰候选不能作为更准轨迹；新鞋刀在部分拟合帧明显改善，但开发检查仍未通过。最终鞋体候选没有反馈到骨盆。此包不宣称恢复了真实厘米精度，也没有实现跨镜头人体联合求解。

## 先读这些内容

1. 本说明和 [勘误／限制](ERRATA_AND_LIMITS.md)。
2. [原 A/B/C 诊断图](figures/diagnostic_summary.png)、[36.020 秒完整旧对照](figures/detail_461056.png)、[32.960 秒问题细节](figures/detail_421888.png)。
3. [新鞋刀 36.020 秒](figures/footwear_v2/semantic_transfer/delivery/before_after_461056.png)和[26.260 秒失败检查](figures/footwear_v2/semantic_transfer/delivery/before_after_336128.png)，以及[38.220 秒检查](figures/footwear_v2/semantic_transfer/delivery/before_after_489216.png)。红线是源图刀刃读数，语义点由助手近似读取，并非用户确认或三维真值。
4. [REVIEW_INDEX.json](REVIEW_INDEX.json)给出每阶段的包内参数、报告、观测与日志地址；[阶段表](tables/stage_outcomes.csv)列出停止条件。原报告原字节保留；相互矛盾处以勘误说明和原始数值共同审查。
5. 源码入口：repository/calibration/program_tracking/body_tracking/rebuild_exposure_body.py、rebuild_body_fit.py、rebuild_body_geometry.py，以及 footwear_fit.py、footwear_geometry.py、footwear_rgb_observations.py。native/frozen_source 是与本轮投影 DLL 构建清单逐个 SHA 核对的相机数学源码。

## 范围与冻结身份

- 赛事 Ankara 2026，冈田芽衣短节目，S01 分析范围 22.120–44.960 秒，PTS 283136–575488，步长 256，time_base=1/12800，1920×1080，原生 50 Hz，共 1143 曝光。
- 其中 125 帧下肢为推断，1018 帧可见人体进入主要诊断。开头裁切和孤立错误 full-body 标志已修正；早期输入及失败仍在 history/ 中。状态按源 PTS 保留，不能把样条补出的鞋位置当成原图观测。
- 相机使用本轮开始时冻结的 lane4_shared/initial_pixels.npz，原 SHA 为 c205f10937004c93529cf40cf3d6030750754146741178547db5abd052891544。它不是邻会话后来继续变化的状态。S01 rig、全部所需 controls、场景数值在 data/body/input.npz 和 camera_snapshot.npz，来源在 input.json。
- 完整 native 相机保留 Brown/wide、入口瞳和滚动快门。动态身体与相机在同一预测行时刻投影；身体由每曝光三个精确姿态样本作二次近似。曝光内冻结身体的旧对照在 history/exposure_frozen/。
- 三维输出定义始终是 SMPL-X 骨盆 P 及其世界竖直投影 G=(P_x,P_y,0)。A/B/C 的 P 可变；后续鞋刀阶段的 P、Q 与 B 完全相同。鞋刀世界最低点只是模型诊断，不是观测到的接冰点。
- 冰场 L=1 暂按 60 条件米，体型使用既有 shared beta 和约 1.53 米身高审计，body_scale={float(p['body_scale']):.12f}。这些不是现场实测尺度；不能将条件米制位移当真实定位误差。

## A/B/C 结果

复用确认曝光对应的 ViTPose COCO17 与 GVHMR 局部姿态，postproc=False；未使用网络世界平移或旧 demo 修补的 XY。全程共享体型和骨长，根位置/朝向样条间隔 0.16 秒。B/C 允许左右髋、膝、踝和三段脊柱共九个屈伸修正。拟合消费 13 个 COCO 点，手肘/手腕留作额外诊断；关节点工程尺度与先验见各 config.json。

| 组 | 可见且合格身体点 px，中位/P95 | 开发诊断窗身体点 px，中位/P95 | 相对 A 的 G 位移，条件米，中位/最大 |
|---|---|---|---|
{chr(10).join(mainrows)}

C 只使用四个粗刀刃图像区域和模型高度条件；另两个时刻及其前后 100 ms 的身体像素不参加该组源像素拟合。但非穿冰先验覆盖可见帧，包括这些诊断窗。因此这些接触高度不是独立三维真值。C 以较大轨迹位移换取模型近冰状态，并牺牲部分未拟合观测，未被选为准确轨迹。ABC 的成功停止均为 ftol，不是梯度收敛。

## 鞋刀改造与失败链

1. 稀疏源裁剪＋SAM2 候选，区分白鞋与含刀刃的整套装备轮廓，排除粘连、左右未决和遮挡。共享鞋体、独立双刃抛物线下缘，先只调几何，再允许腿部修正。
2. 稀疏检查失败后增加到约 5 Hz 鞋部观测，并用源 RGB 白鞋材料、真实边缘剖面与实例大小检查筛选轮廓。保留把冰面误分为鞋的失败及修正前阈值。
3. 冻结尺寸，仅用新窗口白鞋轮廓恢复腿部，刀刃方向仍可能错误。随后补鞋头/鞋跟语义，证明单独白鞋上部轮廓不能确定鞋的三维方向。
4. 最终为 236 个脚候选中的 206 组自动准入轮廓、11 组源图近似鞋头/鞋跟语义、7 条拟合刀刃曲线（34 原始顶点）。三条开发检查刀刃曲线（13 原始顶点）没有进该次拟合，但它们此前已驱动修改，不能称最终独立测试。
5. 鞋型全镜头共享；腿部增加髋各 3 轴、膝各屈伸 1 轴、踝各 3 轴，共 14 个修正自由度，0.4 秒样条。没有逐帧自由鞋体 XYZ、逐帧鞋长或接冰高度损失。尺寸、旋转、加速度先验都是显式工程尺度，不是校准后的观测噪声。

| 版本 | 七条拟合刀刃曲线 px，中位/P95 | 三条开发检查刀刃曲线 px，中位/P95 |
|---|---|---|
{chr(10).join(shoe_rows)}

中位数由每条源曲线等距采样 24 点后的点到有限预测曲线距离汇总；这些采样不是 168 或 72 个独立人工标注。检查尾部减少，但中位数变差，鞋后段长度卡在下界，梯度也未收敛。最终膝点中位误差由约 7.0 增至 8.0 px。新候选只作开发审阅，contact_allowed_for_root 全部为 False。其 1143 帧输出包括大量没有当前鞋像素的样条推断，source_pose_status 逐帧保留这一差别。

## 原像素与离线核验

media/source_s01.mp4 是从官方原片复制的 AV1 压缩包，保留原 PTS，不重编码、无音轨。为解码保留从 21.600 秒开始的 26 个前置曝光；目标仍是 22.120–44.960 秒共 1143 个。media/source_clip_receipt.json 记录原文件与切片 SHA、每包身份和全部目标帧的解码像素哈希；制包时已与原片逐帧比较一致。该切片是原素材，不是新拟合视频；没有生成完整叠加验收视频。

关键原始鞋部裁剪在 media/key_crops/。为减少重复，其余原裁剪不重复存 PNG；裁剪矩形和原 RGB 哈希完整保留在 media/source_crop_catalog.json。可在无网络时从上述切片恢复任何裁剪：

    python tools/RESTORE_CROPS.py --output <新的目录> --pts 336128 461056

完整恢复省略 --pts；需 PyAV、NumPy、OpenCV。普通阅读只需 Markdown/图片和 AV1 解码器。核验包内 SHA 仅需 Python 标准库；复算已保存的原像素指标只需 NumPy，不加载任何模型或本机 DLL：

    python AUDIT.py
    python AUDIT.py --metrics
    python AUDIT.py --metrics --pixels

第三条额外需要 PyAV，检查原码流包和已记录的全部目标像素。像素转换库版本见 ENVIRONMENT.json；压缩包身份核对不依赖 RGB 色彩转换实现。上面的操作只检查已保存的数据，不重新拟合。工程导数测试源码在 repository/tests/，此前本机七项通过；它们并非此包 SHA/指标复算的替代。

## 最小化与可复现边界

- 当前 A/B/C 完整 result.npz、源观测输入、三维紧凑身体表、最终参数原件均保留。鞋体中间输出去掉可由固定 B 和参数恢复的重复三维网格、重复 P/Q；保留全部曝光 UV、最低刃高、拟合控制与源帧。每次裁剪的原 SHA、保留/省略字段见 TRANSFORMS.json；数据值没有改写。
- 有限差分 evaluation 大日志只保留 start、accepted、complete 行；原日志 SHA、原体积和省略行数都有登记。旧导出失败仅有配置、日志与检查点，没有保存异常全文，不能据此断言准确的异常根因。
- 源码为所需模块导入链，另保留已找回的鞋体实际运行版本和早期 source_v1。当前源码、实际运行源码和构建冻结源码分别登记，不把最新工作树伪称成所有历史运行的精确源码。早期部分版本缺少逐文件冻结记录，完整历史拟合不能保证逐位重放。
- 未含完整官方节目视频、网络权重、完整 SMPL-X 原模型、安装环境或 Windows DLL。紧凑 SMPL-X 派生几何表为审核所需；其来源与许可不因打包改变。该包是私有研究审核材料，不构成公开分发许可。
- 全部原盘符路径保持在原件内用于追查；RESOURCE_MAP.json 映射到 ZIP 成员，实际阅读以 REVIEW_INDEX.json 的包内相对路径为入口。缺少运行环境不妨碍阅读和复算已保存的像素统计，但重新推理/原生拟合需要另行准备原依赖。

## 请 PRO 重点审核

{questions}

请依据源观测、参数结构和保留的失败作判断；小重投影误差、平滑、优化器 success 或更少的模型穿冰都不自动证明三维准确。本次没有改写原片、源观测、邻会话联合拟合状态或 Git 历史。
'''
    b.add('README.md',readme.encode('utf8'),kind='human review entry')
    errata='''# 勘误与证据等级

1. 原 programme/body_s01_comparison_20261029.json 的 validation_status_zh 写“工程收敛及导数测试通过”。其中“工程收敛”用词不成立：A/B/C 的 optimality 分别约 0.2174、0.3469、0.3563，而 gtol=1e-5；它们是 ftol 停止。原文件保留原字节，本说明更正解释，不把 optimizer success 当梯度收敛。
2. result.json 的 all_source_detector_reprojection_px 覆盖 17 点及推断/裁切情形，主 comparison 的 source_body_reprojection_px 只覆盖 13 个源点、可见且合格帧。因此不同中位数并非同一评估池，不能直接混用；AUDIT.py 重算主 comparison 的口径。
3. 旧接触 region_distance_px=0 只表示一个模型候选落在粗矩形内，不是刀刃点准确；旧 27.08 秒区域曾没有覆盖实际下刃。鞋体升级保留原矩形历史，但用独立的可见刀刃下缘曲线作新观测。
4. 稀疏鞋体旧检查窗口 30.4–30.8、42.4–42.8 秒在失败后已用于调参。后来的 26.26–26.34、38.22–38.38 秒也已驱动加密与语义修正；最终三条未入拟合的刀刃曲线只属于开发检查，没有最终独立测试集。
5. SAM 是候选搜索，不是接冰证据。白鞋轮廓自动准入并非用户逐点确认；COCO 左右身份也不是全程已确认。鞋头/鞋跟与刀刃曲线由助手近似读数，误差尺度为工程选择，不是实测概率不确定度。
6. 多帧反复看过的问题图、24 点重采样、邻近曝光和同源 RGB 不能计成独立地标。最终 g 的鞋后段下界和姿态迁移失败显示仍有不可辨识/模型错配；骨盆冻结也可能使鞋型或腿姿态承担既有根定位误差，这是需要 PRO 审查的假设。
7. 公开身高、冰场 60 米与旧 9 厘米有效鞋高均是条件假设。模型的近冰高度、P 与 G 以及真实支撑冰刀是不同对象。没有同步多视角或外部三维测量，不能认定厘米级真实定位。
8. 这轮没有改变共同相机/场景，也没有人体反向约束相机或跨镜头联合求解；后续鞋刀阶段保持 B 的 P/Q，故不能用其 G 未变化证明定位准确。所有接触反馈标志保持关闭。
'''
    b.add('ERRATA_AND_LIMITS.md',errata.encode('utf8'),kind='corrected interpretation')
    manifest=dict(schema='rinkspace.private-offline-review-manifest.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection_sha256=digest(config_path.read_bytes()),files=list(b.files.values()),manifest_self_excluded_from_file_list=True,private_review=True)
    manifest_bytes=json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf8');b.zip.writestr('MANIFEST.json',manifest_bytes);b.zip.close();pending.replace(archive)
    receipt=dict(schema='rinkspace.body-review-package-receipt.v1',archive=str(archive),bytes=archive.stat().st_size,sha256=digest(archive.read_bytes()),manifest_sha256=digest(manifest_bytes),files=len(b.files)+1,uncompressed_bytes=sum(r['bytes'] for r in b.files.values())+len(manifest_bytes),native_source_hashes_verified=len(verified),source_python_files=len(closure),source_crop_catalog_frames=len(catalog),stages=len(index['stages']))
    (output/(cfg['package_basename']+'.sha256')).write_text(receipt['sha256']+'  '+archive.name+'\n',encoding='ascii');(output/'package_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(receipt,ensure_ascii=False),flush=True)
    return receipt


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();build(a.project,a.root,a.config,a.output)
