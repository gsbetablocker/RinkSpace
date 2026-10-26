"""Freeze, independently apply, audit, and render an additive observation increment."""
from pathlib import Path
import sys,json,hashlib,gzip,collections,shutil,zipfile
import numpy as np,cv2
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_replacement_s01 as b
import ad_replace_package_s01 as pack
import ad_apply_increment_s01 as applier
import ad_handoff_s01 as handoff
BASE=b.OUT;OUT=BASE/'increments/001';COMBINED=OUT/'combined_input'

def prepare():
    baseline=b.read(BASE/'incomplete_baseline_manifest.json');z=dict(np.load(OUT/'additions.npz'));data=b.load();restored=z['source_rows']<len(data['pts']);rows=z['source_rows'][restored];old=dict(np.load(pack.BUNDLE/'point_observations.npz'));m=b.read(OUT/'mapping.json')
    for k,expected in [('uv',data['uv'][rows]),('pts',data['pts'][rows]),('observation_ids',data['ids'][rows])]:
        if not np.array_equal(z[k][restored],expected):raise ValueError('Original source changed: '+k)
    if len(set(z['observation_ids']))!=len(z['pts']) or len(set(z['source_rows']))!=len(z['pts']):raise ValueError('Duplicate increment identity')
    if set(z['observation_ids'])&set(old['observation_ids']):raise ValueError('Baseline observations repeated')
    if not np.isfinite(z['uv']).all():raise ValueError('Nonfinite source observation')
    evidence=b.read(BASE/'source_evidence_index.json');banned=set(map(int,evidence['withdrawals']))|set(map(int,evidence['retired']))
    banned|={r['row'] for r in b.read(BASE/'boundary_identity_review.json')['changes'] if r['decision']=='exclude'}
    if set(rows)&banned:raise ValueError('Explicit withdrawal reintroduced')
    shutil.copyfile(Path(applier.__file__),OUT/'apply_increment.py')
    provenance=dict(source_rows=z['source_rows'],stable_ids=z['observation_ids'],source_dataset=np.array([str(data['datasets'][r]) if r<len(data['pts']) else 'increment001_actual_source_contour_readout' for r in z['source_rows']]),measurement_origin=np.where(restored,'restored_original_pixel','new_actual_source_pixel'),original_active=np.array([bool(data['old_active'][r]) if r<len(data['pts']) else False for r in z['source_rows']]))
    np.savez_compressed(OUT/'source_provenance.npz',**provenance)
    names=['additions.npz','mapping.json','initialized_points.npz','source_provenance.npz','apply_increment.py','recovery_decisions.jsonl.gz','new_glyph_measurements.jsonl.gz','new_visible_contours.jsonl.gz','new_flow_contours.jsonl.gz','identity_audit.json','source_flow_navigation.json']
    manifest=dict(schema='rinkspace.ad_additive_increment.v1',increment_id='s01_ad_additions_001_20261026',status='prepared_for_testing_and_user_video_review',baseline_id=baseline['baseline_id'],baseline_point_input_sha256=baseline['point_input_sha256'],baseline_manifest_sha256=b.sha(BASE/'incomplete_baseline_manifest.json'),added_rows=len(z['pts']),restored_original_source_rows=int(restored.sum()),new_actual_source_rows=int((~restored).sum()),operations=dict(add=len(z['pts']),rebind=0,remove=0),new_ad_rows_total=baseline['ad_rows']+len(z['pts']),source_scope=dict(first_pts=256256,last_pts=575488,time_base='1/12800',frames=1248),source_measurements=['recovery_decisions.jsonl.gz','new_glyph_measurements.jsonl.gz','new_visible_contours.jsonl.gz','new_flow_contours.jsonl.gz','identity_audit.json'],baseline_parameters_unchanged=True,all_new_measurements_use_current_source_pixels=True,source_navigation_is_not_observed_UV=True,optimizer_started=False,human_visual_acceptance=False,files=[dict(path=n,sha256=b.sha(OUT/n),bytes=(OUT/n).stat().st_size) for n in names])
    b.write(OUT/'increment_manifest.json',manifest)
    receipt=applier.apply(pack.BUNDLE,OUT,COMBINED,pack.BUNDLE/'rebound_numeric_state.npz');print(json.dumps(receipt),flush=True)

def validate():
    baseline=b.read(BASE/'incomplete_baseline_manifest.json');manifest=b.read(OUT/'increment_manifest.json');z=dict(np.load(COMBINED/'point_observations.npz'));old=dict(np.load(pack.BUNDLE/'point_observations.npz'));a=dict(np.load(OUT/'additions.npz'));m=b.read(COMBINED/'mapping.json');n=len(old['pts']);paths=np.asarray(m['path_ids'])[z['path_indices']];ads=np.array([int(p.split('::')[0].split('/Ad')[1]) if p.startswith('print_instance/Ad') else 0 for p in paths]);
    for entry in baseline['files']:
        if b.sha(BASE/entry['path'])!=entry['sha256']:raise ValueError('Frozen baseline modified: '+entry['path'])
    for k in old:
        if not np.array_equal(z[k][:n],old[k]):raise ValueError('Baseline array changed: '+k)
    oldstate=dict(np.load(pack.BUNDLE/'rebound_numeric_state.npz'));state=dict(np.load(COMBINED/'rebound_numeric_state.npz'))
    for k in oldstate:
        if k not in ['stations','source_rows','point_input_sha256'] and not np.array_equal(oldstate[k],state[k]):raise ValueError('Global parameter changed: '+k)
    if not np.array_equal(oldstate['stations'],state['stations'][:n]):raise ValueError('Prior station changed')
    vis={f['pts']:f for f in b.read(BASE/'visibility.json')['frames']};perframe=[]
    for fi,(pt,frame) in enumerate(sorted(vis.items())):
        rr=np.flatnonzero((z['pts']==pt)&(ads>0));uv=np.rint(z['uv'][rr]).astype(int)
        if len(rr):
            mask=cv2.imread(frame['mask_path'],0)
            if b.sha(Path(frame['mask_path']))!=frame['mask_sha256']:raise ValueError('Occlusion authority changed')
            if np.any(uv<0) or np.any(uv[:,0]>=1920) or np.any(uv[:,1]>=1080) or np.any(mask[uv[:,1],uv[:,0]]):raise ValueError('Invalid or occluded source pixel')
        perframe.append(dict(pts=pt,source_seconds=pt/12800,ad_points=len(rr),by_ad={str(k):int((ads[rr]==k).sum()) for k in range(2,13)}))
        if fi%400==0:print(json.dumps(dict(stage='validate_source_frames',frames=fi+1,total=1248)),flush=True)
    core,model,native=pack.current_native()
    if native.scene_revision!=m['scene_revision']:raise ValueError('Current scene revision changed')
    from rinkspace.domain.parameters import ParameterState
    from rinkspace.session.gpu_geometry import NativeSceneGPU
    native=NativeSceneGPU(b.SCENE,core,parameter_state=ParameterState(model.initial_state.layout,state['scene_values']));camera=b.read(m['camera_path']);camera={**camera,'rig':state['rig'].tolist()}
    with native.resident(camera=camera,controls=state['controls'],path_ids=paths.tolist(),stations=state['stations'],pts=z['pts'],observed=z['uv']) as resident:result=resident.evaluate('cuda')
    if np.any(result['status']) or not np.isfinite(result['values']).all():raise ValueError('Invalid actual native consumer evaluation')
    guards={}
    for name,base in [('wrong_baseline',pack.ADS),('double_application',COMBINED)]:
        try:applier.apply(base,OUT,OUT/('should_not_exist_'+name))
        except ValueError as e:guards[name]=str(e)
        else:raise ValueError('Import guard did not reject '+name)
    report=dict(schema='rinkspace.ad_increment_validation.v1',passed=True,baseline_id=baseline['baseline_id'],baseline_point_input_sha256=baseline['point_input_sha256'],baseline_files_unchanged=len(baseline['files']),baseline_rows_unchanged=n,nonad_rows_unchanged=int((ads==0).sum()),restored_rows_original_UV_PTS_ID_unchanged=manifest['restored_original_source_rows'],new_actual_source_rows=manifest['new_actual_source_rows'],added_rows=manifest['added_rows'],total_rows=len(z['pts']),ad_rows=int((ads>0).sum()),ad_frames=int(sum(f['ad_points']>0 for f in perframe)),frames_without_ad_points=int(sum(f['ad_points']==0 for f in perframe)),source_frames=1248,by_ad={str(k):int((ads==k).sum()) for k in range(2,13)},all_ad_points_rechecked_against_existing_SAM2=True,explicit_withdrawals_or_retired_reintroduced=0,all_previous_global_parameters_and_stations_preserved=True,native_consumer_invalid_rows=0,diagnostic_only_residual_quantiles_px=np.percentile(np.linalg.norm(result['values'][:,9:11],axis=1),[0,50,95,99,100]).tolist(),source_identity_not_inferred_from_residuals=True,import_guards=guards,optimizer_started=False,human_visual_acceptance=False,combined_input_sha256=b.sha(COMBINED/'point_observations.npz'))
    b.write(OUT/'validation.json',report);b.write(OUT/'source_frame_coverage.json',dict(frames=perframe));print(json.dumps(report),flush=True)

def render():
    handoff.BUNDLE=COMBINED;handoff.REVIEW=OUT/'review';handoff.render()

if __name__=='__main__':
    {'prepare':prepare,'validate':validate,'render':render}[sys.argv[1]]()
