"""Deliver the checked ad revision on the frozen baseline and current S01 scope."""
from pathlib import Path
import sys,json,gzip,collections,hashlib,shutil,importlib.util,zipfile
import numpy as np,cv2
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_entity_registration_s01 as reg
import ad_replace_package_s01 as pack
import ad_apply_checked_increment_s01 as applier
import ad_handoff_s01 as video
b=reg.b;WORK=reg.OUT;OUT=b.OUT/'increments/002';ADS=OUT/'ads_only';COMBINED=OUT/'combined_input';BASE=b.OUT/'native_bundle'
FIRST=283136;LAST=575488

def prepare():
    if (OUT/'delivery_manifest.json').exists():raise ValueError('Frozen delivery cannot be rewritten')
    if (OUT/'increment_manifest.json').exists():raise ValueError('Prepared increment already exists')
    z=dict(np.load(WORK/'verified_ads.npz'));m=b.read(WORK/'verified_mapping.json');summary=b.read(WORK/'measurement_summary.json');baseline=b.read(b.OUT/'incomplete_baseline_manifest.json')
    if np.any(z['pts']<FIRST) or np.any(z['pts']>LAST) or summary['source_frames']!=1143:raise ValueError('Observation scope does not match the user-selected 22.120-second start')
    if b.sha(BASE/'point_observations.npz')!=baseline['point_input_sha256']:raise ValueError('Frozen baseline changed')
    pack.authority_check();core,model,native=pack.current_native()
    if m['scene_revision']!=native.scene_revision or m['scene_sha256']!=b.sha(b.SCENE):raise ValueError('Native scene changed during source checks')
    ADS.mkdir(parents=True,exist_ok=True);np.savez(ADS/'point_observations.npz',**z)
    bm=b.read(BASE/'mapping.json');m.update(schema='rinkspace.source_checked_current_native_ads.v2',scene_path=str(b.SCENE),camera_path=bm['camera_path'],camera_sha256=bm['camera_sha256'],control_sha256=bm['control_sha256'],selected_rows=len(z['pts']),frames=len(np.unique(z['pts'])),current_point_input=str(ADS/'point_observations.npz'),fit_started=False,source_scope=dict(first_pts=FIRST,last_pts=LAST,time_base='1/12800',frames=1143),native_parameters_replaced_by_dataset=False)
    b.write(ADS/'mapping.json',m)
    state=dict(np.load(BASE/'rebound_numeric_state.npz'));initial={k:v for k,v in state.items() if k not in ['stations','source_rows','point_input_sha256']}
    initial.update(stations=z['stations'],source_rows=z['source_rows'],point_input_sha256=np.array(b.sha(ADS/'point_observations.npz')))
    np.savez(ADS/'initialization_state.npz',**initial)
    spec=importlib.util.spec_from_file_location('checked_ad_initializer',Path(__file__).with_name('initialize_native_s01.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.prep.RUN=ADS;module.initialize(ADS/'initialization_state.npz')
    if b.sha(ADS/'point_observations.npz')!=str(initial['point_input_sha256']):raise ValueError('Initializer changed an observation')
    names=['pointwise_checks.jsonl.gz','coverage_intervals.jsonl.gz','native_entity_registry.json','source_identity_charts.json','additional_source_atlases.json','missing_retained_entity_sources.json','measurement_summary.json','source_chart_join_proofs.json','artwork_source_bridges.json','source_overlap_recovery.json','partial_view_recovery.json','observability_summary.json','observability_regions.jsonl.gz','observability_rules.json','visual_review.json','coverage_gaps.json','border_completion.json','implementation_manifest.json','sampling_report.json','latin_glyph_recheck.json']
    evidence=OUT/'evidence';evidence.mkdir(exist_ok=True)
    for name in names:shutil.copyfile(WORK/name,evidence/name)
    shutil.copyfile(WORK/'dense_verified_pool/verified_ads.npz',evidence/'reference_dense_verified_pool.npz')
    original=b.load();is_original=z['source_rows']<len(original['pts']);rr=z['source_rows'][is_original]
    for key,expected in [('uv',original['uv'][rr]),('pts',original['pts'][rr]),('observation_ids',original['ids'][rr])]:
        if not np.array_equal(z[key][is_original],expected):raise ValueError('Original source UV, PTS or stable ID changed: '+key)
    previous=dict(np.load(b.OUT/'increments/001/additions.npz'));previous_index={int(x):i for i,x in enumerate(previous['source_rows'])}
    restored_extra=(~is_original)&(~np.char.startswith(z['observation_ids'],'s01_native_source_v2_'))
    for i in np.flatnonzero(restored_extra):
        j=previous_index[int(z['source_rows'][i])]
        for k in ['uv','pts','observation_ids']:
            if not np.array_equal(z[k][i],previous[k][j]):raise ValueError('Previously measured increment source pixel changed')
    old=dict(np.load(BASE/'point_observations.npz'));oldpaths=np.asarray(bm['path_ids'])[old['path_indices']];newpaths=np.asarray(m['path_ids'])[z['path_indices']];admask=np.array([p.startswith('print_instance/Ad') for p in oldpaths]);oldby={str(old['observation_ids'][j]):int(j) for j in np.flatnonzero(admask)};newby={str(s):j for j,s in enumerate(z['observation_ids'])}
    baseline_decisions={}
    with gzip.open(WORK/'pointwise_checks.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            c=json.loads(line)
            if c['stable_id'] in oldby:baseline_decisions[c['stable_id']]=c
    ops=collections.Counter();operation_rows=[]
    for sid,j in oldby.items():
        if old['pts'][j]<FIRST:
            action='exclude_before_new_start';reason='user_selected_source_start_PTS_283136'
        elif sid not in newby:
            reason=baseline_decisions.get(sid,{}).get('reason','no_current_source_admission');action='remove_redundant_ad' if reason=='redundant_same_visible_contour' else 'remove_unverified_ad'
        else:
            k=newby[sid]
            for key in ['uv','pts','source_rows']:
                if not np.array_equal(old[key][j],z[key][k]):raise ValueError('Stable source observation changed')
            changed=oldpaths[j]!=newpaths[k] or old['kind'][j]!=z['kind'][k] or (z['kind'][k]==0 and old['stations'][j]!=z['stations'][k])
            action='rebind' if changed else 'retain';reason='current_source_entity_and_material_role_rechecked'
        ops[action]+=1;operation_rows.append(dict(stable_id=sid,baseline_row=j,action=action,reason=reason,previous_native_id=str(oldpaths[j]),current_native_id=str(newpaths[newby[sid]]) if sid in newby else None))
    for sid,k in newby.items():
        if sid not in oldby:ops['add']+=1;operation_rows.append(dict(stable_id=sid,action='add',current_native_id=str(newpaths[k]),source_row=int(z['source_rows'][k]),pts=int(z['pts'][k])))
    with gzip.open(OUT/'delta_operations.jsonl.gz','wt',encoding='utf8') as f:
        for c in operation_rows:f.write(json.dumps(c,ensure_ascii=False)+'\n')
    registry=b.read(WORK/'native_entity_registry.json');counts=collections.Counter(newpaths.tolist());entities=[]
    for c in registry['paths']:entities.append(dict(c,path_id=c['native_id'],accepted_rows=counts[c['native_id']]))
    b.write(ADS/'native_entities.json',dict(entities=entities,unobserved_retained_entities=[c['path_id'] for c in entities if not c['accepted_rows']]))
    b.write(OUT/'source_integrity.json',dict(original_raw_rows_checked=int(is_original.sum()),original_UV_PTS_stable_ids_unchanged=True,source_profiles_read_from_current_native_RGB=True,expected_material_side_from_source_reference=True,entity_from_full_pattern_and_source_continuity=True,current_local_fine_shape_observability_checked=True,small_blurred_text_not_admitted_by_whole_pattern_match=True,increment001_raw_pixels_rechecked=int(restored_extra.sum()),native_geometry_used_for_identity=False,native_geometry_or_parameter_values_changed=False,query_locations_are_not_observations=True,source_support_is_not_calibrated_covariance=True,old_framewide_blur_veto_reused=False,human_visual_acceptance=False))
    shutil.copyfile(Path(applier.__file__),OUT/'apply_increment.py')
    files=[ADS/'point_observations.npz',ADS/'mapping.json',ADS/'initialized_points.npz',ADS/'native_entities.json',OUT/'apply_increment.py',OUT/'delta_operations.jsonl.gz',OUT/'source_integrity.json',*evidence.iterdir()]
    manifest=dict(schema='rinkspace.source_checked_ad_increment.v2',increment_id='s01_ads_checked_increment002_20261026',baseline_id=baseline['baseline_id'],baseline_point_input_sha256=baseline['point_input_sha256'],supersedes_increment001=True,apply_increment001_first=False,operations=dict(ops),ad_rows=len(z['pts']),new_current_RGB_rows=int(np.count_nonzero(np.char.startswith(z['observation_ids'],'s01_native_source_v2_'))),source_scope=dict(first_pts=FIRST,last_pts=LAST,time_base='1/12800',first_source_seconds=22.120,last_source_seconds=44.960,frames=1143),native_parameters_unchanged=True,optimizer_steps=0,human_visual_acceptance=False,files=[dict(path=p.relative_to(OUT).as_posix(),sha256=b.sha(p),bytes=p.stat().st_size) for p in files])
    b.write(OUT/'increment_manifest.json',manifest)
    result=applier.apply(BASE,OUT,COMBINED,BASE/'rebound_numeric_state.npz');print(json.dumps(dict(operations=dict(ops),native_input=result)),flush=True)

def validate():
    z=dict(np.load(COMBINED/'point_observations.npz'));m=b.read(COMBINED/'mapping.json');a=dict(np.load(ADS/'point_observations.npz'));bm=b.read(BASE/'mapping.json');old=dict(np.load(BASE/'point_observations.npz'));oldpaths=np.asarray(bm['path_ids'])[old['path_indices']];keep=(old['pts']>=FIRST)&np.array([not p.startswith('print_instance/Ad') for p in oldpaths]);n=int(keep.sum())
    for k in old:
        if not np.array_equal(z[k][:n],old[k][keep]):raise ValueError('Non-ad input changed in the retained source interval')
    baseline=b.read(b.OUT/'incomplete_baseline_manifest.json')
    for e in baseline['files']:
        if b.sha(b.OUT/e['path'])!=e['sha256']:raise ValueError('Frozen baseline file changed: '+e['path'])
    if len(set(z['observation_ids']))!=len(z['pts']) or len(set(z['source_rows']))!=len(z['pts']):raise ValueError('Duplicate observations')
    state=dict(np.load(COMBINED/'rebound_numeric_state.npz'));oldstate=dict(np.load(BASE/'rebound_numeric_state.npz'))
    for k in oldstate:
        if k not in ['stations','source_rows','point_input_sha256'] and not np.array_equal(oldstate[k],state[k]):raise ValueError('Native global state changed')
    if not np.array_equal(state['stations'][:n],oldstate['stations'][keep]):raise ValueError('Non-ad station changed')
    for pt in np.unique(a['pts']):
        frame=reg.VIS[int(pt)];mask=cv2.imread(frame['mask_path'],0);uv=np.rint(a['uv'][a['pts']==pt]).astype(int)
        if b.sha(frame['mask_path'])!=frame['mask_sha256'] or np.any(mask[uv[:,1],uv[:,0]]):raise ValueError('Foreground exclusion changed or violated')
    guards={}
    for name,base in [('wrong_baseline',pack.ADS),('double_application',COMBINED)]:
        try:applier.apply(base,OUT,OUT/('must_not_exist_'+name))
        except ValueError as e:guards[name]=str(e)
        else:raise ValueError('Import guard failed')
    report=dict(passed=True,rows=len(z['pts']),ad_rows=len(a['pts']),unchanged_nonad_rows=n,source_frames=1143,first_pts=FIRST,last_pts=LAST,baseline_files_unchanged=len(baseline['files']),global_native_parameters_unchanged=True,original_source_identity_unchanged=True,invalid_foreground_points=0,guards=guards,point_input_sha256=b.sha(COMBINED/'point_observations.npz'),optimizer_steps=0,human_visual_acceptance=False)
    b.write(OUT/'validation.json',report);print(json.dumps(report),flush=True)

def render():
    video.BUNDLE=COMBINED;video.REVIEW=OUT/'review';video.render()
    old=video.REVIEW/'S01_replaced_ad_points.mp4';new=video.REVIEW/'S01_ads_checked_increment002.mp4';old.rename(new);doc=b.read(video.REVIEW/'video.json');doc['path']=str(new);b.write(video.REVIEW/'video.json',doc)

if __name__=='__main__':{'prepare':prepare,'validate':validate,'render':render}[sys.argv[1]]()
