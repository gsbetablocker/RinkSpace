"""Validate and wholly replace S04; preserve S01 and native calibration authority."""
from pathlib import Path
import collections,gzip,hashlib,io,json,zipfile,shutil
import numpy as np
import prepare_native_joint_shots as p

PACKAGE=Path(r'D:/CodexData/RinkSpace_Data/runs/native_ice_points_s04_20261027/S04_native_points_repaired_20261027.zip')
SOURCE=p.RUN/'source_s04_repaired/points_shape_repaired'
S01_STATE=Path(r'D:/CodexData/RinkSpace_Data/handoff/1027-2/s01_review/run/accepted_pixels_latest.npz')

def audit(b,m,native):
    if set(b)!=p.FIELDS:raise ValueError('Only seven plain point fields are permitted')
    n=len(b['pts'])
    if n!=242583 or any(len(v)!=n for v in b.values()):raise ValueError('Unexpected repaired S04 population')
    if len(set(b['observation_ids']))!=n or len(set(b['source_rows']))!=n:raise ValueError('Duplicate source identity')
    if not np.array_equal(np.unique(b['pts']),np.arange(745984,948992,256)):raise ValueError('S04 exposure coverage differs')
    if not np.isin(b['kind'],[0,1]).all() or not np.isfinite(b['uv']).all() or not np.isfinite(b['stations']).all():raise ValueError('Invalid raw point')
    if (b['uv']<0).any() or (b['uv'][:,0]>=1920).any() or (b['uv'][:,1]>=1080).any():raise ValueError('Raw source pixel outside image')
    if m['scene_revision']!=native.scene_revision or set(m['path_ids'])-native.paths.keys():raise ValueError('Repaired mapping does not match current native')
    if b['path_indices'].min()<0 or b['path_indices'].max()>=len(m['path_ids']):raise ValueError('Invalid native path index')
    lookup={str(k):i for i,k in enumerate(b['observation_ids'])};seen=set();families=collections.Counter();refs={};midpoint_count=0;physical_midpoints=0;max_physical_midpoint_correction=0.
    raw=p.read(p.prep.SCENE);semantics={a['id']:a for a in raw['paths']} if 'paths' in raw else {}
    def collect(v):
        if isinstance(v,dict):
            if 'id' in v and 'measurement' in v:semantics[v['id']]=v
            for x in v.values():collect(x)
        elif isinstance(v,list):
            for x in v:collect(x)
    collect(raw)
    with gzip.open(SOURCE/'point_records.jsonl.gz','rt',encoding='utf-8-sig') as f:
        for line in f:
            r=json.loads(line);i=lookup[r['observation_id']]
            if i in seen:raise ValueError('Repeated source evidence')
            pid=m['path_ids'][b['path_indices'][i]]
            if r['source_row']!=b['source_rows'][i] or r['pts']!=b['pts'][i] or r['kind']!=b['kind'][i] or r['native_path_id']!=pid or not np.array_equal(r['uv'],b['uv'][i]):raise ValueError('Evidence disagrees with supplied raw point')
            seen.add(i);family=r['review_order']['family'];families[family]+=1
            evidence=r['source_evidence'];ref=evidence.get('source_ref',{})
            if ref.get('sha256'):
                if ref['path'] in refs and refs[ref['path']]!=ref['sha256']:raise ValueError('Conflicting evidence digest')
                refs[ref['path']]=ref['sha256']
            if family=='ice':
                measurement=evidence.get('measurement',{})
                sides=measurement.get('actual_source_sides_uv')
                if semantics.get(pid,{}).get('measurement')=='centreline' and sides is not None:
                    # Only export rounding is checked; no measurement tolerance enters fitting.
                    expected=np.mean(sides,axis=0)
                    if measurement.get('width_midpoint_operator')=='inverse_Brown_z0_measured_two_interfaces_ground_average_then_native_Brown_projection':
                        measured=np.asarray(measurement['actual_measured_sides_ground_xy'])
                        if not np.allclose(np.mean(measured,axis=0),measurement['actual_measured_midpoint_ground_xy'],rtol=0,atol=1e-12):raise ValueError('Physical midpoint does not average measured sides')
                        correction=np.asarray(measurement['physical_midpoint_minus_image_arithmetic_midpoint_px'])
                        expected+=correction;physical_midpoints+=1;max_physical_midpoint_correction=max(max_physical_midpoint_correction,float(np.linalg.norm(correction)))
                    if not np.allclose(expected,b['uv'][i],rtol=0,atol=.00011):raise ValueError('Centreline UV differs from its documented two-sided midpoint operator')
                    midpoint_count+=1
                if any(measurement.get(k) is True for k in ['model_projection_observation','optical_flow_observation']):raise ValueError('Non-source ice point')
    if len(seen)!=n:raise ValueError('Incomplete current source records')
    verified=0
    with zipfile.ZipFile(PACKAGE) as archive:
        members=set(archive.namelist())
        for path,sha in refs.items():
            normalized=path.replace('\\','/')
            relative=normalized.split('/native_ice_points_s04_20261027/',1)[-1]
            if relative in members:data=archive.read(relative)
            else:data=Path(path).read_bytes()
            if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Source evidence changed: '+path)
            verified+=1
    withdrawn={r['observation_id'] for r in p.records(p.RUN/'s04_source_repairs.jsonl.gz')}
    if withdrawn&set(lookup):raise ValueError('Previously withdrawn exact source IDs returned')
    old=p.npz(p.RUN/'source_s04/points/point_observations.npz');oldmap=p.read(p.RUN/'source_s04/points/mapping.json')
    oi={str(k):i for i,k in enumerate(old['observation_ids'])};unchanged=rebound=0
    for key,i in lookup.items():
        if key not in oi:continue
        j=oi[key]
        if not np.array_equal(b['uv'][i],old['uv'][j]) or b['pts'][i]!=old['pts'][j]:raise ValueError('An existing source identity was assigned new raw coordinates')
        unchanged+=1
        rebound+=m['path_ids'][b['path_indices'][i]]!=oldmap['path_ids'][old['path_indices'][j]]
    late=[i for i,k in enumerate(b['observation_ids']) if 'Ad4.ANKARA.native440.boundary_' in k and 72.9<=b['pts'][i]/12800<=73.7]
    return dict(points=n,frames=793,families=dict(families),all_rows_match_original_records=True,source_evidence_files_checked=verified,
        checked_centreline_midpoints=midpoint_count,physical_plane_midpoints=physical_midpoints,max_physical_midpoint_correction_px=max_physical_midpoint_correction,unchanged_old_source_rows=unchanged,rebound_old_source_rows=rebound,
        previous_exact_withdrawals_reintroduced=0,old_S04_union_performed=False,only_seven_point_fields=True,
        source_weights_or_sigma_or_correlation_loaded=False,visual_acceptance=False)


def main():
    root=p.RUN;prior=p.read(root/'joint_manifest.json');old_points=p.npz(root/'point_observations.npz');oldmap=p.read(root/'mapping.json')
    a=p.npz(p.BASE/'point_observations.npz');ma=p.read(p.BASE/'mapping.json');b=p.npz(SOURCE/'point_observations.npz');mb=p.read(SOURCE/'mapping.json')
    if p.digest(SOURCE/'point_observations.npz')!='0244ca71e714ae2143ded8353e90f72e1aaf5536900ff7ea582e6c930d0f0c60':raise ValueError('Wrong repaired package')
    if p.digest(p.prep.LIBRARY)!='e4b089ae5cf28ee25311f7a2e6b73f2f9f37c641204de2b45d0756e971f16fc6':raise ValueError('Reviewed solver build changed')
    model=p.SceneParameterModel(p.read(p.prep.SCENE),p.prep.SCENE,optimize_scene=True);s=p.npz(S01_STATE);camera=p.read(p.prep.CAMERA)
    p.prep.validate_snapshot_identity(s,model.initial_state.layout,camera,ma,points=a,point_input_sha256=p.digest(p.BASE/'point_observations.npz'))
    core=p.NativeCore(p.prep.LIBRARY);native=p.NativeSceneGPU(p.prep.SCENE,core,parameter_state=p.ParameterState(model.initial_state.layout,s['scene_values']))
    receipt=audit(b,mb,native);p.write(root/'repaired_package_validation.json',receipt)
    n=len(a['pts'])
    for k in p.FIELDS-{'path_indices'}:np.testing.assert_array_equal(old_points[k][:n],a[k])
    if [oldmap['path_ids'][i] for i in old_points['path_indices'][:n]]!=[ma['path_ids'][i] for i in a['path_indices']]:raise ValueError('Latest S01 mapping changed')
    paths=list(dict.fromkeys(ma['path_ids']+mb['path_ids']));index={v:i for i,v in enumerate(paths)}
    z={k:np.concatenate([a[k],b[k]]) for k in p.FIELDS if k!='path_indices'}
    z['path_indices']=np.r_[[index[ma['path_ids'][i]] for i in a['path_indices']],[index[mb['path_ids'][i]] for i in b['path_indices']]].astype(np.int32)
    mapping=dict(path_ids=paths,scene_revision=native.scene_revision)
    stage=root/'repaired_input';stage.mkdir(exist_ok=True);np.savez(stage/'point_observations.npz',**z);p.write(stage/'mapping.json',mapping)
    m=dict(prior);m['shots']=[dict(q) for q in prior['shots']];m['shots'][1]['row_stop']=len(z['pts'])
    m.update(point_input_sha256=p.digest(stage/'point_observations.npz'),native_library=str(p.prep.LIBRARY),native_library_sha256=p.digest(p.prep.LIBRARY),
        shared_camera_rig_source=str(S01_STATE),shared_camera_rig_source_sha256=p.digest(S01_STATE),repaired_s04_package=str(PACKAGE),repaired_s04_package_sha256=p.digest(PACKAGE),
        current_s04_input=str(SOURCE/'point_observations.npz'),s04_import_audit=receipt,s04_replacement='complete; no old S04 row is appended')
    m['source_inputs']=dict(s01=dict(path=str(p.BASE/'point_observations.npz'),sha256=p.digest(p.BASE/'point_observations.npz')),s04=dict(path=str(SOURCE/'point_observations.npz'),sha256=p.digest(SOURCE/'point_observations.npz')))
    m['joint_output']=str(root/'joint_repaired');m['source_revision_history_manifest']=str(root/'superseded_s04_input/joint_manifest.json')
    for k in ['s04_source_repair','lane_corner_rebindings','corner_rebinding_withdrawal','before_corner_rebind_input_sha256']:
        m.pop(k,None)
    seed=p.npz(root/'source_review_initial.npz')
    if p.digest(p.prep.SCENE)!=prior['scene_file_sha256']:raise ValueError('Native geometry changed; explicit migration required')
    controls=seed['controls'].copy();controls[:len(s['controls'])]=s['controls']
    initial=dict(controls=controls,rig=s['rig'].copy(),scene_values=s['scene_values'].copy(),
        stations=np.r_[s['stations'],p.npz(SOURCE/'initialized_points.npz')['stations']],source_rows=z['source_rows'],point_input_sha256=np.array(m['point_input_sha256']))
    p.bind_joint_snapshot(initial,model.initial_state.layout,mapping,m);p.validate_joint_snapshot(initial,model.initial_state.layout,mapping,m,z)
    np.savez(stage/'initial_pixels.npz',**initial);p.write(stage/'joint_manifest.json',m)
    # Original authorities and every S01 source field remain unchanged.
    for k in p.FIELDS-{'path_indices'}:np.testing.assert_array_equal(z[k][:n],a[k])
    np.testing.assert_array_equal(initial['rig'],s['rig']);np.testing.assert_array_equal(initial['scene_values'],s['scene_values'])
    receipt.update(s01_rows_unchanged=n,S04_rows=len(b['pts']),joint_rows=len(z['pts']),shared_rig_and_scene_from=str(S01_STATE),
        package_numeric_camera_scene_not_loaded=True,new_input_sha256=m['point_input_sha256'])
    p.write(root/'repaired_package_validation.json',receipt)
    archive=root/'superseded_s04_input';archive.mkdir(exist_ok=True)
    for name in ['point_observations.npz','mapping.json','joint_manifest.json','initial_pixels.npz']:
        dest=archive/name
        if dest.exists() and dest.read_bytes()!=(root/name).read_bytes():raise FileExistsError('Previous archive differs; do not overwrite')
        if not dest.exists():shutil.copy2(root/name,dest)
    # Install the checked arrays and publish the manifest last. Old checkpoints
    # retain their old input digest and cannot silently resume on these rows.
    for name in ['point_observations.npz','mapping.json','initial_pixels.npz','joint_manifest.json']:
        pending=root/(name+'.replacement');shutil.copyfile(stage/name,pending);pending.replace(root/name)
    print(json.dumps(receipt,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
