"""Install explicit source-reviewed shot operations, preserving native state."""
from pathlib import Path
import argparse
import shutil
import numpy as np
import prepare_native_joint_shots as p
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.session.gpu_geometry import initialize_curve_stations


def seed_reviewed_stations(native,state,points,mapping,manifest,rows):
    """Seed changed identities on their named native curves, at frozen globals."""
    rows=np.asarray(rows,dtype=int)
    if not len(rows):return dict(rows=0,invalid=0)
    timeline=ShotMotionTimeline(manifest['shots'],manifest['control_dt'])
    camera={**p.read(p.prep.CAMERA),'rig':state['rig'].tolist(),
            'first_control_time':timeline.first,'control_dt':timeline.dt}
    names=[mapping['path_ids'][int(points['path_indices'][i])] for i in rows]
    times=timeline.row_times(points['pts'])[rows]
    stations,report=initialize_curve_stations(native,camera=camera,controls=state['controls'],
        path_ids=names,stations=state['stations'][rows],pts=points['pts'][rows],
        observed=points['uv'][rows],kinds=points['kind'][rows],motion_times=times,
        half_seconds=manifest['max_readout']/2)
    if report['unresolved']:raise ValueError('Reviewed native curve has no valid projected support: '+str(report))
    with native.resident(camera=camera,controls=state['controls'],path_ids=names,stations=stations,
            pts=points['pts'][rows],observed=points['uv'][rows],motion_times=times) as resident:
        resident.configure_projection_time_support(half_seconds=manifest['max_readout']/2)
        evaluation=resident.evaluate('cuda')
    if np.any(evaluation['status']):raise ValueError('Reviewed station initialization has invalid projections')
    state['stations'][rows]=stations
    return dict(rows=len(rows),invalid=0,**report)


def main(review_path,out,snapshot=None):
    root=p.RUN;review=p.read(review_path);manifest=p.read(root/'joint_manifest.json')
    layer=review.get('source_layer','ice')
    assert layer in ('ice','advertisement')
    instance=review.get('instance_id')
    shot_id=review.get('shot_id','s04');allowed_targets=set(review.get('target_layers',[layer]))
    assert allowed_targets <= {'ice','board','advertisement'}
    if layer=='advertisement':assert isinstance(instance,str) and instance.startswith('print_instance/')
    mapping=p.read(root/'mapping.json');original=p.npz(root/'point_observations.npz')
    assert set(original)=={'kind','observation_ids','path_indices','pts','source_rows','stations','uv'}
    assert p.digest(root/'point_observations.npz')==manifest['point_input_sha256']
    snapshot_path=snapshot or root/'joint_repaired/accepted_pixels_latest.npz'
    state=p.npz(snapshot_path)
    model=p.SceneParameterModel(p.read(p.prep.SCENE),p.prep.SCENE,optimize_scene=True)
    p.validate_joint_snapshot(state,model.initial_state.layout,mapping,manifest,original)
    native=p.NativeSceneGPU(p.prep.SCENE,p.NativeCore(manifest['native_library']),parameter_state=p.ParameterState(model.initial_state.layout,state['scene_values']))
    shot=next(s for s in manifest['shots'] if s['id']==shot_id)
    original_bounds={s['id']:(s['row_begin'],s['row_stop']) for s in manifest['shots']}
    begin,stop=original_bounds[shot_id]
    untouched=np.r_[np.arange(begin),np.arange(stop,len(original['pts']))]
    points={k:v.copy() for k,v in original.items()};lookup={str(v):i for i,v in enumerate(points['observation_ids'])}
    keep=np.ones(len(points['pts']),bool);rebound=[];seen=set();checked=set()
    for op in review['operations']:
        key=op['observation_id'];assert key not in seen;seen.add(key);i=lookup[key]
        assert begin<=i<stop and int(points['pts'][i])==op['pts']
        assert np.array_equal(points['uv'][i],op['uv'])
        assert mapping['path_ids'][int(points['path_indices'][i])]==op['before']
        assert native.paths[op['before']]['layer']==layer
        if instance:assert op['before'].startswith(instance+'::')
        evidence=(op['source_image'],op['source_image_sha256'])
        if evidence not in checked:assert p.digest(evidence[0])==evidence[1];checked.add(evidence)
        if op['action']=='exclude':keep[i]=False
        elif op['action']=='rebind':
            assert op['after'] in native.paths and native.paths[op['after']]['layer'] in allowed_targets
            if instance and native.paths[op['after']]['layer']==layer:assert op['after'].startswith(instance+'::')
            points['path_indices'][i]=mapping['path_ids'].index(op['after']);rebound.append(i)
        else:raise ValueError('Unknown reviewed operation')
    assert int((~keep).sum())==review['excluded_rows'] and len(rebound)==review['rebound_rows']
    assert np.array_equal(np.flatnonzero(points['path_indices']!=original['path_indices']),np.sort(rebound))
    for key in points:
        if key!='path_indices':np.testing.assert_array_equal(points[key],original[key])
        np.testing.assert_array_equal(points[key][untouched],original[key][untouched])
        points[key]=points[key][keep]
    assert np.array_equal(np.unique(points['pts']),np.unique(original['pts']))
    out.mkdir(parents=True,exist_ok=False);np.savez(out/'point_observations.npz',**points)
    old_hash=manifest['point_input_sha256'];manifest['point_input_sha256']=p.digest(out/'point_observations.npz')
    cursor=0
    for s in manifest['shots']:
        a,b=original_bounds[s['id']];count=int(keep[a:b].sum());s['row_begin']=cursor;s['row_stop']=cursor+count;cursor+=count
    s04=next(s for s in manifest['shots'] if s['id']=='s04');s01_stop=s04['row_begin']
    if 'current_s04_input_rows' in manifest:manifest['current_s04_input_rows']=[s04['row_begin'],s04['row_stop']]
    manifest['source_identity_repairs']=[*manifest.get('source_identity_repairs',[]),dict(review=str(review_path),review_sha256=p.digest(review_path),excluded_rows=int((~keep).sum()),rebound_rows=len(rebound),old_input_sha256=old_hash)]
    fixed={k:v.copy() for k,v in state.items()};fixed['stations']=fixed['stations'][keep]
    changed_ids={op['observation_id'] for op in review['operations'] if op['action']=='rebind'}
    changed_rows=[i for i,key in enumerate(points['observation_ids']) if str(key) in changed_ids]
    station_initialization=seed_reviewed_stations(native,fixed,points,mapping,manifest,changed_rows)
    fixed['source_rows']=points['source_rows'];fixed['point_input_sha256']=np.array(manifest['point_input_sha256'])
    p.bind_joint_snapshot(fixed,model.initial_state.layout,mapping,manifest)
    p.validate_joint_snapshot(fixed,model.initial_state.layout,mapping,manifest,points)
    for key in ('rig','controls','scene_values'):np.testing.assert_array_equal(fixed[key],state[key])
    np.savez(out/'initial_pixels.npz',**fixed);p.write(out/'joint_manifest.json',manifest)
    for name in ('point_observations.npz','initial_pixels.npz','joint_manifest.json'):shutil.copy2(root/name,out/('before_'+name))
    shutil.copy2(snapshot_path,out/'before_pixels.npz');shutil.copy2(review_path,out/'source_review.json')
    for name in ('point_observations.npz','initial_pixels.npz','joint_manifest.json'):
        tmp=root/(name+'.pending');shutil.copy2(out/name,tmp);tmp.replace(root/name)
    receipt=dict(rows=len(points['pts']),s01_rows=s01_stop,s04_rows=len(points['pts'])-s01_stop,
        excluded_rows=int((~keep).sum()),rebound_rows=len(rebound),all_retained_uv_pts_ids_unchanged=True,
        reviewed_shot=shot_id,
        s01_all_fields_unchanged=all(np.array_equal(points[k][:s01_stop],original[k][slice(*original_bounds['s01'])]) for k in points),
        all_other_rows_unchanged=True,rig_scene_controls_unchanged=True,
        all_source_frames_preserved=True,point_weights_or_error_ranges_added=False,
        station_initialization=station_initialization,
        numeric_seed=str(snapshot_path),numeric_seed_sha256=p.digest(out/'before_pixels.npz'),
        before_input_sha256=old_hash,new_input_sha256=manifest['point_input_sha256'])
    p.write(out/'replacement_receipt.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--review',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,help='Explicit current-input snapshot when preparing consecutive source repairs without an intervening fit')
    args=parser.parse_args();main(args.review,args.output,args.snapshot)
