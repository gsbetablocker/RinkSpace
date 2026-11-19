"""Evaluate registered continuous poses on one source, mass, and shoe population."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from calibration.program_tracking.body_tracking.phase_motion_quality import flight_acceleration_metrics, joint_domain_changes
from calibration.program_tracking.body_tracking.programme_com import ModelCOMMotionFactor
from calibration.program_tracking.body_tracking.scoped_motion_step import skate_projection
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState
from compare import bind, changed_support, distribution, load, ranges, rotation_step, stage_distributions, write
from motion_compare import SOURCE, SHARED, MOTION, read_fields
from source_review import digest, read


def stage(path):
    return Path(path), read_fields(path/'fitted/result.npz',(*SOURCE,*SHARED,*MOTION,'com_world_mid'))


def clock_masks(clock, spans):
    select = np.zeros(len(clock),bool)
    for lo,hi in spans:
        select |= (clock>=lo)&(clock<=hi)
    return select


def phase_flight(ref, physics_times, packet, original_result, windows):
    # Checkpoint-bound absolute curves are the seed for a zero tangent
    # evaluation; the main objective uses the same mass and stencil operators.
    state = SequenceState(ref,packet['seconds'])
    problem = SimpleNamespace(reference=ref,body=SimpleNamespace(state=state),physics_times=physics_times)
    factor = ModelCOMMotionFactor(problem,ContinuousMassGeometry(ref),ref.source_document['com_motion_policy'])
    x = np.zeros(state.size)
    evaluated = factor.evaluate(x,False)
    np.testing.assert_allclose(factor.last_native_acceleration,
        original_result['com_acceleration_world_mps2'][1:-1],rtol=0.,atol=5.e-7)
    episodes = ref.source_document['source_episode_review']['episodes']
    result = {}
    for episode in episodes:
        name = episode['episode_id']
        if name not in factor.flight_rows:
            continue
        record = {}
        for grid,times,pts,a in (
            ('native',ref.times,ref.p['pts'],factor.last_native_acceleration),
            ('dense',factor.times,factor.source_pts_float,factor.last_acceleration)):
            summary,rows = flight_acceleration_metrics(times,pts,a,episode,factor.gravity)
            record[grid] = dict(summary,**rows)
        result[name] = record
    regional = {}
    for name,seconds in windows.items():
        dense_pts = factor.source_pts_float[1:-1]
        keep = clock_masks(dense_pts,[(seconds[0]*12800,seconds[1]*12800)])
        regional[name] = dict(dense_acceleration=distribution(
            np.linalg.norm(factor.last_acceleration[keep],axis=1),dense_pts[keep]))
    result['phase_acceleration'] = regional
    return result, factor.times


def shoe_measurements(ref, camera, controls, study):
    state = SequenceState(ref,controls['seconds'])
    problem = SimpleNamespace(reference=ref,camera=camera,binding=ref.binding,body=SimpleNamespace(state=state))
    source = study['samples']
    uv,_ = skate_projection(problem,np.zeros(17+state.size),source)
    paired = defaultdict(dict)
    for measurement,value in zip(source,uv,strict=True):
        key = (measurement['role'],measurement['phase'],measurement['foot'],measurement['pts'])
        paired[key][measurement['feature']] = (measurement,value)
    table, traces = [], defaultdict(list)
    for (role,phase,foot,pts),points in sorted(paired.items()):
        record = dict(role=role,phase=phase,foot=foot,pts=pts,projected_uv={},source_uv={})
        for feature,(measure,projected) in points.items():
            record['projected_uv'][feature] = projected.tolist()
            record['source_uv'][feature] = measure['uv']
        if 'toe' in points and 'heel' in points:
            features = ['toe','heel']
            actual = np.subtract(*[points[k][0]['uv'] for k in features])
            posed = np.subtract(*[points[k][1] for k in features])
            cosine = (actual@posed)/(np.linalg.norm(actual)*np.linalg.norm(posed))
            record['direction_error_deg'] = float(np.degrees(np.arccos(np.clip(cosine,-1.,1.))))
        else:
            features = ['sole_middle']
            record['direction_error_deg'] = None
        observed = np.mean([points[k][0]['uv'] for k in features],axis=0)
        projected = np.mean([points[k][1] for k in features],axis=0)
        record['centre_error_px'] = float(np.linalg.norm(projected-observed))
        record['centre_radius_px'] = max(points[k][0]['sigma_px'] for k in features)
        traces[(role,phase,foot)].append((pts,observed,projected,tuple(features)))
        table.append(record)
    for (role,phase,foot),seq in traces.items():
        for (a,s0,m0,k0),(b,s1,m1,k1) in zip(seq,seq[1:]):
            # Source displacements require the same visible material and
            # contiguous frame times. They do not imply zero world velocity.
            if b-a==256 and k0==k1:
                row = next(v for v in table if (v['role'],v['phase'],v['foot'],v['pts'])==(role,phase,foot,b))
                row['displacement_error_px'] = float(np.linalg.norm((m1-m0)-(s1-s0)))
                row['source_displacement_px'] = (s1-s0).tolist()
                row['projected_displacement_px'] = (m1-m0).tolist()
    groups = {}
    metrics=('centre_error_px','direction_error_deg','displacement_error_px')
    for role in {r['role'] for r in table}:
        groups[role]={key:distribution(
            np.asarray([r[key] for r in table if r['role']==role and r.get(key) is not None]),
            np.asarray([r['pts'] for r in table if r['role']==role and r.get(key) is not None]))
            for key in metrics}
    for role,phase,foot in traces:
        rows = [r for r in table if (r['role'],r['phase'],r['foot'])==(role,phase,foot)]
        groups[f'{role}/{phase}/foot_{foot}'] = {key:distribution(
            np.asarray([r[key] for r in rows if r.get(key) is not None]),
            np.asarray([r['pts'] for r in rows if r.get(key) is not None]))
            for key in metrics}
    return dict(groups=groups,rows=table)


def saved_surface_comparison(stages, geometry, *, baseline='N'):
    receipt, scan = {}, {}
    for name,path in stages.items():
        archive=path/'retained_surface.json'
        if not archive.is_file():
            parent=Path(json.loads(read(path/'scoped_import.json'))['parent']).parent.parent
            archive=parent/'retained_surface.json'
        row=json.loads(read(archive))
        if digest(row['surface_path'])!=row['surface_sha256']:
            raise ValueError('Saved surface scan does not match its receipt')
        receipt[name]=row
        scan[name]=read_fields(Path(row['surface_path']),('parts','pair_counts','source_seconds',
            'crossing_counts','topology_uncertain_candidates','sample','initial_gap','face_a','face_b'))
    first=scan[baseline]
    old={tuple(row[:3]):int(row[3]) for row in first['pair_counts']}
    result={}
    for name,z in scan.items():
        np.testing.assert_array_equal(first['source_seconds'],z['source_seconds'])
        np.testing.assert_array_equal(first['parts'],z['parts'])
        crossing_parts=np.sort(np.c_[geometry.parts[z['face_a']],geometry.parts[z['face_b']]],axis=1)
        now={tuple(row[:3]):int(row[3]) for row in z['pair_counts']}

        def evidence(scan, part_ids, pairs, keys):
            result=[]
            for i,a,b in sorted(keys):
                rows=(scan['sample']==i)&(part_ids[:,0]==a)&(part_ids[:,1]==b)
                depth=float(-1000*scan['initial_gap'][rows].min()) if rows.any() else None
                result.append(dict(seconds=float(scan['source_seconds'][i]),part=scan['parts'][[a,b]].tolist(),
                                   triangle_pairs=pairs[(i,a,b)],signed_witness_quantity_mm=depth))
            return result

        added=evidence(z,crossing_parts,now,now.keys()-old.keys())
        result[name]=dict(checked_samples=len(z['source_seconds']),triangle_pairs=int(z['crossing_counts'].sum()),
            topology_uncertain_candidates=int(z['topology_uncertain_candidates'].sum()),
            samples_with_more_crossing_pairs=int((z['crossing_counts']>first['crossing_counts']).sum()),
            **{f'added_semantic_pairs_vs_{baseline}':added},refreshed_guard=receipt[name])
        if baseline!='N':
            base_parts=np.sort(np.c_[geometry.parts[first['face_a']],geometry.parts[first['face_b']]],axis=1)
            result[name]['resolved_semantic_pairs']=evidence(first,base_parts,old,old.keys()-now.keys())
    return result


def compare(prepared, source, study_path, stages, output, targets=None, *, fit_evidence_only=False):
    study = json.loads(read(study_path))
    if fit_evidence_only:
        study['samples']=[row for row in study['samples'] if row['role']=='fit']
    windows = dict(full_action=(80.88,82.70),lunge=(80.92,81.10),swing=(81.12,81.34),
                   flight=(81.36,81.48),landing=(81.50,81.60),initial_spin=(81.62,82.46),
                   stable_spin=(81.74,82.46),jump_2a=(77.60,78.56),prior_focus=(81.14,81.62))
    variants = {name:stage(path) for name,path in stages.items()}
    arrays = {name:z for name,(_,z) in variants.items()}
    pts = arrays['N']['source_pts'];seconds=pts/12800.
    for key in (*SHARED,'source_pts','time_base','source_shot_ids','source_uv_native','source_sigma_px',
                'source_body_model_joint_ids','source_support_mode_mid','sequence_contact_modes','sequence_contact_status'):
        for name,z in arrays.items():
            np.testing.assert_array_equal(arrays['N'][key],z[key],err_msg=f'{name}/{key}')
    for key in (*SHARED,*MOTION,'com_world_mid'):
        np.testing.assert_array_equal(arrays['N0'][key],arrays['N'][key],err_msg='Zero update changed '+key)
    if 'M' in arrays:
        for key in SOURCE:
            np.testing.assert_array_equal(arrays['M'][key],arrays['N0'][key],err_msg='Motion comparison changed '+key)
    common = np.logical_and.reduce([z['source_observation_objective_use'] for z in arrays.values()])
    retired = np.argwhere(arrays['N']['source_observation_objective_use'] & ~arrays['N0']['source_observation_objective_use'])
    results = dict(exposures=len(pts),time_base=arrays['N']['time_base'].tolist(),
        same_N_N0_pose=True,common_body_population=int(common.sum()),
        retired_pts_and_slots=np.c_[pts[retired[:,0]],retired[:,1]].tolist(),phase_windows=windows,
        poses={},joint_changes={},shoe={},fit_shoe_evidence_only=fit_evidence_only,
        accuracy_accepted=False,user_accepted=False)
    selections = {key:clock_masks(seconds,[cut]) for key,cut in windows.items()}
    shot = arrays['N']['source_shot_ids']
    selections.update({str(key):shot==key for key in np.unique(shot)})
    anatomy={}
    controls={}
    surface_geometry=None
    points=arrays['N']['source_body_model_joint_ids']
    for name,(run,z) in variants.items():
        geometry,camera,info,doc,packet = bind(prepared,source,run/'solve/candidate.npz')
        controls[name]=packet
        lookup = read_fields(run/'anatomical_frames.npz',('joint_names','normalized_violation','source_pts'))
        np.testing.assert_array_equal(pts,lookup['source_pts'])
        anatomy[name]=lookup
        body_error=np.linalg.norm(z['COCO_uv'][np.arange(len(pts))[:,None],points]-z['source_uv_native'],axis=-1)
        summary = {k:dict(bodies=distribution(body_error[take],pts[take],common[take]),
                          structure=stage_distributions(z,lookup,pts,take)) for k,take in selections.items()}
        results['poses'][name]=summary
        if name=='N0':
            continue
        results['shoe'][name]=shoe_measurements(geometry,camera,packet,study)
        results['poses'][name]['physics'],clock=phase_flight(geometry,packet['physics_times'],packet,z,windows)
        if name=='N':
            results['camera_sha256']=info['state_sha256']
            results['dense_grid_samples']=len(clock)
        if name=='M':
            from calibration.program_tracking.body_tracking.programme_surface import CollisionSurface
            surface_geometry=CollisionSurface(geometry,doc)
        if targets is not None:
            regions = targets['image_regions']
            projections=[]
            for row in regions:
                at=np.searchsorted(pts,row['pts']); lo,hi=np.asarray(row['bounds_uv'])
                point=z['COCO_uv'][at,row['joint']]
                projections.append(dict(pts=row['pts'],joint=row['joint'],uv=point.tolist(),
                    excess_px=float(np.linalg.norm(np.maximum(np.maximum(lo-point,point-hi),0.)))))
            results['poses'][name]['knee_and_ankle_source_regions']=projections
    for name in ('N','M'):
        if name not in anatomy:
            continue
        for window in ('full_action','prior_focus','jump_2a'):
            mask=selections[window]
            results['joint_changes'][f'R_{name}/{window}']=joint_domain_changes(pts[mask],anatomy['R']['joint_names'],
                anatomy['R']['normalized_violation'][mask],anatomy[name]['normalized_violation'][mask])
    if 'M' in arrays:
        np.testing.assert_array_equal(controls['N']['x'],controls['N0']['x'])
        np.testing.assert_array_equal(controls['M']['x'][:17],controls['N0']['x'][:17])
        result = arrays['M']; prior=arrays['N0']
        clock = seconds-seconds[0]
        native = changed_support(controls['N0'],controls['M'],clock)
        jd = np.linalg.norm(result['native22_world_mid']-prior['native22_world_mid'],axis=-1)*1000
        sd = np.linalg.norm(result['skate_origin_world_mid']-prior['skate_origin_world_mid'],axis=-1)*1000
        np.testing.assert_allclose(jd[~native],0.,atol=2.e-7,rtol=0.)
        np.testing.assert_allclose(sd[~native],0.,atol=2.e-7,rtol=0.)
        support = ranges(native,pts)
        edges = np.flatnonzero(np.diff(native.astype(int)))+1
        cuts = np.flatnonzero(shot[:-1]!=shot[1:])+1
        results.update(exact_native_support=support,maximum_joint_motion_mm=float(jd.max()),maximum_shoe_motion_mm=float(sd.max()),
            cuts=[dict(pts=pts[i-1:i+1].tolist(),joint_delta_mm=float(jd[i-1:i+1].max()),
                       shoe_delta_mm=float(sd[i-1:i+1].max())) for i in cuts],
            support_edges=[dict(pts=pts[i-1:i+1].tolist(),
                local_rotation_step_deg={k:float(rotation_step(arrays[k]['native_body_local_R_mid'][i-1:i+1]).max()) for k in ('R','N','M')},
                shoe_step_mm={k:(1000*np.linalg.norm(np.diff(arrays[k]['skate_origin_world_mid'][i-1:i+1],axis=0),axis=-1)[0]).tolist()
                              for k in ('R','N','M')}) for i in edges])
        results['surfaces']=saved_surface_comparison({k:v[0] for k,v in variants.items() if k!='N0'},surface_geometry)
        results['left_knee_step_81_38_deg']={}
        results['root_rotation_step_81_38_deg']={}
        i=np.searchsorted(pts,1041664)
        for name in ('R','N','M'):
            z=arrays[name]
            results['left_knee_step_81_38_deg'][name]=float(rotation_step(z['native_body_local_R_mid'][i-1:i+1])[0,3])
            results['root_rotation_step_81_38_deg'][name]=float(rotation_step(z['Q_world_mid'][i-1:i+1])[0])
    write(output,results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('prepared','source','shoe_study','R','N','N0','output'):
        parser.add_argument(k,type=lambda value:Path(value).resolve())
    parser.add_argument('--M',type=Path)
    parser.add_argument('--targets',type=Path)
    parser.add_argument('--fit-evidence-only',action='store_true')
    a=parser.parse_args()
    runs={k:getattr(a,k) for k in ('R','N','N0','M') if getattr(a,k) is not None}
    compare(a.prepared,a.source,a.shoe_study,runs,a.output,
            None if a.targets is None else json.loads(read(a.targets)),fit_evidence_only=a.fit_evidence_only)


if __name__=='__main__':
    main()
