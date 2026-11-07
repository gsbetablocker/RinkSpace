"""Same-clock world motion and source checks for saved action candidates."""
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation

from .absolute_motion_curve import _log
from .blade_contact_modes import blade_axes_diagnostics
from .fullbody_round import write_json
from .fullbody_fit import body_joint_assignment
from .programme_contract import required_clock
from .rebuild_prepare import digest, load
from .trajectory_finalize import distribution


def saved_state(path):
    keys=('pts','time_base','P_world_mid','G_world_mid','Q_world_mid','com_world_mid',
          'P_velocity_world_mps','native22_world_mid','skate_R_world_mid','skate_origin_world_mid',
          'skate_reference_world_mid','COCO_uv','metres_per_scene_unit','shared_body_scale',
          'skate_clearance_m_mid','skate_support_gap_m_mid')
    with np.load(path,allow_pickle=False) as z:
        return {key:z[key] for key in keys}


def candidate_metrics(z, pts, shots, observations, action, camera):
    t=pts/12800.
    Q=z['Q_world_mid']
    P=z['P_world_mid']
    omega=_log(Q[1:]@Q[:-1].swapaxes(-1,-2))/np.diff(t)[:,None]
    ids=body_joint_assignment(observations)
    prediction=z['COCO_uv'][np.arange(len(t))[:,None],ids]
    errors=np.linalg.norm(prediction-observations['uv'],axis=-1)
    errors[~observations['use']]=np.nan
    body_source={name:distribution(errors[:,group]) for name,group in
                 [('torso',[5,6,11,12]),('legs',[13,14,15,16]),('all',list(range(17)))]}
    cuts=[]
    for i in np.flatnonzero(shots[1:]!=shots[:-1]):
        before,after=str(shots[i]),str(shots[i+1])
        cuts.append(dict(from_shot=before,to_shot=after,previous_pts=int(pts[i]),pts=int(pts[i+1]),
            dt_s=float(t[i+1]-t[i]),P_before_world_m=P[i].tolist(),P_after_world_m=P[i+1].tolist(),
            P_step_m=float(np.linalg.norm(P[i+1]-P[i])),
            P_velocities_world_mps=z['P_velocity_world_mps'][[i,i+1]].tolist(),
            Q_before=Q[i].tolist(),Q_after=Q[i+1].tolist(),world_rotation_rate_radps=omega[i].tolist(),
            previous_camera_state_sha256=camera['shots'][before]['state_sha256'],
            camera_state_sha256=camera['shots'][after]['state_sha256']))
    paths={}
    for row in action['episodes']:
        if row['kind']!='rotation_source_path':
            continue
        lo,hi=row['start_pts'],row['end_pts_inclusive']
        within=(pts[:-1]>=lo)&(pts[1:]<=hi)
        raw=omega[within]
        sign=int(row['turn_direction_sign'])
        peak=int(np.flatnonzero(within)[np.argmax(np.linalg.norm(raw,axis=1))])
        paths[row['episode_id']]=dict(interval_pts=[lo,hi],samples=len(raw),
            world_omega_z_radps=distribution(raw[:,2]),world_angular_speed_radps=distribution(np.linalg.norm(raw,axis=1)),
            largest_rate_start_pts=int(pts[peak]),signed_root_yaw_proxy_turns=float(raw[:,2]@np.diff(t)[within]/(2*np.pi)),
            reversals_against_inferred_source_sign=None if not sign else int(np.count_nonzero(sign*raw[:,2]<-.5)),
            source_direction_hypotheses=row['direction_hypotheses'])
    axes=blade_axes_diagnostics(z['skate_R_world_mid'])
    supports={}
    for row in action['episodes']:
        if row['kind']!='support_candidate':
            continue
        mask=(pts>=row['start_pts'])&(pts<=row['end_pts_inclusive'])
        foot=int(row['foot'])
        fixed=z['skate_reference_world_mid'][mask,foot]
        supports[row['episode_id']]=dict(foot=foot,native_exposures=int(mask.sum()),
            source_evidence_status=row['evidence_status'],
            model_fore_aft_pitch_deg=distribution(axes['fore_aft_pitch_degrees'][mask,foot]),
            model_bank_deg=distribution(axes['edge_bank_degrees'][mask,foot]),
            fixed_material_reference_span_XY_m=np.ptp(fixed[:,:2],axis=0).tolist(),
            model_skate_origin_span_XY_m=np.ptp(z['skate_origin_world_mid'][mask,foot,:2],axis=0).tolist())
    return dict(body_native_px=body_source,cut_continuity=cuts,rotation_paths=paths,
        source_gated_support=supports,pelvis_step_m=distribution(np.linalg.norm(np.diff(P,axis=0),axis=1)),
        raw_reference_plane_clearance_m=distribution(z['skate_clearance_m_mid']),
        raw_source_support_gap_m=distribution(z['skate_support_gap_m_mid']))


def compare(project,prepared,ledger,paths,output):
    project,prepared=Path(project),Path(prepared)
    scope=project/'events/ankara_2026/programmes/okada_mei_sp/rebuild_scope.json'
    pts,shots,tb=required_clock(json.loads(scope.read_bytes()))
    obs=load(prepared/'observations.npz')
    np.testing.assert_array_equal(obs['pts'],pts)
    action=json.loads(Path(ledger).read_bytes())
    camera=json.loads((prepared/'camera_context.json').read_bytes())
    base=saved_state(paths[0][1])
    cases={}
    for name,path in paths:
        z=base if name==paths[0][0] else saved_state(path)
        np.testing.assert_array_equal(z['pts'],pts)
        np.testing.assert_array_equal(z['time_base'],tb)
        for k in ('shared_body_scale','metres_per_scene_unit'):
            np.testing.assert_array_equal(z[k],base[k])
        np.testing.assert_array_equal(z['G_world_mid'][:,2],0.)
        metrics=candidate_metrics(z,pts,shots,obs,action,camera)
        metrics.update(path=str(path),sha256=digest(path),
            delta_P_m=distribution(np.linalg.norm(z['P_world_mid']-base['P_world_mid'],axis=-1)),
            delta_G_m=distribution(np.linalg.norm(z['G_world_mid']-base['G_world_mid'],axis=-1)),
            delta_skate_origin_m=[distribution(v) for v in
                np.linalg.norm(z['skate_origin_world_mid']-base['skate_origin_world_mid'],axis=-1).T],
            delta_skate_rotation_deg=[distribution(v) for v in
                np.rad2deg(np.linalg.norm(_log(base['skate_R_world_mid'].swapaxes(-1,-2)@z['skate_R_world_mid']),axis=-1)).T])
        cases[name]=metrics
    report=dict(source_exposures=len(pts),source_pts_first_last=pts[[0,-1]].tolist(),time_base=list(tb),
        source_sha256=action['source_id'],ledger_sha256=digest(ledger),fixed_camera_sha256=digest(prepared/'camera_context.json'),
        one_world_no_alignment=True,candidates=cases,accuracy_accepted=False)
    write_json(output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','prepared','ledger','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--candidate',action='append',required=True,help='ID=result.npz, baseline first')
    a=parser.parse_args()
    paths=[(name,Path(value)) for name,value in (item.split('=',1) for item in a.candidate)]
    compare(a.project,a.prepared,a.ledger,paths,a.output)
    print(a.output,flush=True)


if __name__=='__main__':
    main()
