"""Replay original saved FIT pairs through four historical camera states."""
from pathlib import Path
import os
import sys
import json
import hashlib
import time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):
    os.environ[key]='1'
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root),str(root/'src')]
import numpy as np
import cv2
from calibration.solver_runtime import install
install()
from z190_motion_native import native_motion
from z190_continuous_source_bundle import SCENE_UNIT_M
from structured_programme_source_candidate.native_board_track_source import current_far_board_points
from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown

BASE=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
RUN=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_full9_review82_speed_candidate_20261010_101450_850')
QS=['5f540331ea6bcfa29287e59d806e260b7eb839d421ce3cc47a639e060f484d92',
    '9a5a2de751e1bb74d0c756b30e935436bc5ee0415166a220e02413da078e6951',
    'a34c29e9dad9a70113717819eb34f174c95d65e1e1d6a16f1022bcef7dbcb641',
    'e2165dc71142efa7436ec20d10cb97b1a7b9bcb69ec373e3f2b2ac9bcca7c739',
    '1a234701c67c5c5b17138ab7364e72bb52babb1789f60e47ec429f2ff714b341',
    '0f71a34689cca82ca692513e9f82b1ba2f5a458ee41fb8621d83993d3c15f651']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stat(v):return dict(n=len(v),median=float(np.median(v)),p90=float(np.percentile(v,90)),maximum=float(np.max(v)))

def main():
    start=time.perf_counter()
    uv=dict(np.load(BASE/'actual_fit_motion_near_new_jumps.npz'))
    sg=dict(np.load(BASE/'actual_fit_motion_near_new_jumps_original_sigma.npz'))
    records=[];states=[];arrays={}
    tests=[('secondary_229649152',229650688,229650944,[1100,1770,440,680]),
           ('low_230315776',230316800,230317312,[50,600,390,550])]
    for qsha in QS:
        folder=RUN/('initial_states' if qsha.startswith('5f540331') else 'accepted_states')/qsha
        identity=json.loads((folder/'state_identity.json').read_text(encoding='utf8'))
        cameras=dict(np.load(identity['camera_path']))
        assert sha(identity['camera_path'])==identity['camera_sha256']
        scene=json.loads((folder/'scene_UNACCEPTED.json').read_text(encoding='utf8'))
        assert sha(folder/'scene_UNACCEPTED.json')==identity['scene_sha256']
        border=np.asarray(scene['board_surface']['bottom_edge_xyz'],float)
        top=np.max(np.asarray(scene['board_surface']['top_edge_xyz'],float)[:,2])
        pts={int(p):i for i,p in enumerate(cameras['pts'])}
        with np.load(folder/'complete_state_UNACCEPTED.npz') as state:
            assert hashlib.sha256(state['q'].tobytes()).hexdigest()==qsha
            damping=float(state['damping'])
            seed=float(state['next_line_search_fraction'])
        truth_path=folder/'truth_restoration_state.json'
        if not truth_path.exists():
            truth_path=RUN.parent/'okada_full9_review82_remaining_fixes_20261010_091826_128'/'accepted_states'/qsha/'truth_restoration_state.json'
        truth=json.loads(truth_path.read_text(encoding='utf8'))
        assert truth['complete_q_sha256']==qsha
        states.append(dict(q_sha256=qsha,state_path=str(folder/'complete_state_UNACCEPTED.npz'),
            camera_path=identity['camera_path'],camera_sha256=identity['camera_sha256'],
            scene_path=str(folder/'scene_UNACCEPTED.json'),damping=damping,next_line_search_fraction=seed,
            truth_path=str(truth_path),truth_q_sha256=truth['complete_q_sha256'],
            truth_keys=list(truth)))
        for prefix,pa,pb,box in tests:
            for name in ('source_pts','target_pts'):
                np.testing.assert_array_equal(uv[prefix+'_'+name],sg[prefix+'_'+name])
            pair=(uv[prefix+'_source_pts']==pa)&(uv[prefix+'_target_pts']==pb)
            assert not sg[prefix+'_finite_mask'][pair].any()
            a,b=uv[prefix+'_uv0'][pair],uv[prefix+'_uv1'][pair]
            sigma,weights=sg[prefix+'_sigma'][pair],uv[prefix+'_weights'][pair]
            region=(a[:,0]>=box[0])&(a[:,0]<=box[1])&(a[:,1]>=box[2])&(a[:,1]<=box[3])
            ca,cb=[cameras['camera25'][pts[p]].copy() for p in (pa,pb)]
            assert np.all(ca[12:]==0) and np.all(cb[12:]==0) and ca[10]==cb[10]==0
            assert ca[6]==ca[7] and cb[6]==cb[7] and np.array_equal(ca[8:12],cb[8:12])
            r,checks,trace=native_motion(a=a,b=b,fa=np.exp(ca[6]),fb=np.exp(cb[6]),
                ra=cv2.Rodrigues(ca[3:6])[0],rb=cv2.Rodrigues(cb[3:6])[0],da=None,db=None,
                ca=ca[:3],cb=cb[:3],dca_shared=None,dcb_shared=None,dca_view=None,dcb_view=None,
                pp=ca[8:10],k1=ca[11],planes=np.zeros((len(a),4)),jacobian=False,
                sigma=sigma,profile_weight=weights,minimum_source_depth_scene_units=1/SCENE_UNIT_M,
                schur=True,return_trace=True)
            r=r.reshape(-1,2)
            xyz=current_far_board_points(ca,a[region],border)
            predicted,depth,valid=project_central_valid_full_brown(cb,xyz)
            assert np.all(valid & (xyz[:,2]>=-1e-6)&(xyz[:,2]<=top+1e-6))
            delta=predicted-b[region]
            name=qsha[:8]+'_'+prefix
            arrays[name+'_source_uv']=a[region];arrays[name+'_target_uv']=b[region]
            arrays[name+'_unknown_px']=r[region];arrays[name+'_finite_board_px']=delta
            arrays[name+'_native_trace16']=trace
            records.append(dict(q_sha256=qsha,block=prefix.split('_')[0],pair=[pa,pb],
                region_xyxy=box,selected_rows=int(region.sum()),original_pair_rows=len(a),
                unknown_all_px=stat(np.linalg.norm(r,axis=1)),
                unknown_selected_px=stat(np.linalg.norm(r[region],axis=1)),
                finite_board_selected_px=stat(np.linalg.norm(delta,axis=1)),
                native_interior_rows=int(trace[:,9].sum()),native_boundary_rows=int(trace[:,10].sum()),
                native_unidentified_rows=int(trace[:,11].sum()),native_checks=checks))
    report=dict(schema='rinkspace.original-FIT-two-historical-uphill-steps-readonly.v1',
        states=states,records=records,exact_original_UV_and_sigma_retained=True,
        same_qc38_saved_frozen_weights_for_all_comparisons=True,prior_states_original_source_objective_not_replayed=True,
        board_geometry_is_same_q_exported_sampled_boundary_not_new_observation=True,
        no_role_source_weight_limit_or_seed_change=True,no_full_graph_or_J_or_solver=True,
        regions_are_development_diagnostics_not_unseen_holdouts=True,seconds=time.perf_counter()-start)
    (BASE/'uphill_steps_motion_localization_readonly.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    np.savez(BASE/'uphill_steps_motion_localization_readonly.npz',**arrays)
    print(json.dumps(dict(states=states,records=records,seconds=report['seconds']),ensure_ascii=False))

if __name__=='__main__':main()
