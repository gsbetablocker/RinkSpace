"""Readonly native profile versus current finite-board transfer on saved FIT UV."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RINKSPACE_MOTION_THREADS'):
    os.environ[key] = '1'
root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root), str(root/'src')]
import numpy as np
import cv2
from calibration.solver_runtime import install
install()
from z190_motion_native import native_motion, metadata
from z190_continuous_source_bundle import SCENE_UNIT_M
from rinkspace.vision_camera_initialization import current_complete_source_geometry
from structured_programme_source_candidate.native_board_track_source import current_far_board_points
from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    x = np.asarray(values, float)
    return dict(count=len(x), median=float(np.median(x)), p90=float(np.percentile(x, 90)), maximum=float(x.max()))


def main():
    begin = time.perf_counter()
    base = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
    uvpath = base/'actual_fit_motion_near_new_jumps.npz'
    sigpath = base/'actual_fit_motion_near_new_jumps_original_sigma.npz'
    uv, sig = dict(np.load(uvpath, allow_pickle=False)), dict(np.load(sigpath, allow_pickle=False))
    models = json.loads((base/'camera_probe_contract_readonly.json').read_text(encoding='utf8'))['models']
    pan = {label:json.loads((base/f'pan_{"27fa_review" if label == "27fa" else "c38b_latest"}_20261010.json').read_text(encoding='utf8')) for label in models}
    tests = [('secondary_229649152', 229650688, 229650944, [1100.,1770.,440.,680.]),
             ('low_230315776', 230316800, 230317312, [50.,600.,390.,550.])]
    arrays, records = {}, []
    for label, model in models.items():
        cameras = dict(np.load(model['camera_path'], allow_pickle=False))
        index = {int(p):i for i,p in enumerate(cameras['pts'])}
        provider, world, proof = current_complete_source_geometry(pan[label]['state'], pan[label]['geometry_proof']['auxiliary_path'])
        course, cq, _ = provider.current.print_course_binding(world)
        border = course.sample_perimeter(cq)[0]
        top = float(np.max(np.asarray(json.loads(Path(model['scene_path']).read_text(encoding='utf8'))['board_surface']['top_edge_xyz'])[:,2]))
        for prefix, pa, pb, box in tests:
            for name in ('source_pts', 'target_pts'):
                np.testing.assert_array_equal(uv[prefix+'_'+name], sig[prefix+'_'+name])
            pair = (uv[prefix+'_source_pts'] == pa) & (uv[prefix+'_target_pts'] == pb)
            if sig[prefix+'_finite_mask'][pair].any():
                raise ValueError('This diagnostic pair must retain its authenticated all-unknown-depth role')
            a, b = uv[prefix+'_uv0'][pair], uv[prefix+'_uv1'][pair]
            weights, sigma = uv[prefix+'_weights'][pair], sig[prefix+'_sigma'][pair]
            region = (a[:,0]>=box[0]) & (a[:,0]<=box[1]) & (a[:,1]>=box[2]) & (a[:,1]<=box[3])
            ca, cb = [cameras['camera25'][index[p]].copy() for p in (pa,pb)]
            if (np.any(ca[12:] != 0.) or np.any(cb[12:] != 0.) or ca[10] != 0. or cb[10] != 0.
                    or ca[6] != ca[7] or cb[6] != cb[7] or not np.array_equal(ca[8:12],cb[8:12])):
                raise ValueError('Saved cameras do not match the original native fixed-PP/k1 square-focal kernel')
            (residual, checks, trace) = native_motion(a=a,b=b,fa=np.exp(ca[6]),fb=np.exp(cb[6]),
                ra=cv2.Rodrigues(ca[3:6])[0],rb=cv2.Rodrigues(cb[3:6])[0],
                da=None,db=None,ca=ca[:3],cb=cb[:3],dca_shared=None,dcb_shared=None,
                dca_view=None,dcb_view=None,pp=ca[8:10],k1=ca[11],planes=np.zeros((len(a),4)),
                jacobian=False,sigma=sigma,profile_weight=weights,
                minimum_source_depth_scene_units=1./SCENE_UNIT_M,schur=True,return_trace=True)
            r = residual.reshape(-1,2)
            xyz = current_far_board_points(ca,a[region],border)
            prediction, depth, valid = project_central_valid_full_brown(cb,xyz)
            bounded = valid & (xyz[:,2] >= -1e-6) & (xyz[:,2] <= top+1e-6)
            if not bounded.all():
                raise ValueError('Visually selected board source rays leave the actual current finite board height')
            transfer = prediction-b[region]
            depth_profile = np.divide(1.-trace[:,1],trace[:,1],out=np.full(len(a),np.inf),where=trace[:,1]>0)
            key = label+'_'+prefix
            arrays[key+'_source_uv'] = a[region]; arrays[key+'_target_uv'] = b[region]
            arrays[key+'_global_source_indices'] = sig[prefix+'_global_indices'][pair][region]
            arrays[key+'_sigma'] = sigma[region]; arrays[key+'_frozen_weights'] = weights[region]
            arrays[key+'_unknown_profile_residual_native_px'] = r[region]
            arrays[key+'_finite_board_transfer_residual_native_px'] = transfer
            arrays[key+'_native_profile_trace16'] = trace[region]
            arrays[key+'_conditional_profile_source_Z_working_m'] = depth_profile[region]*SCENE_UNIT_M
            arrays[key+'_conditional_finite_board_xyz'] = xyz
            records.append(dict(model=label,block=prefix.split('_')[0],source_pts=pa,target_pts=pb,
                original_pair_rows=len(a),selected_board_rows=int(region.sum()),source_only_visual_rectangle_xyxy=box,
                original_sigma_minmax=[float(sigma.min()),float(sigma.max())],
                original_all_unknown_depth_roles_preserved=True,
                unknown_profile_all_rows_native_px=stats(np.linalg.norm(r,axis=1)),
                unknown_profile_selected_board_native_px=stats(np.linalg.norm(r[region],axis=1)),
                finite_board_transfer_selected_native_px=stats(np.linalg.norm(transfer,axis=1)),
                unknown_profile_signed_median_xy_px=np.median(r[region],axis=0).tolist(),
                finite_board_signed_median_xy_px=np.median(transfer,axis=0).tolist(),
                selected_profile_rows_converged=int(trace[region,8].sum()),
                selected_profile_rows_boundary=int(trace[region,10].sum()),
                selected_profile_depth_Z_working_m=stats(depth_profile[region]*SCENE_UNIT_M),
                all_selected_source_intersections_inside_current_board_height=True,
                native_checks=checks,actual_current_geometry_proof=proof))
    report = dict(schema='rinkspace.saved-FIT-unknown-depth-versus-finite-board-readonly.v1',
        immutable_original_uv_file=str(uvpath),immutable_original_uv_sha256=sha(uvpath),
        original_sigma_file=str(sigpath),original_sigma_sha256=sha(sigpath),
        native_backend=metadata(),minimum_source_depth_scene_units=1./SCENE_UNIT_M,
        minimum_source_depth_working_m=1.,depth_floor_engineering_assumption_not_measured=True,
        same_saved_qc38_frozen_component_weights_used_for_both_camera_comparisons=True,
        prior_27fa_original_profile_objective_not_replayed=True,
        no_new_FIT_or_role_change=True,visually_selected_regions_are_development_diagnostics_not_unseen_holdouts=True,
        actual_current_board_depth_is_conditional_geometry_not_measured_depth=True,
        full_source_graph_J_LP_or_solver_not_loaded_or_run=True,
        records=records,elapsed_seconds=time.perf_counter()-begin)
    np.savez(base/'unknown_depth_board_absorption_readonly.npz', **arrays)
    (base/'unknown_depth_board_absorption_readonly.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (base/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(records=[{k:r[k] for k in ('model','block','selected_board_rows',
        'unknown_profile_selected_board_native_px','finite_board_transfer_selected_native_px',
        'selected_profile_depth_Z_working_m')} for r in records],elapsed_seconds=report['elapsed_seconds'])))


if __name__ == '__main__':
    main()
