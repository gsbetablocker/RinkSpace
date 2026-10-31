"""Source-domain and geometry diagnostics for the frozen S01 body ablation."""
from pathlib import Path
import argparse,json
import numpy as np
from .rebuild_prepare import load,digest
from .rebuild_body_fit import FixedNativeProjector
from .fullprogramme_body_sequence import SOURCE_INDICES
from .rebuild_exposure_body import ExposureArticulatedSplineProblem
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation,Slerp

MODES=('fixed','pose','pose_contact')

def stats(v):
    v=np.asarray(v,float).ravel()
    return dict(n=len(v),median=float(np.median(v)),p95=float(np.percentile(v,95)),maximum=float(v.max()),minimum=float(v.min())) if len(v) else dict(n=0)

def audit(root):
    root=Path(root);p=load(root/'input.npz');meta=json.loads((root/'input.json').read_text(encoding='utf8'))
    evidence=json.loads((root/'contact_evidence.json').read_text(encoding='utf8'));camera=FixedNativeProjector(p,meta['native_library'])
    results={k:load(root/k/'result.npz') for k in MODES};visible=~p['lower_body_inferred'];allframes=np.arange(len(p['pts']))
    report=dict(schema='rinkspace.fixed-camera-body-comparison.v1',source_seconds=meta['source_seconds'],frames=len(allframes),
        metric_scale_status='conditional: 60 m per scene length unit; shared-beta stature and equipment geometry are assumptions',
        independent_final_3d_truth=False,source='GVHMR local pose and ViTPose native pixels on confirmed official exposures',
        pose_dofs='nine flexion corrections; shared fixed beta and bone lengths',
        contact_roles='four coarse source regions in fit; two diagnostic regions excluded with +/-100 ms body-pixel windows',
        contact_holdout_caveat='nonpenetration is a general physical prior on visible-body frames, including diagnostic windows; the contact heights are not independent ground truth',
        models={},inputs_unchanged={})
    base=results['fixed']
    for mode,z in results.items():
        error=np.linalg.norm(z['COCO_uv']-p['source_uv'],axis=2)
        selected=np.zeros_like(p['source_qualified']);selected[:,SOURCE_INDICES]=p['source_qualified'][:,SOURCE_INDICES]
        selected&=visible[:,None]
        train=np.zeros(len(allframes),bool);train[z['source_fit_frames']]=True
        rows=[]
        for c in evidence['records']:
            i=int(np.searchsorted(p['pts'],c['pts']));f=c['foot'];k=int(np.argmin(z['blade_world_m'][i,f,:,2]));uv=z['blade_uv'][i,f,k];region=np.asarray(c['region'])
            distance=np.linalg.norm(np.minimum(uv-region[:2],0)+np.maximum(uv-region[2:],0))
            rows.append(dict(pts=c['pts'],source_seconds=c['pts']/12800,physical_foot='left' if f==0 else 'right',role=c['role'],
                lowest_rocker_height_conditional_m=float(z['blade_world_m'][i,f,k,2]),source_region_distance_px=float(distance),
                region_is_precise_blade_point=False,selected_rocker_candidate=k))
        delta=z['G_world_m']-base['G_world_m'];dh=z['P_world_m'][:,2]-base['P_world_m'][:,2]
        view=base['G_world_m'][:,:2]-p['camera_centre'][:,:2]*float(p['metres_per_scene_unit']);view/=np.linalg.norm(view,axis=1)[:,None]
        along=np.sum(delta[:,:2]*view,axis=1);across=delta[:,0]*view[:,1]-delta[:,1]*view[:,0]
        frozen_uv=camera.project(z['COCO_world_m'],allframes)
        rs=np.linalg.norm(z['COCO_uv']-frozen_uv,axis=2)
        # Evaluate the actual articulated body directly at every predicted row
        # to measure the remaining three-node approximation error.
        problem=ExposureArticulatedSplineProblem(root,mode);x=z['x'];t=p['fit_time_seconds'];tau=float(p['rig'][37])
        observed_rows=z['COCO_uv'][:,:,1];times=t[:,None]+tau*(np.clip(observed_rows,0,1079)/1079-.5)
        frame_ids=np.repeat(allframes,17);joint_ids=np.tile(np.arange(17),len(t));flat=times.ravel()
        pose_slerps=[Slerp(t,Rotation.from_rotvec(p['pose_axis_angle'][:,j])) for j in range(21)]
        q_slerp=Slerp(t,Rotation.from_matrix(p['raw_root_prior_Q_world']))
        exact_world=np.empty((len(flat),3))
        for begin in range(0,len(flat),1024):
            stop=min(begin+1024,len(flat));tt=flat[begin:stop];clipped=np.clip(tt,t[0],t[-1]);B=BSpline.design_matrix(tt,z['position_knots'],3,extrapolate=True)
            pose=np.stack([s(clipped).as_rotvec() for s in pose_slerps],axis=1)
            Q=Rotation.from_rotvec(B@x[3*problem.m:6*problem.m].reshape(problem.m,3)).as_matrix()@q_slerp(clipped).as_matrix()
            pose_delta=B@x[6*problem.m:].reshape(problem.m,problem.dof) if problem.articulate else np.zeros((len(tt),problem.dof))
            local=problem.body.forward(pose,pose_delta)[0][np.arange(len(tt)),joint_ids[begin:stop]]
            exact_world[begin:stop]=B@x[:3*problem.m].reshape(problem.m,3)+np.einsum('nij,nj->ni',Q,local)
        instant_rig=p['rig'].copy();instant_rig[37]=0
        rr=camera.core.project_motion(instant_rig,p['controls'],float(p['first_control_time']),float(p['control_dt']),1080,
            p['motion_times'][frame_ids]+flat-t[frame_ids],exact_world/float(p['metres_per_scene_unit']))
        if np.any(rr['status']):raise ValueError('Exact row-time body diagnostic projection failed')
        exact_uv=rr['uv'].reshape(-1,17,2);approx_error=np.linalg.norm(exact_uv-z['COCO_uv'],axis=2)
        _,jp=camera.with_jacobian(z['P_world_m'][:,None],allframes)
        gain=np.linalg.solve(jp[:,0,:,:2],-jp[:,0,:,2,None])[:,:,0];height_gain=np.linalg.norm(gain,axis=1)
        blade_min=z['blade_world_m'][:,:,:,2].min(2)
        report['models'][mode]=dict(
            source_body_reprojection_px=stats(error[selected]),
            fitted_body_reprojection_px=stats(error[selected&train[:,None]]),
            unsampled_native_exposure_reprojection_px=stats(error[selected&~train[:,None]&~z['heldout_frames'][:,None]]),
            diagnostic_window_body_reprojection_px=stats(error[selected&z['heldout_frames'][:,None]]),
            nonfit_elbows_wrists_reprojection_px=stats(error[:,[7,8,9,10]][p['source_qualified'][:,[7,8,9,10]]&visible[:,None]]),
            ground_displacement_vs_fixed_conditional_m=stats(np.linalg.norm(delta,axis=1)[visible]),
            ground_displacement_vs_fixed_scene_L=stats(np.linalg.norm(delta,axis=1)[visible]/float(p['metres_per_scene_unit'])),
            ground_displacement_along_view_signed_conditional_m=stats(along[visible]),
            ground_displacement_across_view_signed_conditional_m=stats(across[visible]),
            height_change_vs_fixed_conditional_m=stats(dh[visible]),
            blade_below_minus_2cm_visible_foot_frames=int(np.count_nonzero(blade_min[visible]<-.02)),
            blade_below_minus_5cm_visible_foot_frames=int(np.count_nonzero(blade_min[visible]<-.05)),
            visible_foot_frames=int(visible.sum()*2),contacts=rows,
            moving_body_vs_frozen_exposure_shift_px=stats(rs[selected]),
            three_node_motion_vs_exact_row_articulation_px=stats(approx_error[selected]),
            horizontal_shift_for_1cm_pelvis_height_change_conditional_m=stats(height_gain[visible]*.01),
            result_sha256=digest(root/mode/'result.npz'))
        np.savez_compressed(root/mode/'exposure_motion_diagnostic.npz',pts=p['pts'],frozen_exposure_uv=frozen_uv,exact_row_time_body_uv=exact_uv,shift_px=rs,approximation_error_px=approx_error)
    for f,sha in meta['frozen_inputs'].items():report['inputs_unchanged'][f]=dict(expected_sha256=sha,current_sha256=digest(f),unchanged=digest(f)==sha)
    report['camera_and_scene_updated_by_this_experiment']=False
    (root/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();audit(a.root)
