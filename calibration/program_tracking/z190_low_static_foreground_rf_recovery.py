"""Recover low-installation R/f with source foreground outside static motion.

Original motion UV/roles/sigma remain in the immutable input ledger.  A row
inside the existing exact-source SAM2 person envelope is not a static-rink
motion observation.  All other untyped material keeps positive-depth nuisance
geometry; it is never promoted to ice or a finite wall by a mask or residual.
The current nonzero tripod/pivot/lever/optics are fixed throughout.
"""
from __future__ import annotations
import argparse,gc,json,time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from z190_low_nonzero_mount_full_source import prepare as original_prepare,source_changes,IDENTIFIER
from z190_main_nonzero_mount_full_source import moving_motion_wrapper,derivative_checks
from z190_main_zoom_k1_source_information import read,write,ref,CURRENT
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle
from z190_physical_camera import PhysicalCameraModel,CanonicalView
from z190_continuous_source_bundle import SOURCE_SHA,SCENE_UNIT_M
from main7_source_bundle_candidate import huber_cost
from render_result import PersonMasks,DEFAULT_MASKS,DEFAULT_PEOPLE
from continuous_central_brown4_optin import enable_projection_backend

CURRENT_FULL=CURRENT/'z190_full_programme_nonzero_mount_source_v3/z190_full_programme_camera_candidate.npz'
CURRENT_SHA='e3ba4157a31b5d2b1d31e2022b46515a1946c5e843abb25b90b0bb290d5f32e7'


def sample_foreground(mask,uv):
    """Sample the existing conservative native person envelope, no projection."""
    uv=np.asarray(uv,float);xy=np.rint(uv).astype(np.int64)
    inside=np.isfinite(uv).all(1)&(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080)
    hits=np.zeros(len(uv),bool);hits[inside]=mask[xy[inside,1],xy[inside,0]]
    return hits


def fixed_mount_rf_solve(bundle,q0,output,*,require_undamped_precision=False,phases=(False,True)):
    q=q0.copy();free=np.arange(4,len(q));n=bundle.source_residual_rows
    history=[];damping=1e-3;start=time.perf_counter()
    quadratic=np.asarray(getattr(bundle,'source_quadratic_mask',np.zeros(n,bool)),bool)
    if quadratic.shape!=(n,):raise ValueError('Source loss inventory changed')
    def cost(r,robust):
        if not quadratic.any():return huber_cost(r[:n],robust)+.5*float(r[n:]@r[n:])
        source=r[:n]
        return huber_cost(source[~quadratic],robust)+.5*float(source[quadratic]@source[quadratic])+ .5*float(r[n:]@r[n:])
    for robust in phases:
        bundle.robust_phase=robust;iteration=0
        if require_undamped_precision:damping=min(damping,1e-3)
        while True:
            if robust:
                preliminary=bundle.evaluate(q,False,include_priors=False)
                bundle.profile_weights=np.minimum(1.,2./np.maximum(np.abs(preliminary[:bundle.motion_rows]),1e-12)).reshape(-1,2)
            r,fullj=bundle.evaluate(q);j=fullj[:,free];before=cost(r,robust)
            weight=np.ones(len(r))
            if robust:
                weight[:n]=np.minimum(1.,2./np.maximum(np.abs(r[:n]),1e-12))
                weight[np.flatnonzero(quadratic)]=1.
            jw=j.multiply(np.sqrt(weight)[:,None]);normal=(jw.T@jw).tocsc();grad=j.T@(weight*r)
            diag=np.maximum(normal.diagonal(),1.);accepted=False;reason='';step=np.zeros(len(free));after=before
            if require_undamped_precision:
                gn_step=spsolve(normal+diags(1e-8*diag),-grad,permc_spec='MMD_AT_PLUS_A')
                gn_mapped=j@gn_step
                gn_source_step=float(np.max(np.abs(gn_mapped[:n]*bundle.native_residual_sigma)))
                gn_prior_step=float(np.max(np.abs(gn_mapped[n:]),initial=0.))
            else:gn_source_step=gn_prior_step=None
            for trial in range(12):
                step=spsolve(normal+diags(damping*diag),-grad,permc_spec='MMD_AT_PLUS_A')
                trialq=q.copy();trialq[free]+=step
                try:
                    if not np.isfinite(trialq).all() or np.any((trialq[7::4]<np.log(100.))|(trialq[7::4]>np.log(50000.))):raise ValueError('Numerical focal domain')
                    after=cost(bundle.evaluate(trialq,False),robust)
                    floor=1e-10*max(1.,abs(before))if require_undamped_precision else 0.
                    if after<before-floor:q=trialq;accepted=True;damping=max(damping*.3,1e-12);break
                    reason='objective_reduction_below_profile_numeric_precision'if after<before else'objective_not_lower'
                except(ValueError,RuntimeError)as e:reason=str(e)[:160]
                damping*=10.
            mapped=j@step;source_step=float(np.max(np.abs(mapped[:n]*bundle.native_residual_sigma)))
            prior_step=float(np.max(np.abs(mapped[n:]),initial=0.))
            rec=dict(phase='huber_source'if robust else'linear_source',iteration=iteration,accepted=accepted,cost=before,next_cost=after if accepted else before,
                damping=damping,trials=trial+1,failure_reason=None if accepted else reason,maximum_linearized_source_step_px=source_step,
                maximum_prior_step_sigma=prior_step,elapsed_seconds=time.perf_counter()-start,fixed_mount_unchanged=bool(np.array_equal(q[:4],q0[:4])))
            rec['maximum_undamped_GN_source_step_px']=gn_source_step;rec['maximum_undamped_GN_prior_step_sigma']=gn_prior_step
            rec['scaled_gradient_maximum']=float(np.max(np.abs(grad)/np.sqrt(diag),initial=0.))
            history.append(rec);write(output/'bundle_iterations.json',history)
            np.savez_compressed(output/'latest_accepted_state.npz',q=q,canonical_pts=bundle.stamps)
            print(json.dumps(dict(stage='foreground_static_rf_iteration',**rec)),flush=True)
            if not accepted:break
            if require_undamped_precision:
                if gn_source_step<.05 and gn_prior_step<.001:break
            elif source_step<.05 and prior_step<.001:break
            iteration+=1
    if not np.array_equal(q[:4],q0[:4]):raise ValueError('Fixed current mechanical installation changed')
    return q,history,time.perf_counter()-start


def prepare(output):
    if ref(CURRENT_FULL)['sha256']!=CURRENT_SHA:raise ValueError('Latest complete camera changed')
    olddir=output/'original_source_adapter_inputs';olddir.mkdir()
    original,_,_,_,native,aliases,_,rows,historical=original_prepare(olddir)
    model,allnative,allaliases,allviews=PhysicalCameraModel.load_npz(CURRENT_FULL)
    cal=model.calibrations[IDENTIFIER];views={p:v for p,v in allviews.items()if v.installation_id==IDENTIFIER}
    if sorted(views)!=list(original.stamps):raise ValueError('Current full low exposure scope differs')
    if {p:allaliases[p]for p in native}!=aliases:raise ValueError('Confirmed low exposure aliases changed')
    masks=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE)
    maskrefs={}
    def maskref(path):
        key=str(path)
        if key not in maskrefs:maskrefs[key]=ref(path)
        return maskrefs[key]
    original_pairs=read(olddir/'retained_original_pair_bindings.json')
    keep=np.ones(len(original.a),bool);source_mask_rows=[];cursor=0
    for k,r in enumerate(original_pairs):
        count=r['retained_row_end_exclusive']-r['retained_row_start'];hi=cursor+count
        pa,pb=r['native_pair_pts'];ma,sa=masks.get(pa,require_sam2=True);mb,sb=masks.get(pb,require_sam2=True)
        ha=sample_foreground(ma,original.a[cursor:hi]);hb=sample_foreground(mb,original.b[cursor:hi])
        keep[cursor:hi]=~(ha|hb)
        source_mask_rows.append(dict(**r,global_retained_start=cursor,global_retained_end_exclusive=hi,source_mask_status_a=sa,source_mask_status_b=sb,
            source_mask_npz_a=maskref(masks.lookup[pa][0]),source_mask_npz_b=maskref(masks.lookup[pb][0]),source_person_envelope_rows_a=int(ha.sum()),source_person_envelope_rows_b=int(hb.sum()),
            static_rows_active=int(keep[cursor:hi].sum()),source_material_review_state='existing_exact_native_SAM2_foreground_envelope_not_static_rink'))
        cursor=hi
        if k%400==0:print(json.dumps(dict(stage='source_foreground_binding',pairs_completed=k+1,original_pairs=len(original_pairs),rows_completed=hi)),flush=True)
    if cursor!=len(original.a):raise ValueError('Original pair/source motion order changed')
    active=np.flatnonzero(keep);inactive=np.flatnonzero(~keep)
    absolute_sig=original.native_residual_sigma[original.motion_rows:]
    motion_sig=original.motion_sigma.reshape(-1,2)[active].ravel()
    base=SimpleNamespace(stamps=original.stamps,index=original.index,a=original.a[active],b=original.b[active],ia=original.ia[active],ib=original.ib[active],
        motion_sigma=motion_sig,abs_groups=original.abs_groups,native_residual_sigma=np.r_[motion_sig,absolute_sig])
    bundle=MovingPivotSourceBundle(base,cal,moving_motion_wrapper,reference_pts=original.stamps[0],
        initial_reference_optical_height_scene=model.camera25(views[original.stamps[0]])[2],scene=read(Path(historical['scene']['path'])))
    q=np.r_[bundle.initial_shared(),np.concatenate([np.r_[views[p].view.rvec_world_to_camera,views[p].view.log_f_ratio+np.log(1920.)]for p in bundle.stamps])]
    # All source-motion weights remain original, not recomputed after admission.
    ledger=output/'original_motion_foreground_admission.npz'
    np.savez_compressed(ledger,original_global_index=np.arange(len(keep)),static_factor_active=keep,active_original_global_index=active,inactive_foreground_original_global_index=inactive,
        source_uv_a=original.a,source_uv_b=original.b,source_endpoint_index_a=original.ia,source_endpoint_index_b=original.ib,original_sigma_xy=original.motion_sigma.reshape(-1,2))
    write(output/'original_pair_source_mask_bindings.json',source_mask_rows)
    np.savez_compressed(output/'initial_current_state4rf.npz',q=q,canonical_pts=bundle.stamps)
    admission=dict(path=str(ledger),sha256=ref(ledger)['sha256'],rule='either endpoint in existing exact-native conservative SAM2 person envelope is outside static rink-motion fit',
        source_only_no_current_projection_or_residual_selection=True,raw_rows_and_source_UV_and_sigma_preserved=True,
        original_rows=len(keep),active_static_conditional_rows=len(active),inactive_source_foreground_rows=len(inactive),
        original_mask_source=ref(DEFAULT_PEOPLE),mask_directory=str(DEFAULT_MASKS),
        conservative_envelope='unchanged existing PersonMasks.get: source SAM2 half-size masks; existing9x9 dilation then native nearest resize',
        inactive_role='unfitted_source_foreground_not_rigid_rink_motion',remaining_material_untyped_positive_depth_nuisance=True,arbitrary_ice_or_wall_assignment=False)
    manifest=dict(schema='rinkspace.current-low-fixed-mount-source-foreground-Rf-recovery.v1',current_complete=ref(CURRENT_FULL),hard_scene=historical['scene'],source_sha256=SOURCE_SHA,
        current_source_absolute=historical['source_authority_rows'],current_source_roles=dict(fit=96,held=7,pending=10),source_motion_admission=admission,
        actual_exposures=len(bundle.stamps),native_frames=len(native),only_free='all1851 actual exposure R3/log-effective-f',
        fixed=dict(pivot_xyz_scene_units=cal.pivot_xyz,lever_xyz_scene_units=cal.lever_xyz,PPx=cal.principal_x_fraction,PPy=cal.principal_y_fraction,k1=cal.k1),
        original_source_roles_and_sigma_unchanged=True,held_pending_not_fit=True,per_frame_free_C=False,current_nonzero_mechanical_model_preserved=True,
        entrypoint=ref(__file__),adapter=ref(Path(__file__).with_name('z190_moving_pivot_source_bundle.py')),motion_kernel=ref(Path(__file__).with_name('z190_moving_center_motion.py')),
        accuracy_accepted=False,video_generated=False)
    write(output/'input_manifest.json',manifest)
    print(json.dumps(dict(stage='current_low_foreground_rf_prepared',original_motion=len(keep),active_motion=len(active),inactive_foreground=len(inactive),actual=len(bundle.stamps))),flush=True)
    del original,base;gc.collect()
    return bundle,q,model,allviews,native,aliases,views,rows,manifest


def run(args):
    output=args.output.resolve()
    if not output.is_relative_to(Path('D:/CodexData').resolve()):raise ValueError('Bulk output must stay in CodexData')
    output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1);enable_projection_backend()
    bundle,q,baseline,oldviews,native,aliases,views,rows,manifest=prepare(output)
    if not args.solve:return
    checks=derivative_checks(bundle,q);write(output/'actual_source_derivative_checks.json',checks)
    print(json.dumps(dict(stage='fixed_mount_source_derivative_ready',maximum_relative=max(r['relative_error']for r in checks))),flush=True)
    print(json.dumps(dict(stage='current_low_fixed_mount_rf_solver_started')),flush=True)
    fitted,history,seconds=fixed_mount_rf_solve(bundle,q,output)
    cal=bundle.current_calibration(fitted)
    settings=dict(cal.field_settings);settings['static_motion_source_admission']=manifest['source_motion_admission']
    settings['source_recovery_binding']=dict(current_complete=manifest['current_complete'],input_manifest=ref(output/'input_manifest.json'),
        fitted='actual_R3_effective_f_only',fixed_current_mount_and_optics=True,accuracy_accepted=False)
    cal=replace(cal,field_settings=settings);model=PhysicalCameraModel({IDENTIFIER:cal})
    newviews={p:CanonicalView(IDENTIFIER,bundle.view(fitted,p))for p in bundle.stamps}
    path=output/'physical_low_fixed_mount_foreground_rf_candidate_unaccepted.npz'
    model.save_npz(path,native,aliases,newviews,source_sha256=SOURCE_SHA,time_base='1/15360')
    loaded,pts2,aliases2,views2=PhysicalCameraModel.load_npz(path)
    if aliases2!=aliases or not np.array_equal(pts2,native):raise ValueError('Current source exposure roles changed')
    for key in('pivot_xyz','lever_xyz','principal_x_fraction','principal_y_fraction','k1','k2','axial'):
        if getattr(cal,key)!=getattr(baseline.calibrations[IDENTIFIER],key):raise ValueError('Frozen calibration changed:'+key)
    changes=source_changes(rows,baseline,oldviews,loaded,views2,aliases);write(output/'all_absolute_source_changes.json',changes)
    np.savez_compressed(output/'optimized_state4rf.npz',q=fitted,canonical_pts=bundle.stamps)
    report=dict(manifest,candidate=ref(path),solve_seconds=seconds,history=history,all_source_changes=ref(output/'all_absolute_source_changes.json'),
        disposition='actual_source_review_pending_not_adopted',accuracy_accepted=False)
    write(output/'report.json',report)
    print(json.dumps(dict(stage='current_low_fixed_mount_rf_exported',candidate=str(path),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=CURRENT/'z190_low_fixed_mount_foreground_rf_v1');p.add_argument('--solve',action='store_true')
    run(p.parse_args())
