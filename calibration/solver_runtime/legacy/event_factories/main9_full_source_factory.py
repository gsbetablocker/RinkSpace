"""Prepare main9 complete immutable source bundle without solving/exporting.

Import this factory by absolute filename. It creates source manifests only;
no camera step, solver, scene production write or camera export is performed.
The only conditional scene revision is one shared concentric F-column X.
An optional warm calibration may change main pivot Z only. All source graph,
aliases, native UV, roles, sigmas, finite materials and other optics stay.
"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict,replace
import sys,copy,hashlib
HERE=Path(__file__).resolve().parent
if str(HERE)not in sys.path:sys.path.insert(0,str(HERE))
P=Path('D:/OneDrive/codex_projs/RinkSpace/three_frame_refit/program_tracking');sys.path.insert(0,str(P))
import cv2,numpy as np
from continuous_central_brown4_optin import enable_projection_backend
from z190_main_cut_depth_profile_recovery import prepare,CURRENT,CURRENT_SHA,MANIFEST
from z190_main_zoom_k1_source_information import read,write,ref
from z190_physical_camera import PhysicalCameraModel
from shared_F_scene_candidate import load_candidate,rebind
from main9_typed_source_adapter import Main9TypedFixedCutBundle,typed_absolute_value_jac15
from run_main9_typed_core_source_rf import new_rows,SCENE,PACKETS

def prepare_full_main9_core_source(output,candidate_scene=SCENE,*,calibration_warm=None,view_warm=None,source_admission_cache=None):
    """Return SimpleNamespace(bundle,q,model,native,aliases,views,cal,authority,manifest,additional,original,candidate).

    calibration_warm: optional SharedCalibration for main installation;
      pivot-Z only may differ from immutable 7625 installation.
    view_warm: optional mapping actual PTS -> CanonicalView; only cut9 R/f
      values may differ, its actual keys/installation identity remain exact.
    The fresh current model remains available as original_model for export
      merging. No rejected main9 camera is read implicitly.
    """
    enable_projection_backend();cv2.setNumThreads(1)
    out=Path(output).resolve()
    if not out.is_relative_to(Path('D:/CodexData').resolve()):raise ValueError('Source preparation bulk output must stay in CodexData')
    out.mkdir(parents=True,exist_ok=False);sourceout=out/'original_source_adapter';sourceout.mkdir()
    if ref(CURRENT)['sha256']!=CURRENT_SHA:raise ValueError('Immutable complete 7625 current camera changed')
    fdpath=HERE/'main9_ice_source/typed_y_unique_x_audit/actual_source_C_typed_unique_x_FD_domain.json';fd=read(fdpath)
    if fd['private_adapter_sha256']!=ref(HERE/'main9_typed_source_adapter.py')['sha256']:raise ValueError('Current typed C adapter must have passed real-source FD')
    args=SimpleNamespace(current=CURRENT,cut=9,output=sourceout,minimum_unknown_depth_m=1.,new_ice_source=None,new_ice_motion_source=None,source_admission_cache=source_admission_cache)
    old,oldq,model,native,aliases,views,cal,authority,source_manifest,_=prepare(args)
    original,candidate,binding=load_candidate(Path(candidate_scene),Path(read(MANIFEST)['scene']['path']))
    rows,packetrefs=new_rows(aliases,original,candidate)
    if calibration_warm is None:warm_cal=cal
    else:
        # New-height condition is allowed explicitly; reject silent lever,
        # PP, distortion, pivot XY, focal calibration or provenance changes.
        if asdict(replace(calibration_warm,pivot_xyz=cal.pivot_xyz))!=asdict(cal):raise ValueError('Warm main calibration may change pivot Z only')
        warm_cal=calibration_warm
    warm_views=dict(views)
    if view_warm is not None:
        given=set(map(int,view_warm));expected=set(old.stamps)
        if given!=expected:raise ValueError('Warm view mapping must exactly retain all actual cut9 exposures')
        for p,view in view_warm.items():
            if view.installation_id!=views[p].installation_id:raise ValueError('Warm source view changed installation identity')
            warm_views[p]=view
    observed={p:[rebind(r,original,candidate)for r in rr]for p,rr in old.abs_groups};held={}
    for row in rows:
        p=row['pts']
        if p not in old.index:raise ValueError('New source crosses main9 complete cut')
        (observed if row['split']=='fit'else held).setdefault(p,[]).append(row)
    groups=[(p,observed[p])for p in old.stamps if p in observed];heldgroups=[(p,held[p])for p in old.stamps if p in held]
    sigma=np.concatenate([np.repeat(r['sigma'],2)for _,rr in groups for r in rr])
    base=SimpleNamespace(stamps=old.stamps,index=old.index,a=old.a,b=old.b,ia=old.ia,ib=old.ib,motion_sigma=old.motion_sigma,abs_groups=groups,native_residual_sigma=np.r_[old.motion_sigma,sigma])
    bundle=Main9TypedFixedCutBundle(base,warm_cal,old.motion_kernel,reference_pts=old.reference_pts,
        initial_reference_optical_height_scene=warm_cal.camera25(warm_views[old.reference_pts].view)[2],finite_planes=old.finite_planes,finite_records=old.finite_records,scene=candidate)
    q=np.r_[bundle.initial_shared(),np.concatenate([np.r_[warm_views[p].view.rvec_world_to_camera,warm_views[p].view.log_f_ratio+np.log(1920)]for p in bundle.stamps])]
    manifest=dict(schema='rinkspace.main9.complete-typed-core-source.prepare-only-factory.v1',current=ref(CURRENT),source_adapter=ref(sourceout/'input_manifest.json'),shared_F_scene_binding=binding,
        new_source_packets=packetrefs,source_UV_roles_sigmas_unchanged=True,source_motion_original_rows=source_manifest['original_motion_rows'],source_motion_active_rows=len(bundle.a),finite_source_material_rows=len(bundle.finite_records),actual_exposures=len(bundle.stamps),
        source_exposure_aliases_unchanged=True,new_fit_samples=sum(len(r['uv'])for r in rows if r['split']=='fit'),new_held_samples=sum(len(r['uv'])for r in rows if r['split']=='held'),held_pending_future_not_fit=True,
        C_y_is_only_source_scanline_label=True,C_residual_is_x_support_only=True,private_absolute_dispatch='only C axis_scanline_branch y/unique_x handled here; every other row delegated to current original helper, including clipped_horizontal_extent',
        calibration_warm_requested=calibration_warm is not None,warm_only_pivot_Z_permitted=True,view_warm_requested=view_warm is not None,view_warm_scope='exact cut9 actual R/f only',
        private_adapter=ref(HERE/'main9_typed_source_adapter.py'),typed_C_actual_FD=ref(fdpath),factory_entrypoint=ref(__file__),solve_called=False,camera_exported=False,accuracy_accepted=False,video_generated=False)
    write(out/'input_manifest.json',manifest);write(out/'candidate_scene_world_source_binding.json',binding)
    np.savez_compressed(out/'initial_warm_state_no_fit.npz',q=q,canonical_pts=bundle.stamps)
    model_calibrations=dict(model.calibrations);model_calibrations[warm_cal.installation_id]=warm_cal;warm_model=PhysicalCameraModel(model_calibrations)
    return SimpleNamespace(bundle=bundle,q=q,q0=q,model=warm_model,original_model=model,native=native,aliases=aliases,views=warm_views,original_views=views,cal=warm_cal,calibration=warm_cal,authority=authority,manifest=manifest,input_manifest=manifest,
        additional=rows,additional_held_groups=heldgroups,original=original,candidate=candidate,scene=candidate,output=out)

def records(prepared,q=None,*,new_only=True):
    """Source interval residual audit; does not solve or export."""
    q=prepared.q if q is None else q;bundle=prepared.bundle;groups=bundle.abs_groups+prepared.additional_held_groups;out=[]
    for p,rows in groups:
        for row in rows:
            if new_only and row.get('origin')not in[str(p)for p in PACKETS]:continue
            error=typed_absolute_value_jac15(bundle.camera(q,p),[row],False).reshape(-1,2)*row['sigma'][:,None]
            out.append({'actual_pts':p,'source_native_pts':row.get('source_native_pts',row.get('pts',p)),'id':row['id'],'source_region':row.get('source_region'),'split':row.get('split','original_fit'),
                'outside_support_xy_px':error.tolist(),'median_outside_support_norm_px':float(np.median(np.linalg.norm(error,axis=1))),'max_outside_support_norm_px':float(np.max(np.linalg.norm(error,axis=1)))})
    return out
