"""One full-main Z190 shared k1(log effective f) source diagnostic.

Current v6 alone initializes every real main exposure. Cxyz, saved independent
height, PP and k1 intercept remain fixed. One shared k1 slope and all R/f may
change. Source authority/splits, original ray pairs/sigmas/roll are unchanged.
No automatic acceptance, body update, scene edit, or programme video.
"""
from __future__ import annotations
import argparse,json,time,gc
from pathlib import Path
from dataclasses import replace
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from z190_main_zoom_k1_source_information import prepare,read,write,CURRENT_PHYSICAL,AUTHORITY,HELD,CURRENT,IDENTIFIER,ref
from z190_zoom_radial_source_audit import ZoomRadialBundle,radial_inverse_varying
from z190_continuous_source_bundle import StandingSourceBundle,HARD_SCENE_SHA,SOURCE_SHA,SCENE_UNIT_M
from main7_source_bundle_candidate import Bundle,huber_cost
from z190_physical_camera import PhysicalCameraModel,CanonicalView,ViewState
from z190_source_camera_diagnostic import solve_with_crew_priors,source_errors
from z190_low_opening_rf_refine import convert_row
from camera_hardware_metadata import calibration_context
from continuous_central_brown4_optin import enable_projection_backend

class MainSharedK1SourceBundle(ZoomRadialBundle):
    def __init__(self,base):
        super().__init__(base,'k1_slope')
        self.optimized_local_columns=np.r_[5,np.arange(6,6+4*len(self.stamps))]
    def evaluate(self,q,jacobian=True,include_priors=True):
        # The entire observed image must have the same central inverse branch,
        # even when some fitting source samples do not reach its outer corners.
        logf=q[9::4];f=np.exp(logf);x=logf-np.log(1920.)-self.fixed_calibration.q_reference
        pp=np.asarray([1920*self.fixed_calibration.principal_x_fraction[0],1080*self.fixed_calibration.principal_y_fraction[0]])
        corner=np.array([[0.,0.],[1920.,0.],[0.,1080.],[1920.,1080.]])
        worst=corner[np.argmax(np.sum((corner-pp)**2,axis=1))]-pp
        radial_inverse_varying(worst[None,:]/f[:,None],self.fixed_calibration.k1[0]+q[5]*x)
        return super().evaluate(q,jacobian,include_priors)

def complete_bundle(g):
    stamps=sorted(g.views);pairs=[p for cut in (1,4,7,9)for p in g.rays[cut]]
    data=Bundle(stamps,g.observed,pairs)
    base=StandingSourceBundle(data,g.calibration,prior_wide_focal_px=g.baseline['frozen_wide_prior_focal_px'],
        fixed_height=True,fixed_height_scene_units=g.calibration.pivot_xyz[2])
    bundle=MainSharedK1SourceBundle(base)
    rf=np.concatenate([np.r_[g.views[p].view.rvec_world_to_camera,g.views[p].view.log_f_ratio+np.log(1920.)]for p in stamps])
    q0=np.r_[base.initial_shared(),rf];q=np.r_[np.asarray(g.calibration.pivot_xyz)/100.,0.,0.,0.,rf]
    if not all(np.array_equal(bundle.camera(q,p),g.model.camera25(g.views[p]))for p in stamps):raise ValueError('Zero slope not bitexact current')
    r0=base.evaluate(q0,False,include_priors=False);r=bundle.evaluate(q,False,include_priors=False)
    diff=float(np.max(np.abs(r-r0)))
    if diff>1e-6:raise ValueError('Zero-slope full source objective differs')
    return bundle,q,dict(camera25_zero_slope_bitexact_all_5591=True,full_source_objective_zero_slope_max_sigma=diff,
        motion_points=len(data.a),motion_pair_count=len(pairs),fit_absolute_rows=sum(len(rr)for _,rr in data.abs_groups),
        source_scalar_residual_rows=len(r),roll_prior_rows=len(stamps))

def all_rows(g):
    for role,key in [('fit','fit_rows'),('original_held','held_rows'),('unfitted_identity_pending','pending_rows')]:
        for r in g.packet[key]:yield role,r
    for h in g.packet['untouched_future_groups']:
        for r in h['rows']:
            row=dict(r);row.setdefault('pts',h['pts']);row.setdefault('source_native_pts',h['pts'])
            yield 'untouched_future',row
    for r in g.held:yield 'new_41300_corner_held',r

def source_changes(g,model,views):
    out=[]
    for role,raw in all_rows(g):
        p=g.aliases[int(raw['pts'])];row=convert_row(raw)
        before=g.model.camera25(g.views[p]);after=model.camera25(views[p])
        rec=dict(role=role,pts=p,source_native_pts=raw.get('source_native_pts',raw['pts']),cut=g.cuts[p],id=row['id'],
            original_split=raw.get('split'),origin=raw.get('origin'),physical_ids=raw.get('physical_ids'),
            point_correspondence=bool(raw.get('point_correspondence')),source_uv_float64_sha256=raw.get('source_uv_float64_sha256'),
            before=source_errors(before,[row])[0],after=source_errors(after,[row])[0])
        if rec['point_correspondence']:
            uv=np.asarray(row['uv']);world=np.asarray(row['world'])
            rec['native_signed_before_uv']=((cv2.projectPoints(world,before[3:6],-cv2.Rodrigues(before[3:6])[0]@before[:3],
                np.array([[np.exp(before[6]),0,before[8]],[0,np.exp(before[7]),before[9]],[0,0,1.]]),before[11:25])[0].reshape(-1,2))-uv).tolist()
            rec['native_signed_after_uv']=((cv2.projectPoints(world,after[3:6],-cv2.Rodrigues(after[3:6])[0]@after[:3],
                np.array([[np.exp(after[6]),0,after[8]],[0,np.exp(after[7]),after[9]],[0,0,1.]]),after[11:25])[0].reshape(-1,2))-uv).tolist()
        out.append(rec)
    return out

def source_summary(rows):
    selected=[r for r in rows if r['id'].startswith('R2')or r['role']in ('original_held','untouched_future','new_41300_corner_held')
        or r['id'].startswith('ad2_right')or r['pts']in (229186304,229413632,229416960,229301760)]
    return [dict(role=r['role'],pts=r['pts'],source_native_pts=r['source_native_pts'],cut=r['cut'],id=r['id'],
        physical_ids=r['physical_ids'],before=r['before']['median_px'],after=r['after']['median_px'],
        p95_before=r['before']['p95_px'],p95_after=r['after']['p95_px'])for r in selected]

def render_sources(g,model,views,output):
    """Small source-role/time selected inventory set; no residual-based picks."""
    import hashlib
    from render_result import SceneRenderer,PersonMasks,DEFAULT_MASKS,DEFAULT_PEOPLE,decode_selected
    from review_shared_intrinsic_source_candidate import source_time
    selected={229186560,229189632,229191168,229194240,229413632,229301760,230676480,230699520,230704384,230708224,230715904}
    renderer=SceneRenderer(read(g.scene));masks=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE);folder=output/'source_full_inventory_review';folder.mkdir()
    records=[]
    for p,native in decode_selected(Path('D:/OneDrive/codex_projs/RinkSpace/data/input/oxXu1avhrvs.mp4'),sorted(selected)):
        cp=g.aliases[p]
        try:mask,status=masks.get(p,require_sam2=True)
        except ValueError:mask,status=None,'no confirmed source person mask'
        for label,mm,vv in [('current_v6',g.model,g.views),('shared_k1_slope_diagnostic',model,views)]:
            pic,metrics=renderer.draw(native,mm.camera25(vv[cp]),mask,title=f'{source_time(p)} | Z190 + source | {label} / unaccepted')
            path=folder/f'{p}_{label}_all_model.jpg';cv2.imwrite(str(path),pic,[cv2.IMWRITE_JPEG_QUALITY,94])
            records.append(dict(source_native_pts=p,canonical_pts=cp,source_time=source_time(p),label=label,image=ref(path),
                native_bgr_sha256=hashlib.sha256(native.tobytes()).hexdigest(),mask_status=status,all_inventory_metrics=metrics))
    write(folder/'source_review_manifest.json',dict(images=records,all_modeled_paths_and_advertisements=True,no_confidence_gating=True))
    return records

def run(a):
    a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1);enable_projection_backend()
    print(json.dumps(dict(stage='prepare_current_v6_complete_source')),flush=True);g=prepare(a)
    bundle,q,checks=complete_bundle(g)
    manifest=dict(schema='rinkspace.main-z190-shared-linear-k1-full-source-diagnostic-input.v1',current_physical=ref(a.current),
        current_binding=ref(g.binding_path),source_authority=ref(a.authority),source_authority_admission=g.packet['source_authority_admission'],
        pending_rows_excluded_from_fit=68,new_41300_corner_held=ref(a.held),original_held_future_unchanged=True,
        scene=ref(g.scene),source_sha256=SOURCE_SHA,source_time_base='1/15360',complete_source_checks=checks,
        rays=g.raybindings,sole_initial='current_v6_whole_physical_main_numeric_exact_v5_four_source',frozen_Cxyz=g.calibration.pivot_xyz,
        frozen_independent_height_m=g.calibration.pivot_xyz[2]*SCENE_UNIT_M,frozen_PP_fraction=[g.calibration.principal_x_fraction,g.calibration.principal_y_fraction],
        frozen_k1_intercept=g.calibration.k1[0],q_reference=g.calibration.q_reference,only_shared_active='k1_degree_1',
        per_exposure_active='R3 + log(final_effective_f)',lever_axial_skew_k2_tangential_fixed_zero=True,
        absolute_policy='authority11500fit only; source UV/world/sigma unchanged; no held/pending into fit',
        motion_assumption='all original retained source infinite rays at fixed C, no depth reassignment or source filtering',
        sigma_and_robust_policy='original pair max(2,sqrt(N)/3), original absolute sigma, scalar Huber2 after linear phase',
        prior_policy='original quadratic roll0.5degree; frozen five optics prior rows are constants; no new slope prior',
        image_inverse_policy='same physical central cubic inverse valid at full native image corners for every real exposure',
        information_report=ref(CURRENT/'z190_main_zoom_k1_information_v1/report.json'),entrypoint=ref(__file__),
        solver=ref(Path(__file__).parent/'z190_source_camera_diagnostic.py'),zoom_kernel=ref(Path(__file__).parent/'z190_zoom_radial_source_audit.py'),
        solve_requested=a.solve,accuracy_accepted=False,whole_cut_visual_acceptance=False,production_pointer_changed=False,video_generated=False)
    write(a.output/'input_manifest.json',manifest);np.savez_compressed(a.output/'initial_state6rf.npz',q=q,canonical_pts=bundle.stamps)
    baseline=source_changes(g,g.model,g.views);write(a.output/'all_source_baseline.json',baseline)
    print(json.dumps(dict(stage='full_source_prepared',**checks,saved_height_m=manifest['frozen_independent_height_m'])),flush=True)
    if not a.solve:return
    print(json.dumps(dict(stage='one_full_main_shared_k1_solver_started',exposures=len(bundle.stamps))),flush=True)
    fitted,history,seconds=solve_with_crew_priors(bundle,q,a.output)
    if not np.array_equal(fitted[:5],q[:5]):raise ValueError('Frozen C/PP slope slots changed')
    np.savez_compressed(a.output/'optimized_state6rf.npz',q=fitted,canonical_pts=bundle.stamps)
    calibration=bundle.current_calibration(fitted)
    if calibration.pivot_xyz!=g.calibration.pivot_xyz or calibration.principal_x_fraction[0]!=g.calibration.principal_x_fraction[0]or calibration.principal_y_fraction[0]!=g.calibration.principal_y_fraction[0]or calibration.k1[0]!=g.calibration.k1[0]:
        raise ValueError('Frozen physical calibration changed')
    settings=dict(calibration.field_settings);settings['hardware_context']=calibration_context('main_long_side',source_sha256=SOURCE_SHA)
    settings['source_authority_admission']=dict(**g.packet['source_authority_admission'],unconfirmed_ice_proposals_excluded_from_fit=True)
    settings['zoom_optical_hypothesis']=dict(shared_degree1_k1_only=True,driver='effective logf, no hardware encoder calibration',
        input_manifest=ref(a.output/'input_manifest.json'),current_numeric_initial=ref(a.current),height_frozen_current_independent=True,
        accuracy_accepted=False,diagnostic_only=True)
    calibration=replace(calibration,field_settings=settings);model=PhysicalCameraModel({IDENTIFIER:calibration})
    views={p:CanonicalView(IDENTIFIER,ViewState(tuple(fitted[6+4*i:9+4*i]),float(fitted[9+4*i]-np.log(1920.))))for i,p in enumerate(bundle.stamps)}
    path=a.output/'physical_main_shared_zoom_k1_diagnostic_unaccepted.npz'
    model.save_npz(path,g.native,g.aliases,views,source_sha256=SOURCE_SHA,time_base='1/15360')
    saved,native,aliases,loaded=PhysicalCameraModel.load_npz(path)
    if aliases!=g.aliases or not np.array_equal(native,g.native)or not all(np.array_equal(saved.camera25(loaded[p]),bundle.camera(fitted,p))for p in loaded):
        raise ValueError('Polynomial physical/alias replay failed')
    saved.save_calibration_json(a.output/'calibration_shared_k1_diagnostic.json')
    changes=source_changes(g,saved,loaded);write(a.output/'all_absolute_source_changes.json',changes)
    summary=source_summary(changes);write(a.output/'source_summary.json',summary)
    r=bundle.evaluate(fitted,False);n=bundle.source_residual_rows
    report=dict(manifest,physical_diagnostic=ref(path),k1_coefficients=calibration.k1,solve_seconds=seconds,
        canonical_exposures=len(loaded),native_frames=len(native),source_aliases_bitexact=True,frozen_installation_C_PP_k1_intercept_bitexact=True,
        all_actual_Rf_initialized_current=True,all_source_changes=ref(a.output/'all_absolute_source_changes.json'),
        source_summary=ref(a.output/'source_summary.json'),history=history,final_source_Huber_cost=huber_cost(r[:n],True),
        final_quadratic_prior_cost=.5*float(r[n:]@r[n:]),disposition='source_review_pending_not_adopted',body_ranges_updated=False)
    write(a.output/'report.json',report)
    print(json.dumps(dict(stage='solver_and_physical_export_complete',k1=calibration.k1,seconds=seconds,physical=str(path))),flush=True)
    if a.render:
        report['source_images']=render_sources(g,saved,loaded,a.output);write(a.output/'report.json',report)
    print(json.dumps(dict(stage='diagnostic_complete',physical=str(path),summary=str(a.output/'source_summary.json'))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--current',type=Path,default=CURRENT_PHYSICAL)
    p.add_argument('--authority',type=Path,default=AUTHORITY);p.add_argument('--held',type=Path,default=HELD)
    p.add_argument('--output',type=Path,default=CURRENT/'z190_main_shared_zoom_k1_full_source_v1')
    p.add_argument('--solve',action='store_true');p.add_argument('--render',action='store_true');run(p.parse_args())
