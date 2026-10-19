"""Restore every low-installation R/f against one guarded shared-F scene.

The latest physical calibration, native aliases, retained motion pixels,
source person admission, finite material identities and existing source roles
stay fixed.  This writes an explicitly unaccepted component, never an entry.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
from pathlib import Path
from types import SimpleNamespace

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np

from z190_low_nonzero_mount_full_source import prepare as original_prepare, IDENTIFIER
from z190_low_static_foreground_rf_recovery import fixed_mount_rf_solve
from z190_low8_direct_source_rf_recovery import bind_source_finite_board, new_direct_rows, save_conditional_depth_trace
from z190_low5_depth_profile_recovery import bind_core_ice_source
from z190_low_opening_rf_refine import convert_row
from z190_main_zoom_k1_source_information import read, write, ref, CURRENT
from z190_physical_camera import PhysicalCameraModel, CanonicalView
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle
from z190_profiled_motion_schur import SchurMotionKernel
from z190_continuous_source_bundle import SCENE_UNIT_M, SOURCE_SHA, HARD_SCENE_SHA
from z190_source_camera_diagnostic import absolute_value_jac15, source_errors
from continuous_central_brown4_optin import enable_projection_backend
from shared_F_scene_candidate import load_candidate, rebind

CURRENT_SHA = '7625b07752027a857fcc9ea2afa979f2c639152308851958e94bdeee27994e1d'
CANDIDATE_SHA = '427806726d1cae8e5a276bf94acd9c4551296f96cf6987da295b0c5940db01bb'


def fingerprint(row):
    return {key: hashlib.sha256(np.asarray(row[key], np.float64).tobytes()).hexdigest()
            for key in ('uv', 'sigma', 'normal', 'normal_interval') if key in row}


def native_check(row):
    pixels = cv2.imread(str(row['raw_png']))
    if pixels is None or hashlib.sha256(pixels.tobytes()).hexdigest() != row['native_bgr_sha256']:
        raise ValueError('New source native BGR changed')
    if row.get('model_projection_used_to_generate_or_select_uv'):
        raise ValueError('Model-generated source pixels are forbidden')
    expected = row.get('source_uv_float64_sha256')
    if expected and fingerprint(row)['uv'] != expected:
        raise ValueError('Native measured source UV changed')


def cut_base(original, lo, hi, expected):
    stamps = [p for p in original.stamps if lo <= p < hi]
    inside = np.array([lo <= p < hi for p in original.stamps])
    if np.any(inside[original.ia] ^ inside[original.ib]):
        raise ValueError('Original motion crosses a cut')
    active = np.flatnonzero(inside[original.ia] & inside[original.ib])
    if len(active) != expected:
        raise ValueError('Complete original cut motion count changed')
    index = {p:i for i,p in enumerate(stamps)}
    remap = np.array([index.get(p,-1) for p in original.stamps])
    base = SimpleNamespace(stamps=stamps, a=original.a[active], b=original.b[active],
                           ia=remap[original.ia[active]], ib=remap[original.ib[active]])
    return base, active


def absolute_derivative_checks(bundle, q, rows):
    checks = []
    scale = np.array([.01]*3+[1.]*5+[.001]*3+[1.]*4)
    for row in rows:
        if row.get('curve_observation_kind') != 'horizontal_extent' and not row['id'].startswith('ad1:'):
            continue
        p = row['pts']
        cam, sj, vj = bundle.current_calibration(q).camera25_and_jacobians(bundle.view(q,p))
        _, raw = absolute_value_jac15(cam,[row],True)
        analytic = raw @ (np.c_[sj[:15,bundle.mapping_names],vj[:15]]*scale[:,None])
        for axis in range(4):
            col = 4+4*bundle.index[p]+axis
            plus, minus = q.copy(), q.copy(); plus[col] += 1e-7; minus[col] -= 1e-7
            fd = (absolute_value_jac15(bundle.camera(plus,p),[row],False)-
                  absolute_value_jac15(bundle.camera(minus,p),[row],False))/(2e-7)
            error = float(np.linalg.norm(fd-analytic[:,4+axis])/max(1.,np.linalg.norm(fd)))
            checks.append(dict(pts=p,id=row['id'],axis=axis,relative_error=error))
    if max((r['relative_error'] for r in checks),default=0.) > 5e-4:
        raise ValueError('Candidate-scene actual extent/ad source derivative failed')
    return checks


def default_arguments(output):
    fresh=Path('D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00008/RinkSpace_tmp')
    low=fresh/'low5_repair'
    return SimpleNamespace(output=Path(output),current=CURRENT/'z190_full_programme_depth_profile_low8_v1/z190_full_programme_camera_candidate.npz',
        scene=fresh/'three_installation_shared_F_column_keys_v2_guarded/shared_Fx_fixed_height/candidate_scene_concentric_DIAGNOSTIC_ONLY.json',
        F2_source=low/'F2_inner_source_intervals_extent_v3/source_rows.json',Ad1_source=low/'F2_shared_height_compatibility_v1/report.json',
        finite_ice_source=low/'core_ice_motion_source_v2/source_admission_reviewed.json',
        finite_board_source=CURRENT/'low8_mid_source_v1/pair35_finite_Ad4_source_admission_v2.json',
        low8_source=CURRENT/'low8_mid_source_v1/source_external_corners.json',minimum_unknown_depth_m=.01,resume_state=None,export_state=None,solve=False)


def prepare(output,args=None):
    """Return a named complete source bundle without fitting any camera state.

    Joint diagnostics can call prepare(output) to reuse exactly this admitted
    fixed-calibration graph. Candidate scene/source guards remain active.
    """
    args=default_arguments(output) if args is None else args
    out = args.output.resolve()
    if not out.is_relative_to(Path('D:/CodexData').resolve()):
        raise ValueError('Bulk output must stay in CodexData')
    out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1); enable_projection_backend()
    if ref(args.current)['sha256'] != CURRENT_SHA or ref(args.scene)['sha256'] != CANDIDATE_SHA:
        raise ValueError('Current camera or corrected concentric candidate changed')
    adapter = out/'original_source_adapter_inputs'; adapter.mkdir()
    original,_,_,_,native,aliases,_,oldrows,historical = original_prepare(adapter)
    baseline_scene, scene, scene_binding = load_candidate(args.scene,Path(historical['scene']['path']))
    current,_,allaliases,allviews = PhysicalCameraModel.load_npz(args.current)
    cal = current.calibrations[IDENTIFIER]
    views = {p:v for p,v in allviews.items() if v.installation_id == IDENTIFIER}
    stamps = sorted(views)
    if len(stamps) != 1851 or len(native) != 2219 or stamps != list(original.stamps):
        raise ValueError('Full low installation exposure scope changed')
    if {int(p):allaliases[int(p)] for p in native} != aliases:
        raise ValueError('Confirmed official50Hz exposure aliases changed')
    observed = {p:[] for p in stamps}; bound_rows=[]; audits=[]
    def admit(old,role,*,fit=False):
        row = rebind(old,baseline_scene,scene)
        if fingerprint(old) != fingerprint(row) or old['split'] != row['split']:
            raise ValueError('Scene rebinding changed source UV/sigma/normal/role')
        p = aliases[int(row.get('source_native_pts',row['pts']))]
        if p not in observed: raise ValueError('Absolute source outside low installation')
        if fit: observed[p].append(row)
        bound_rows.append((role,old,row,p))
        audits.append(dict(role=role,source_split=old['split'],pts=p,id=row['id'],
                           source_fingerprint=fingerprint(old),old_world_sha256=hashlib.sha256(np.asarray(old['world'],np.float64).tobytes()).hexdigest(),
                           candidate_world_sha256=hashlib.sha256(np.asarray(row['world'],np.float64).tobytes()).hexdigest(),
                           only_F_world_X_shifted=bool(row.get('world_scene_binding',{}).get('shared_complete_F_column_translation'))))
    for role,source in oldrows:
        old = convert_row(source)
        admit(old,role,fit=role=='fit_rows')
    low8rows, low8packet = new_direct_rows(args.low8_source,baseline_scene,aliases)
    native_manifest = read(Path(low8packet['source_manifest']))
    low8pixels = cv2.imread(native_manifest['source_png'])
    if low8pixels is None or hashlib.sha256(low8pixels.tobytes()).hexdigest() != native_manifest['native_bgr_sha256']:
        raise ValueError('Latest low8 direct source native pixels changed')
    for row in low8rows:
        if row['split'] != 'fit' or row['source_native_pts'] != 230907648:
            raise ValueError('Latest low8 direct source role changed')
        admit(row,'latest_low8_direct_source_fit',fit=True)
    F2packet = read(args.F2_source)
    if F2packet['scene_sha256'] != HARD_SCENE_SHA or F2packet['source_sha256'] != SOURCE_SHA:
        raise ValueError('F2 physical source authority changed')
    for role in ('fit_rows','held_rows'):
        for source in F2packet[role]:
            row = convert_row(source); native_check(row)
            ident = row['physical_identity']
            if (ident['kind'],ident['feature_id'],ident['ring_role']) != ('physical_ice_ring_boundary','F2','inner'):
                raise ValueError('Only qualified F2 inner boundary admitted')
            ring = next(r for r in next(f for f in baseline_scene['ice_features'] if f['id']=='F2')['rings'] if r['role']=='inner')
            angles = np.linspace(0.,2*np.pi,181)
            exact = np.asarray(ring['center_xyz'])+ring['radius']*np.c_[np.cos(angles),np.sin(angles),np.zeros(len(angles))]
            if not np.array_equal(exact,row['world']): raise ValueError('F2 source world curve differs from baseline renderer')
            if (role=='fit_rows') != (row['split']=='fit'): raise ValueError('F2 held source role changed')
            # The detailed immutable source ID identifies an observed side;
            # the reviewed physical_identity and exact full world establish
            # the one canonical F2 inner circle for shared geometry binding.
            row['physical_curve_id']='F2_inner'
            row['canonical_physical_geometry_id']='F2_inner'
            row['physical_curve_id_authority']=ref(args.F2_source)
            admit(row,'F2_inner_'+role,fit=role=='fit_rows')
    ads = read(args.Ad1_source)['additional_Ad1_source']
    from render_result import advertisement_boundary_paths
    inventory = {(p['number'],p['role']):np.asarray(p['xyz']) for p in advertisement_boundary_paths(baseline_scene)}
    for source in ads:
        old = convert_row(source); native_check(old); ident=old['physical_identity']
        if old['source_native_pts'] != 230368768 or ident['ad_id'] != 1 or old['split'] != 'compatibility_only_new_source':
            raise ValueError('Only explicitly authorized Ad1 development source permitted')
        if not np.array_equal(old['world'],inventory[1,ident['geometry_role']]):
            raise ValueError('Ad1 real external geometry changed')
        old['original_compatibility_split']=old['split']; old['split']='fit'
        old['development_admission']='parent-authorized native Ad1-only development; F2 same-exposure held remains held'
        admit(old,'new_Ad1_development_fit',fit=True)
    admission_path = CURRENT/'z190_low_fixed_mount_foreground_rf_v1/original_motion_foreground_admission.npz'
    admission_manifest = read(admission_path.parent/'input_manifest.json')['source_motion_admission']
    if ref(admission_path)['sha256'] != admission_manifest['sha256']:
        raise ValueError('Existing source person admission changed')
    with np.load(admission_path,allow_pickle=False) as admission:
        for key,value in [('source_uv_a',original.a),('source_uv_b',original.b),('source_endpoint_index_a',original.ia),
                          ('source_endpoint_index_b',original.ib),('original_sigma_xy',original.motion_sigma.reshape(-1,2))]:
            if not np.array_equal(admission[key],value): raise ValueError('Original motion/person ledger changed:'+key)
        active=admission['active_original_global_index'].copy();keep=admission['static_factor_active'].copy()
    if len(active)!=434991 or int((~keep).sum())!=17: raise ValueError('Existing person admission count changed')
    planes=np.zeros((len(original.a),4));records=[]; finite_refs=[]
    cut5, idx5 = cut_base(original,229945856,230369024,319568)
    motion5=next(r for r in historical['source_motion'] if r['cut']==5)
    p5,r5,f5=bind_core_ice_source(args.finite_ice_source,motion5,cut5,aliases)
    cut8, idx8 = cut_base(original,230898688,231013376,83453)
    motion8=next(r for r in historical['source_motion'] if r['cut']==8)
    p8,r8,f8=bind_source_finite_board(args.finite_board_source,motion8,cut8,baseline_scene,aliases)
    for idx,pp,rr,ff in [(idx5,p5,r5,f5),(idx8,p8,r8,f8)]:
        planes[idx]=pp;finite_refs.append(ff)
        for r in rr:
            item=dict(r);item['original_full_low_bundle_index']=int(idx[r['global_bundle_index']]);records.append(item)
    remap=np.full(len(original.a),-1,int);remap[active]=np.arange(len(active))
    for r in records:
        g=r['original_full_low_bundle_index']
        if not keep[g]:raise ValueError('Qualified finite source conflicts with frozen person admission')
        r['global_bundle_index']=int(remap[g])
    groups=[(p,observed[p]) for p in stamps if observed[p]]
    motion_sigma=original.motion_sigma.reshape(-1,2)[active].ravel()
    absolute_sigma=np.concatenate([np.concatenate([np.repeat(r['sigma'],2) for r in rr]) for _,rr in groups])
    base=SimpleNamespace(stamps=stamps,index=original.index,a=original.a[active],b=original.b[active],
                         ia=original.ia[active],ib=original.ib[active],motion_sigma=motion_sigma,
                         abs_groups=groups,native_residual_sigma=np.r_[motion_sigma,absolute_sigma])
    kernel=SchurMotionKernel(minimum_source_depth_scene_units=args.minimum_unknown_depth_m/SCENE_UNIT_M)
    bundle=MovingPivotSourceBundle(base,cal,kernel,reference_pts=stamps[0],
        initial_reference_optical_height_scene=current.camera25(views[stamps[0]])[2],scene=scene,
        finite_planes=planes[active],finite_records=records)
    q=np.r_[bundle.initial_shared(),np.concatenate([np.r_[views[p].view.rvec_world_to_camera,
          views[p].view.log_f_ratio+np.log(1920.)] for p in stamps])]
    if args.resume_state:
        with np.load(args.resume_state,allow_pickle=False) as saved:
            if not np.array_equal(saved['canonical_pts'],stamps) or not np.array_equal(saved['q'][:4],q[:4]):
                raise ValueError('Resume state changed fixed installation or actual exposure scope')
            q=saved['q'].copy()
    manifest=dict(schema='rinkspace.low-complete-shared-F-scene-fixed-calibration-Rf.v1',current_complete=ref(args.current),
        scene_binding=scene_binding,source_sha256=SOURCE_SHA,original_source_authority=historical['source_authority_rows'],
        original_adapter=ref(adapter/'input_manifest.json'),native_frames=2219,actual_exposures=1851,
        original_motion_rows=435008,active_static_motion_rows=len(active),inactive_source_person_rows=17,
        source_person_admission=admission_manifest,finite_material_rows=len(records),finite_sources=finite_refs,
        new_F2_source=ref(args.F2_source),new_Ad1_source=ref(args.Ad1_source),latest_low8_source=ref(args.low8_source),
        F2_held_retained_even_with_same_exposure_Ad1_fit=True,new_Ad1_source_original_role='compatibility_only_new_source',
        new_Ad1_source_authorization='parent explicitly requested advertisement-only development admission',
        actual_cut_counts={str(i):sum(lo<=p<hi for p in stamps) for i,lo,hi in [(3,229669376,229702144),(5,229945856,230369024),(8,230898688,231013376)]},
        fit_rows=sum(len(rr) for _,rr in groups),only_free='all1851 actual low-installation exposure R3/log-effective-f',
        fixed=['latest pivotXYZ','latest nonzero leverXYZ','PP','k1','all radii/rowY/R/Lane/board/ads/metric','new complete concentric F column X'],
        current_calibration_values=dict(pivot_xyz=cal.pivot_xyz,lever_xyz=cal.lever_xyz,PPx=cal.principal_x_fraction,PPy=cal.principal_y_fraction,k1=cal.k1),
        original_roll_and_single_reference_lens_height_prior_retained=True,unknown_depth_minimum_m=args.minimum_unknown_depth_m,
        unknown_depth_condition_is_engineering_not_measurement=True,source_UV_sigma_existing_roles_unchanged=True,
        original_pending_world_identity_not_promoted=True,native_aliases_bitexact=True,all_low_cuts_recovered=True,
        entrypoint=ref(__file__),scene_guard=ref(Path(__file__).with_name('shared_F_scene_candidate.py')),
        accuracy_accepted=False,scene_accepted=False,video_generated=False)
    write(out/'input_manifest.json',manifest);write(out/'absolute_source_binding_audit.json',audits)
    write(out/'finite_material_source_binding.json',records)
    write(out/'all_source_rows_candidate_binding.json',[dict(role=role,source=row) for role,_,row,_ in bound_rows])
    np.savez_compressed(out/'initial_current_state4rf.npz',q=q,canonical_pts=stamps)
    np.savez_compressed(out/'retained_original_motion_indices.npz',original_full_low_bundle_index=active,source_sigma_xy=motion_sigma.reshape(-1,2))
    exact=MovingPivotSourceBundle(base,cal,SchurMotionKernel(minimum_source_depth_scene_units=args.minimum_unknown_depth_m/SCENE_UNIT_M,schur=False),
        reference_pts=stamps[0],initial_reference_optical_height_scene=current.camera25(views[stamps[0]])[2],scene=scene,
        finite_planes=planes[active],finite_records=records)
    rv,_=bundle.evaluate(q)
    if not np.array_equal(rv,exact.evaluate(q,False)):raise ValueError('Schur source objective replay changed')
    write(out/'source_objective_replay.json',dict(native_objective_bitexact=True,motion_sigma_preserved=True,
        actual_extent_Ad1_derivatives=absolute_derivative_checks(bundle,q,[row for _,_,row,_ in bound_rows]),checks=kernel.latest_checks))
    print(dict(stage='complete_low_candidate_scene_prepared',actual=len(stamps),motion=len(active),finite=len(records),fit_rows=manifest['fit_rows']),flush=True)
    del original,base,exact,cut5,cut8;gc.collect()
    return SimpleNamespace(bundle=bundle,q0=q,current_model=current,current_all_views=allviews,
        native_pts=native,aliases=aliases,current_low_views=views,source_rows=bound_rows,
        manifest=manifest,active_original_motion_indices=active,scene=scene,baseline_scene=baseline_scene,args=args)


def run(args):
    prepared=prepare(args.output,args)
    if not args.solve:return
    bundle,q=prepared.bundle,prepared.q0
    current,views=prepared.current_model,prepared.current_low_views
    native,aliases=prepared.native_pts,prepared.aliases
    bound_rows,manifest=prepared.source_rows,prepared.manifest
    active=prepared.active_original_motion_indices
    out=args.output.resolve();cal=bundle.calibration;stamps=bundle.stamps
    scene_binding=manifest['scene_binding']
    if args.export_state:
        with np.load(args.export_state,allow_pickle=False) as saved:
            if not np.array_equal(saved['canonical_pts'],stamps) or not np.array_equal(saved['q'][:4],q[:4]):
                raise ValueError('Export checkpoint changed fixed installation or exposure scope')
            fitted=saved['q'].copy()
        previous=read(args.export_state.parent/'input_manifest.json')
        for key in ['current_complete','scene_binding','source_person_admission','finite_sources','new_F2_source','new_Ad1_source','latest_low8_source']:
            if manifest[key]!=previous[key]:raise ValueError('Export checkpoint changed source:'+key)
        history=read(args.export_state.parent/'bundle_iterations.json');seconds=history[-1]['elapsed_seconds']
        write(out/'bundle_iterations.json',history)
        manifest['export_checkpoint_without_further_fitting']=ref(args.export_state)
        manifest['export_reason']='Real whole-circle source review rejected this fixed-Fx direction; retain complete accepted state without further cost reduction.'
    else:
        fitted,history,seconds=fixed_mount_rf_solve(bundle,q,out,phases=(True,))
    newviews={p:CanonicalView(IDENTIFIER,bundle.view(fitted,p)) for p in stamps}
    settings=dict(cal.field_settings);settings['shared_F_scene_rf_candidate']=dict(input_manifest=ref(out/'input_manifest.json'),scene_binding=scene_binding,accuracy_accepted=False)
    model=PhysicalCameraModel({IDENTIFIER:replace(cal,field_settings=settings)})
    path=out/'physical_low_complete_shared_F_scene_rf_candidate_unaccepted.npz'
    model.save_npz(path,native,aliases,newviews,source_sha256=SOURCE_SHA,time_base='1/15360')
    loaded,native2,aliases2,views2=PhysicalCameraModel.load_npz(path)
    if not np.array_equal(native2,native) or aliases2!=aliases:raise ValueError('Export changed exposure aliases')
    c2=loaded.calibrations[IDENTIFIER]
    for key in ['pivot_xyz','lever_xyz','principal_x_fraction','principal_y_fraction','k1']:
        if not np.array_equal(getattr(c2,key),getattr(cal,key)):raise ValueError('Export changed fixed calibration:'+key)
    changes=[]
    for role,old,row,p in bound_rows:
        changes.append(dict(source_native_pts=int(old.get('source_native_pts',old['pts'])),canonical_pts=p,id=row['id'],role=role,split=row['split'],
            before=source_errors(current.camera25(views[p]),[old])[0],candidate=source_errors(loaded.camera25(views2[p]),[row])[0]))
    write(out/'all_absolute_source_changes.json',changes)
    np.savez_compressed(out/'optimized_state4rf.npz',q=fitted,canonical_pts=stamps)
    depth=save_conditional_depth_trace(bundle,fitted,active,args.minimum_unknown_depth_m/SCENE_UNIT_M,out)
    write(out/'report.json',dict(manifest,candidate=ref(path),history=history,seconds=seconds,
        all_absolute_source_changes=ref(out/'all_absolute_source_changes.json'),conditional_depth=depth,
        disposition='full_real_source_advertisements_and_Lane_review_pending_not_adopted'))
    print(dict(stage='complete_low_candidate_scene_exported',candidate=str(path),seconds=seconds),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--current',type=Path,required=True);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--F2-source',dest='F2_source',type=Path,required=True)
    p.add_argument('--Ad1-source',dest='Ad1_source',type=Path,required=True)
    p.add_argument('--finite-ice-source',type=Path,required=True)
    p.add_argument('--finite-board-source',type=Path,required=True)
    p.add_argument('--low8-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--minimum-unknown-depth-m',type=float,default=.01)
    p.add_argument('--resume-state',type=Path);p.add_argument('--export-state',type=Path);p.add_argument('--solve',action='store_true')
    run(p.parse_args())
