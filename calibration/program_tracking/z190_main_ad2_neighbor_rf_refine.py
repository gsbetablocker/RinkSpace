"""Prepare/recover complete main cut1 R/f with frozen completed calibration.

Uses the current saved 1.622907m main height, not a reset to 1.6m. Other main
cuts retain their original warm CanonicalView objects exactly. No video or
production pointer is written. New corners remain transparent development data.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from z190_main_source_objective_audit import load_current_graph,DEFAULT_CURRENT,CURRENT,ref,jsonable,IDENTIFIER
from z190_main_ad2_single_source_check import candidate_rows
from z190_continuous_source_bundle import StandingSourceBundle,HARD_SCENE_SHA,SOURCE_SHA,SCENE_UNIT_M
from main7_source_bundle_candidate import Bundle
from z190_low_opening_rf_refine import solve_frozen_pivot
from z190_source_camera_diagnostic import source_errors,future_groups
from z190_physical_camera import PhysicalCameraModel,CanonicalView,ViewState
from continuous_central_brown4_optin import enable_projection_backend
from camera_hardware_metadata import calibration_context


def prepare_cut(a):
    if a.current.resolve()!=DEFAULT_CURRENT.resolve():raise ValueError('This entry accepts only the explicitly specified completed main warm')
    g=load_current_graph(a.current);added=candidate_rows(a.new_source,g)
    for path in getattr(a,'additional_source',[]):
        added.extend(candidate_rows(path,g))
    stamps=[p for p in g.bundle.stamps if g.cuts[p]==1];selected=set(stamps)
    observations={p:list(rows) for p,rows in g.bundle.abs_groups if p in selected}
    observations.update({p:[] for p in stamps if p not in observations})
    for row in added:
        identities={x for r in observations[row['pts']] for x in r.get('physical_ids',[])}
        if set(row['physical_ids']) & identities:raise ValueError('New corner duplicates an already fitted physical corner in this exposure')
        observations[row['pts']].append(row)
    all_pair_cuts=np.array([g.cuts[p] for p in g.bundle.stamps])
    if np.any(all_pair_cuts[g.bundle.ia]!=all_pair_cuts[g.bundle.ib]):raise ValueError('Existing objective contains a cross-cut edge; cut decomposition invalid')
    binding=next(b for b in g.bindings if b.get('cut')==1 and b.get('role')=='within_cut_observed_source_ray_pairs')
    z=np.load(binding['path'],allow_pickle=False);pairs=[]
    for i,(pa,pb) in enumerate(z['pair_pts']):
        pa,pb=int(pa),int(pb)
        if pa not in selected or pb not in selected or pa==pb:raise ValueError('Original main1 pair identity changed')
        lo,hi=map(int,z['offsets'][i:i+2]);pairs.append((pa,pb,z['source_uv'][lo:hi],z['target_uv'][lo:hi]))
    base=Bundle(stamps,observations,pairs)
    old_mask=all_pair_cuts[g.bundle.ia]==1
    if (not np.array_equal(base.a,g.bundle.a[old_mask]) or not np.array_equal(base.b,g.bundle.b[old_mask])
            or not np.array_equal(base.motion_sigma,g.bundle.motion_sigma[np.repeat(old_mask,2)])):
        raise ValueError('Full cut1 original motion pixels/sigmas changed')
    bundle=StandingSourceBundle(base,g.calibration,prior_wide_focal_px=g.manifest['frozen_wide_prior_focal_px'],
        fixed_height=True,fixed_height_scene_units=g.calibration.pivot_xyz[2])
    q0=np.r_[g.q[:6],np.concatenate([g.q[6+4*g.bundle.index[p]:10+4*g.bundle.index[p]] for p in stamps])]
    for p in stamps:
        if not np.array_equal(bundle.camera(q0,p),g.model.camera25(g.views[p])):raise ValueError('Cut1 warm camera differs from completed state')
    future,fb=future_groups(g.base,g.scene,g.previous/'all_main_candidate_cameras.npz')
    future=[h for h in future if int(h['cut'])==1]
    if any(g.aliases[int(h['pts'])] in {r['pts'] for r in added} for h in future):raise ValueError('New source exposure was untouched future evidence')
    held=[r for r in g.rows if g.cuts[g.aliases[int(r['pts'])]]==1 and r['split']!='fit']
    return g,bundle,q0,added,held,future,fb


def derivative_replay(bundle,q,calibration,p):
    idx=bundle.index[p];keep=(bundle.ia==idx)|(bundle.ib==idx);ss=np.repeat(keep,2)
    rows=dict(bundle.abs_groups)[p]
    sigma=np.r_[bundle.motion_sigma[ss],np.concatenate([np.repeat(r['sigma'],2) for r in rows])]
    base=SimpleNamespace(stamps=bundle.stamps,index=bundle.index,a=bundle.a[keep],b=bundle.b[keep],
        ia=bundle.ia[keep],ib=bundle.ib[keep],motion_sigma=bundle.motion_sigma[ss],abs_groups=[(p,rows)],native_residual_sigma=sigma)
    local=StandingSourceBundle(base,calibration,prior_wide_focal_px=bundle.prior_wide_focal_px,
        fixed_height=True,fixed_height_scene_units=calibration.pivot_xyz[2])
    r,j=local.evaluate(q,include_priors=False);checks=[]
    for c in 6+4*idx+np.arange(4):
        h=1e-7;plus,minus=q.copy(),q.copy();plus[c]+=h;minus[c]-=h
        numeric=(local.evaluate(plus,False,include_priors=False)-local.evaluate(minus,False,include_priors=False))/(2*h)
        analytic=j[:,c].toarray().ravel();checks.append(dict(column=int(c),relative_l2=float(np.linalg.norm(numeric-analytic)/max(1.,np.linalg.norm(numeric))),maximum_absolute=float(np.max(np.abs(numeric-analytic)))))
    if max(c['relative_l2'] for c in checks)>2e-4:raise ValueError('Actual new-source+incident motion derivative replay failed')
    return dict(canonical_pts=p,actual_corner_and_original_curve_rows=sum(len(r['uv']) for r in rows),
        original_incident_motion_points=len(local.a),checks=checks)


def source_comparison(g,bundle,q,added,held,future):
    fitted={p:bundle.camera(q,p) for p in bundle.stamps}
    checks=[];new_row_objects={id(r) for r in added}
    for p,rows in bundle.abs_groups:
        for row in rows:
            checks.append(dict(canonical_pts=p,source_native_pts=row.get('source_native_pts',row['pts']),id=row['id'],split=row['split'],
                new_source=id(row) in new_row_objects,physical_ids=row.get('physical_ids'),origin=row['origin'],
                before=source_errors(g.model.camera25(g.views[p]),[row])[0],after=source_errors(fitted[p],[row])[0]))
    withheld=[]
    for role,groups in [('original_held',[(int(r['pts']),[r]) for r in held]),('untouched_future',[(int(h['pts']),h['rows']) for h in future])]:
        for native,rows in groups:
            p=g.aliases[native]
            for row in rows:
                withheld.append(dict(source_native_pts=native,canonical_pts=p,role=role,id=row['id'],split=row['split'],
                    before=source_errors(g.model.camera25(g.views[p]),[row])[0],after=source_errors(fitted[p],[row])[0]))
    return checks,withheld


def run(a):
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'report.json').exists():raise ValueError('Completed result already exists; do not repeat the solve')
    g,bundle,q0,added,held,future,fb=prepare_cut(a)
    enable_projection_backend()
    original_fit=[r for p,rows in g.bundle.abs_groups if g.cuts[p]==1 for r in rows]
    manifest=dict(schema='rinkspace.z190-main1-new-ad-corner-fixed-calibration-rf.v1',accuracy_accepted=False,
        sole_completed_Z190_warm=ref(a.current/'physical_main_soft_height_candidate_unaccepted.npz'),
        sole_completed_state=ref(a.current/'optimized_state6rf.npz'),current_source_manifest=ref(a.current/'input_manifest.json'),
        hard_scene=ref(g.scene_path),new_source_packet=ref(a.new_source),
        additional_source_packets=[ref(path) for path in getattr(a,'additional_source',[])],
        frozen_shared_slots=[0,1,2,3,4,5],shared_state6=q0[:6],fixed_height_scene_units=g.calibration.pivot_xyz[2],
        fixed_height_m=g.calibration.pivot_xyz[2]*SCENE_UNIT_M,
        model_position_and_geometry_changed=False,calibration_frozen_at_saved_current_values=True,
        independent_per_installation_height=True,source_and_native_alias_roles_preserved=True,
        complete_recovered_cut=1,source_exposures=len(bundle.stamps),cut_native_frames=sum(g.cuts[g.aliases[int(p)]]==1 for p in g.native),
        original_motion_pairs=len(next(np.load(b['path'],allow_pickle=False)['pair_pts'] for b in g.bindings if b.get('cut')==1 and b.get('role')=='within_cut_observed_source_ray_pairs')),
        original_motion_points=len(bundle.a),original_absolute_fit_rows=len(original_fit),new_corner_rows=len(added),new_corner_points=sum(len(r['uv']) for r in added),
        source_residual_rows=bundle.source_residual_rows,original_sigmas_and_robust_policy_unchanged=True,
        original_held_rows=len(held),untouched_future_frames=[h['pts'] for h in future],held_or_future_pixels_fitted=False,
        full_main_native_frames=len(g.native),full_main_exposures=len(g.views),
        untouched_cuts=[4,7,9],untouched_view_policy='copy original warm CanonicalView objects and native aliases exactly',
        cut_decomposition_basis='six shared slots fixed; no cross-cut motion; per-view roll prior independent; shared priors constant',
        solver='existing solve_frozen_pivot free columns6 onward; fixed_saved_height avoids nominal1.6 reset',
        original_roll_prior_retained=True,new_source_uncertainty=[r['source_uncertainty'] for r in added],
        latest_hardware_and_height_provenance=calibration_context('main_long_side',source_sha256=SOURCE_SHA),
        source_bindings=g.bindings,future_source_bindings=fb,solve_started=a.solve,no_video_generated=True)
    snapshot=a.output/'absolute_source_rows.json'
    packet=dict(original_fit_rows=original_fit,new_direct_fit_rows=added,original_held_rows=held,untouched_future_groups=future)
    snapshot.write_text(json.dumps(jsonable(packet),ensure_ascii=False,indent=2),encoding='utf8')
    np.savez_compressed(a.output/'prepared_state6rf.npz',q=q0,canonical_pts=np.asarray(bundle.stamps))
    checks=[derivative_replay(bundle,q0,g.calibration,p) for p in sorted({r['pts'] for r in added})]
    manifest['actual_source_derivative_replay']=checks
    manifest['source_snapshot']=ref(snapshot)
    (a.output/'input_manifest.json').write_text(json.dumps(jsonable(manifest),ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(jsonable(dict(stage='prepared_current_main1_fixed_calibration',exposures=len(bundle.stamps),original_motion_points=len(bundle.a),new_corner_points=manifest['new_corner_points'],height_m=manifest['fixed_height_m'],solve_requested=a.solve))),flush=True)
    if not a.solve:return manifest
    q,history,seconds=solve_frozen_pivot(bundle,q0,a.output)
    if not np.array_equal(q[:6],q0[:6]):raise ValueError('Completed main shared calibration changed')
    model=PhysicalCameraModel({IDENTIFIER:g.calibration});views=dict(g.views)
    for p in bundle.stamps:
        i=bundle.index[p];views[p]=CanonicalView(IDENTIFIER,ViewState(tuple(q[6+4*i:9+4*i]),float(q[9+4*i]-np.log(1920.))))
    unchanged=[p for p in views if g.cuts[p]!=1]
    if any(views[p]!=g.views[p] for p in unchanged):raise ValueError('An untouched cut view changed')
    model.save_npz(a.output/'physical_main1_neighbor_rf_candidate_unaccepted.npz',g.native,g.aliases,views,source_sha256=SOURCE_SHA,time_base='1/15360')
    data=model.exposure_arrays(g.native,g.aliases,views)
    warm=np.load(a.current/'candidate_cameras.npz',allow_pickle=False)
    untouched_native=np.array([g.cuts[g.aliases[int(p)]]!=1 for p in g.native])
    if not np.array_equal(data['camera25'][untouched_native],warm['camera25'][untouched_native]):raise ValueError('Unchanged other-cut camera arrays differ')
    if not np.array_equal(data['pts'],warm['pts']) or not np.array_equal(data['exposure_representative_pts'],warm['exposure_representative_pts']):raise ValueError('Original native/exposure roles changed')
    np.savez_compressed(a.output/'candidate_cameras.npz',**data,model_sha256=HARD_SCENE_SHA,source_sha256=SOURCE_SHA)
    np.savez_compressed(a.output/'optimized_cut1_state6rf.npz',q=q,canonical_pts=np.asarray(bundle.stamps))
    absolute,withheld=source_comparison(g,bundle,q,added,held,future)
    report=dict(manifest,solve_seconds=seconds,history=history,source_comparison=absolute,held_and_future_source_comparison=withheld,
        shared_state6_bit_exact=True,untouched_cut_views_bit_exact=True,untouched_cut_dense_camera_arrays_bit_exact=True,
        native_source_aliases_bit_exact=True,whole_cut_visual_acceptance=False,no_video_generated=True)
    (a.output/'report.json').write_text(json.dumps(jsonable(report),ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(jsonable(dict(stage='completed_current_main1_fixed_calibration',seconds=seconds,
        new_source=[x for x in absolute if x['new_source']],shared_state_unchanged=True,untouched_cut_views_unchanged=True))),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--current',type=Path,default=DEFAULT_CURRENT)
    p.add_argument('--new-source',type=Path,default=CURRENT/'z190_main_ad2_right_source_audit_v1/ad2_right_external_229194240_source_candidate.json')
    p.add_argument('--additional-source',type=Path,action='append',default=[],help='Additional directly read development packets; exact original fit/future roles remain checked')
    p.add_argument('--output',type=Path,default=CURRENT/'z190_main_ad2_neighbor_rf_v1')
    p.add_argument('--solve',action='store_true',help='Run existing full-cut source RF recovery; omission only prepares')
    run(p.parse_args())
