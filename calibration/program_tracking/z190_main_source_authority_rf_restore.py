"""Recover main1/9 R/f from current-v5 physical states with source admission.

Absolute rows come only from the complete authority packet. Original ray UVs,
pair sigmas, per-view roll and fixed calibration remain unchanged. No source
depth, world identity, motion row removal, interpolation or video is invented.
"""
from __future__ import annotations
import argparse,json
from dataclasses import replace
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from z190_main_source_objective_audit import CURRENT,DEFAULT_CURRENT,ref,jsonable,IDENTIFIER
from main7_source_camera_candidate import sha
from main7_source_bundle_candidate import Bundle
from z190_continuous_source_bundle import StandingSourceBundle,HARD_SCENE_SHA,SOURCE_SHA,SCENE_UNIT_M
from z190_physical_camera import PhysicalCameraModel,CanonicalView,ViewState
from z190_low_opening_rf_refine import solve_frozen_pivot,convert_row
from z190_main_ad2_neighbor_rf_refine import derivative_replay
from z190_source_camera_diagnostic import source_errors
from continuous_central_brown4_optin import enable_projection_backend
from camera_hardware_metadata import calibration_context

AUTHORITY=CURRENT/'z190_main_source_authority_v1/source_authority_rows.json'
CURRENT_PHYSICAL=CURRENT/'z190_main_ad2_four_exposures_rf_v1/physical_main1_neighbor_rf_candidate_unaccepted.npz'
AUTHORITY_SHA='31ab40333fc27e55a0105bd9f9f61d221a097eac3715aedea5c7316be1f06576'
ADMISSION_SHA='9e876ecaf14b587cc23466c63063cb1cb26114f79f739f3f3bab5912eb744c41'

def read_json(path):return json.loads(Path(path).read_text(encoding='utf8'))

def numeric_identity(row):
    """Admission may add metadata/split, never change measured or world data."""
    keys=('pts','source_native_pts','id','origin','point_correspondence','physical_ids','uv','world','sigma','normal','normal_interval')
    return json.dumps({k:row[k] for k in keys if k in row},sort_keys=True,separators=(',',':'),ensure_ascii=False)

def prepare(a):
    if sha(a.authority)!=AUTHORITY_SHA:raise ValueError('Authorized complete source admission packet changed')
    packet=read_json(a.authority);record=packet['source_authority_admission']
    if sha(record['path'])!=ADMISSION_SHA or record['sha256']!=ADMISSION_SHA:raise ValueError('Source-authority provenance changed')
    admission=read_json(record['path'])
    if admission['scene_sha256']!=HARD_SCENE_SHA or admission['source_sha256']!=SOURCE_SHA:raise ValueError('Authority belongs to another hard scene/source')
    original=admission['original_packet']
    if sha(original['path'])!=original['sha256']:raise ValueError('Original immutable absolute inventory changed')
    old=read_json(original['path'])
    if Counter(map(numeric_identity,old['fit_rows']))!=Counter(map(numeric_identity,packet['fit_rows']+packet['pending_rows'])):
        raise ValueError('Authority changes original measured/world/sigma/role fields or original multiplicity')
    if old['held_rows']!=packet['held_rows'] or old['untouched_future_groups']!=packet['untouched_future_groups']:
        raise ValueError('Original held/future record changed')
    if any(r['split']!='fit' for r in packet['fit_rows']) or any(r['split']!='unfitted_identity_pending' for r in packet['pending_rows']):
        raise ValueError('Invalid source-authority split')
    parent_manifest=admission['current_source_manifest']
    if sha(parent_manifest['path'])!=parent_manifest['sha256']:raise ValueError('Current v5 main source binding changed')
    v5_manifest=read_json(parent_manifest['path']);warm_ref=Path(parent_manifest['path']).parent/'physical_main1_neighbor_rf_candidate_unaccepted.npz'
    if a.current.resolve()!=warm_ref.resolve():raise ValueError('Warm must be current four-source physical component')
    model,native,aliases,views=PhysicalCameraModel.load_npz(a.current)
    calibration=model.calibrations[IDENTIFIER]
    whole=read_json(CURRENT/'z190_full_programme_source_authority_v5/z190_full_programme_camera_candidate.binding.json')
    exact_binding=next(b for b in whole['fitted_installation_bindings'] if Path(b['path']).resolve()==a.current.resolve())
    if exact_binding['sha256']!=sha(a.current):raise ValueError('Warm does not equal current v5 main component')
    # Read original classification/exposure inventory only. Historical numeric
    # camera arrays are never extracted or used as R/f initializers here.
    baseline=read_json(admission['baseline_full_source_manifest']['path']);argv=baseline['run_command_argv']
    def arg(name):return Path(argv[argv.index('--'+name)+1])
    base,previous,scene_path=map(arg,['base','previous-source-data','scene'])
    if sha(scene_path)!=HARD_SCENE_SHA:raise ValueError('Hard scene changed')
    seven=previous/'candidate_cameras.npz'
    if not seven.exists():seven=previous/'main7_candidate_cameras.npz'
    stream_paths={1:base/'main1_geometry/main1_B_reparameterized_numerical_seed.npz',
        4:previous/'main4_candidate_cameras.npz',7:seven,9:base/'main9_geometry/main9_B_reparameterized_numerical_seed.npz'}
    cuts={};classification=[]
    for cut,path in stream_paths.items():
        z=np.load(path,allow_pickle=False);pts=z['pts'];representatives=z['exposure_representative_pts']
        for n,p in zip(pts,representatives):
            n,p=int(n),int(p)
            if aliases.get(n)!=p:raise ValueError('Current physical exposure alias differs from original classification')
            if p in cuts and cuts[p]!=cut:raise ValueError('Cross-cut exposure')
            cuts[p]=cut
        classification.append(dict(cut=cut,source_exposure_inventory=ref(path),camera_arrays_read=False,native_frames=len(pts)))
    if set(cuts)!=set(views) or len(native)!=sum(r['native_frames'] for r in classification):raise ValueError('Incomplete current full-main exposure inventory')
    observed={p:[] for p in views}
    for r in packet['fit_rows']:
        p=aliases[int(r['pts'])];observed[p].append(convert_row(r))
    raybindings=[b for b in v5_manifest['source_bindings'] if b.get('role')=='within_cut_observed_source_ray_pairs']
    pairs={};rayrecords=[]
    for b in raybindings:
        if sha(b['path'])!=b['sha256']:raise ValueError('Original motion source packet changed')
        arc=np.load(b['path'],allow_pickle=False);z={k:arc[k] for k in arc.files};cut=int(b['cut']);pp=[]
        for i,(pa,pb) in enumerate(z['pair_pts']):
            pa,pb=int(pa),int(pb)
            if aliases[pa]!=pa or aliases[pb]!=pb or cuts[pa]!=cut or cuts[pb]!=cut or pa==pb:raise ValueError('Original source pair exposure/direction/cut failed')
            lo,hi=map(int,z['offsets'][i:i+2]);pp.append((pa,pb,z['source_uv'][lo:hi],z['target_uv'][lo:hi]))
        pairs[cut]=pp;rayrecords.append(dict(**b,pairs=len(pp),points=len(z['source_uv']),direction_uv_unchanged=True,
            bad_bridge_exclusions_preserved=z['excluded_original_pair_indices'].tolist() if 'excluded_original_pair_indices' in z else []))
    if set(pairs)!={1,4,7,9}:raise ValueError('Full main source graph incomplete')
    bundles={};states={};prepared_checks=[]
    for cut in (1,9):
        stamps=sorted(p for p in views if cuts[p]==cut);data=Bundle(stamps,observed,pairs[cut])
        bundle=StandingSourceBundle(data,calibration,prior_wide_focal_px=baseline['frozen_wide_prior_focal_px'],
            fixed_height=True,fixed_height_scene_units=calibration.pivot_xyz[2])
        q=np.r_[np.asarray(calibration.pivot_xyz)/100.,calibration.principal_x_fraction[0],calibration.principal_y_fraction[0],calibration.k1[0],
            np.concatenate([np.r_[views[p].view.rvec_world_to_camera,views[p].view.log_f_ratio+np.log(1920.)] for p in stamps])]
        if not all(np.array_equal(bundle.camera(q,p),model.camera25(views[p])) for p in stamps):raise ValueError('Physical current R/f reconstruction is not exact')
        if cut==1 and (len(data.a)!=246171 or len(pairs[cut])!=1384):raise ValueError('Original full cut1 rays replaced or missing')
        if cut==1 and any(int(p) not in [r['pts'] for rr in observed.values() for r in rr if r['id'].startswith('ad2_right_external')] for p in [229181696,229189376,229192448,229193984]):
            raise ValueError('Four direct Ad2 right source rows not preserved')
        bundles[cut],states[cut]=bundle,q
        # One actual source exposure per changed cut; exact original incident
        # rays plus source curves/corners, both RF endpoint derivatives retained.
        sample=229193984 if cut==1 else next(p for p in stamps if observed[p])
        prepared_checks.append(dict(cut=cut,**derivative_replay(bundle,q,calibration,sample)))
    return SimpleNamespace(packet=packet,admission=admission,model=model,native=native,aliases=aliases,views=views,
        calibration=calibration,cuts=cuts,observed=observed,bundles=bundles,states=states,rayrecords=rayrecords,
        classification=classification,scene_path=scene_path,source_checks=prepared_checks,baseline=baseline)

def all_source_changes(g,after_views):
    result=[]
    for role,rows in [('fit',g.packet['fit_rows']),('original_held',g.packet['held_rows']),('unfitted_identity_pending',g.packet['pending_rows'])]:
        for original in rows:
            p=g.aliases[int(original['pts'])]
            if g.cuts[p] not in (1,9):continue
            row=convert_row(original)
            result.append(dict(role=role,canonical_pts=p,source_native_pts=original.get('source_native_pts',original['pts']),cut=g.cuts[p],
                id=row['id'],origin=row['origin'],physical_ids=row.get('physical_ids'),point_correspondence=bool(row.get('point_correspondence')),
                before=source_errors(g.model.camera25(g.views[p]),[row])[0],after=source_errors(g.model.camera25(after_views[p]),[row])[0]))
    for group in g.packet['untouched_future_groups']:
        for original in group['rows']:
            p=g.aliases[int(group['pts'])];row=convert_row(original)
            result.append(dict(role='untouched_future',canonical_pts=p,source_native_pts=group['pts'],cut=group['cut'],
                id=row['id'],origin=row['origin'],point_correspondence=bool(row.get('point_correspondence')),
                identity_scope='original world proposal retained; this projection does not confirm physical circle/Lane identity',
                before=source_errors(g.model.camera25(g.views[p]),[row])[0],after=source_errors(g.model.camera25(after_views[p]),[row])[0]))
    return result

def render_check(g,views,output):
    """Source review images show the complete unchanged renderer inventory."""
    from render_result import SceneRenderer,decode_selected,PersonMasks,DEFAULT_MASKS,DEFAULT_PEOPLE
    from review_shared_intrinsic_source_candidate import source_time
    import cv2,hashlib
    # All four newly added sources, original right, adjacent future and main9
    # first pending/last source. Selection is time/source role, not residual.
    selected={229181952,229186560,229189632,229192704,229194240,229198592}
    pending9=[int(r.get('source_native_pts',r['pts'])) for r in g.packet['pending_rows'] if g.cuts[g.aliases[int(r['pts'])]]==9]
    selected.update((min(pending9),max(pending9)))
    selected.update(int(h['pts']) for h in g.packet['untouched_future_groups'] if h['cut']==9)
    renderer=SceneRenderer(read_json(g.scene_path));masks=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE);images=[]
    folder=output/'source_full_inventory_review';folder.mkdir()
    video=Path('D:/OneDrive/codex_projs/RinkSpace/data/input/oxXu1avhrvs.mp4')
    for p,native in decode_selected(video,sorted(selected)):
        canonical=g.aliases[p]
        try:mask,status=masks.get(p,require_sam2=True)
        except ValueError:mask,status=None,'no confirmed source mask'
        for label,vv in [('v5_before',g.views),('authority_candidate',views)]:
            pic,metrics=renderer.draw(native,g.model.camera25(vv[canonical]),mask,
                title=f'{source_time(p)} | Z190 + source / {label} / unaccepted')
            path=folder/f'{p}_{label}_all_model.jpg';cv2.imwrite(str(path),pic,[cv2.IMWRITE_JPEG_QUALITY,94])
            images.append(dict(source_native_pts=p,canonical_pts=canonical,cut=g.cuts[canonical],label=label,source_time=source_time(p),
                native_pixels_bgr_sha256=hashlib.sha256(native.tobytes()).hexdigest(),image=ref(path),mask_status=status,metrics=metrics))
    return images

def run(a):
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'input_manifest.json').exists():raise ValueError('Existing prepared run; do not duplicate recovery')
    enable_projection_backend();g=prepare(a)
    manifest=dict(schema='rinkspace.main-z190-source-authority-fixed-calibration-rf-restore.v1',accuracy_accepted=False,
        current_v5_physical_warm=ref(a.current),authority_source_packet=ref(a.authority),source_authority_record=ref(g.packet['source_authority_admission']['path']),
        hard_scene=ref(g.scene_path),source_sha256=SOURCE_SHA,units='fixed native scene; uniform assumed metric multiplier0.31353729994266055',
        source_roles=dict(fit_rows=len(g.packet['fit_rows']),held_rows=len(g.packet['held_rows']),pending_rows=len(g.packet['pending_rows'])),
        frozen_shared_slots=[0,1,2,3,4,5],saved_main_independent_height_m=g.calibration.pivot_xyz[2]*SCENE_UNIT_M,
        pivot_xyz_scene=g.calibration.pivot_xyz,principal_xy_fraction=[g.calibration.principal_x_fraction[0],g.calibration.principal_y_fraction[0]],k1=g.calibration.k1[0],
        exact_complete_current_Rf_reconstruction=True,no_old_RF_initializer=True,restored_cuts=[1,9],untouched_cuts=[4,7],
        classification_exposure_inventory=g.classification,motion_bindings=g.rayrecords,actual_source_derivative_replay=g.source_checks,
        all_original_rays_sigmas_roll_preserved=True,no_current_residual_based_source_exclusion=True,no_held_pending_future_fit=True,
        per_cut={str(c):dict(exposures=len(b.stamps),original_motion_points=len(b.a),absolute_fit_rows=sum(len(rr) for _,rr in b.abs_groups),source_scalar_residual_rows=b.source_residual_rows) for c,b in g.bundles.items()},
        source_admission_policy='Complete authority inventory replaces historical unconditional additional_fit_sources; pending source identity remains raw review only',
        solve_requested=a.solve,whole_cut_visual_acceptance=False,no_video_generated=True)
    (a.output/'input_manifest.json').write_text(json.dumps(jsonable(manifest),ensure_ascii=False,indent=2),encoding='utf8')
    for c,q in g.states.items():np.savez_compressed(a.output/f'prepared_cut{c}_state6rf.npz',q=q,canonical_pts=g.bundles[c].stamps)
    print(json.dumps(jsonable(dict(stage='source_authority_prepared',per_cut=manifest['per_cut'],height_m=manifest['saved_main_independent_height_m'],solve_requested=a.solve))),flush=True)
    if not a.solve:return manifest
    views=dict(g.views);history={};seconds={}
    for cut in (1,9):
        folder=a.output/f'cut{cut}';folder.mkdir()
        print(json.dumps(dict(stage='source_authority_restore_started',cut=cut,exposures=len(g.bundles[cut].stamps))),flush=True)
        q,h,elapsed=solve_frozen_pivot(g.bundles[cut],g.states[cut],folder)
        if not np.array_equal(q[:6],g.states[cut][:6]):raise ValueError('Fixed current main calibration changed')
        for i,p in enumerate(g.bundles[cut].stamps):views[p]=CanonicalView(IDENTIFIER,ViewState(tuple(q[6+4*i:9+4*i]),float(q[9+4*i]-np.log(1920.))))
        np.savez_compressed(a.output/f'optimized_cut{cut}_state6rf.npz',q=q,canonical_pts=g.bundles[cut].stamps)
        history[str(cut)],seconds[str(cut)]=h,elapsed
        print(json.dumps(dict(stage='source_authority_restore_complete',cut=cut,seconds=elapsed)),flush=True)
    untouched=[p for p in views if g.cuts[p] in (4,7)]
    if any(views[p]!=g.views[p] for p in untouched):raise ValueError('Untouched main4/7 view changed')
    physical=a.output/'physical_main_source_authority_candidate_unaccepted.npz'
    settings=dict(g.calibration.field_settings)
    settings['source_authority_admission']=dict(**ref(g.packet['source_authority_admission']['path']),
        unconfirmed_ice_proposals_excluded_from_fit=True)
    settings['hardware_context']=calibration_context('main_long_side',source_sha256=SOURCE_SHA)
    output_calibration=replace(g.calibration,field_settings=settings)
    output_model=PhysicalCameraModel({IDENTIFIER:output_calibration})
    output_model.save_npz(physical,g.native,g.aliases,views,source_sha256=SOURCE_SHA,time_base='1/15360')
    saved,native,aliases,loaded=PhysicalCameraModel.load_npz(physical)
    if aliases!=g.aliases or not np.array_equal(native,g.native) or saved.calibrations[IDENTIFIER]!=output_calibration:
        raise ValueError('Calibration/source-role roundtrip changed')
    if not all(np.array_equal(saved.camera25(loaded[p]),g.model.camera25(views[p])) for p in views):
        raise ValueError('Provenance metadata update changed numerical camera values')
    data=saved.exposure_arrays(native,aliases,loaded);before=g.model.exposure_arrays(g.native,g.aliases,g.views)
    untouched_native=np.asarray([g.cuts[g.aliases[int(p)]] in (4,7) for p in native])
    if not np.array_equal(data['camera25'][untouched_native],before['camera25'][untouched_native]):raise ValueError('Other-cut dense camera changed')
    np.savez_compressed(a.output/'candidate_cameras.npz',**data,model_sha256=HARD_SCENE_SHA,source_sha256=SOURCE_SHA)
    changes=all_source_changes(g,views)
    (a.output/'all_absolute_source_changes.json').write_text(json.dumps(jsonable(changes),ensure_ascii=False,indent=2),encoding='utf8')
    images=render_check(g,views,a.output) if a.render else []
    report=dict(manifest,physical_candidate=ref(physical),history=history,solve_seconds_by_cut=seconds,
        calibration_numeric_bit_exact=True,provenance_only_calibration_metadata_updated=True,
        other_cut4_7_views_and_dense_cameras_bit_exact=True,source_aliases_bit_exact=True,
        complete_main_native_frames=len(native),complete_main_exposures=len(views),all_absolute_checks=ref(a.output/'all_absolute_source_changes.json'),
        source_review_images=images,source_identity_pending_not_accuracy_validation=True,main9_changed_exposures=sum(views[p]!=g.views[p] for p in views if g.cuts[p]==9))
    (a.output/'report.json').write_text(json.dumps(jsonable(report),ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(jsonable(dict(stage='source_authority_completed',seconds=seconds,main9_changed=report['main9_changed_exposures'],images=len(images),physical=str(physical)))),flush=True)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--current',type=Path,default=CURRENT_PHYSICAL)
    ap.add_argument('--authority',type=Path,default=AUTHORITY)
    ap.add_argument('--output',type=Path,default=CURRENT/'z190_main_source_authority_rf_v1')
    ap.add_argument('--solve',action='store_true');ap.add_argument('--render',action='store_true')
    run(ap.parse_args())
