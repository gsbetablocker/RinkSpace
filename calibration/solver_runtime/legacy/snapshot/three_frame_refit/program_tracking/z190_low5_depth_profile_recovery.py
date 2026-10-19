"""Re-evaluate current complete cut5 with bounded profiled motion depth.

The current nonzero installation, optics and hard93c scene stay fixed.
Every original cut5 motion row, UV, sigma and absolute-source role is retained.
Other cuts are copied exactly.  A depth lower bound is an explicit engineering
condition on source camera Z; it supplies no measured depth or surface identity.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
from pathlib import Path
from types import SimpleNamespace

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np

from z190_low_nonzero_mount_full_source import prepare as original_prepare, source_changes, IDENTIFIER
from z190_low_static_foreground_rf_recovery import fixed_mount_rf_solve
from z190_low8_direct_source_rf_recovery import FixedCutBundle, save_conditional_depth_trace
from z190_low_opening_rf_refine import convert_row
from z190_main_zoom_k1_source_information import read, write, ref
from z190_physical_camera import PhysicalCameraModel, CanonicalView
from z190_profiled_motion_schur import SchurMotionKernel
from z190_continuous_source_bundle import SCENE_UNIT_M, SOURCE_SHA, HARD_SCENE_SHA
from continuous_central_brown4_optin import enable_projection_backend


def bind_core_ice_source(path, source_motion, base, aliases):
    """Keep original rows; give only reviewed static ice material its z=0 plane."""
    planes = np.zeros((len(base.a), 4))
    records = []
    if path is None:
        return planes, records, None
    packet = read(path)
    if packet['source_motion_graph'] != ref(Path(source_motion['rawpath'])):
        raise ValueError('Core ice source original motion graph changed')
    if packet['scene_sha256'] != HARD_SCENE_SHA or packet['source_sha256'] != SOURCE_SHA:
        raise ValueError('Core ice source event or hard scene changed')
    if packet['source_review_status'] != 'source_native_material_checked_no_global_Lane_id_claim':
        raise ValueError('Core ice material source review is incomplete')
    with np.load(source_motion['rawpath'], allow_pickle=False) as raw, np.load(source_motion['selected_rows'], allow_pickle=False) as selected:
        ids = selected['original_global_row_indices']
        if len(ids) != len(base.a):
            raise ValueError('Core ice source retained row order changed')
        lookup = {int(g):j for j,g in enumerate(ids)}
        for row in packet['qualified_rows']:
            if not row['both_end_source_material_reviewed'] or row['physical_surface_identity'] != 'native_visible_ice_plane_z0':
                raise ValueError('Core ice motion lacks both-end real static material support')
            if not all(check['qualified'] and not check['clock_reflection_nearby'] and not check['person_nearby'] for check in row['source_endpoint_material_checks']):
                raise ValueError('Core ice motion contains source reflection or foreground')
            pair = int(row['original_pair_index'])
            g = int(row['original_global_index'])
            if g != int(raw['offsets'][pair])+int(row['original_local_index']):
                raise ValueError('Core ice original source row identity changed')
            pts = raw['pair_pts'][pair].astype(int).tolist()
            if pts != row['native_pair_pts']:
                raise ValueError('Core ice original native PTS changed')
            j = lookup[g]
            if not np.array_equal(raw['xy0'][g], row['xy0']) or not np.array_equal(raw['xy1'][g], row['xy1']):
                raise ValueError('Core ice original native UV changed')
            if not np.array_equal(base.a[j], row['xy0']) or not np.array_equal(base.b[j], row['xy1']):
                raise ValueError('Core ice original solver UV changed')
            if [base.stamps[base.ia[j]], base.stamps[base.ib[j]]] != [aliases[p] for p in pts]:
                raise ValueError('Core ice exposure mapping changed')
            if row['split'] == 'held':
                continue
            if row['split'] != 'fit':
                raise ValueError('Core ice source split changed')
            planes[j] = [0.,0.,1.,0.]
            records.append(dict(global_bundle_index=j,cut=5,original_pair_index=pair,
                original_local_index=row['original_local_index'],original_global_index=g,
                source_native_pair_pts=pts,canonical_pair_pts=[aliases[p] for p in pts],
                xy0=row['xy0'],xy1=row['xy1'],surface_kind='ice_plane',
                physical_surface_identity=row['physical_surface_identity'],material_identity=row['material_identity'],
                source_role='original_motion_retained',review=ref(path),
                admission='both_end_native_static_ice_paint_or_texture_source_review',match_precision_not_measured=True))
    if len({r['global_bundle_index'] for r in records}) != len(records):
        raise ValueError('Core ice source finite rows duplicate')
    return planes, records, ref(path)


def run(a):
    out = a.output.resolve()
    if not out.is_relative_to(Path('D:/CodexData').resolve()):
        raise ValueError('Bulk output must stay in CodexData')
    out.mkdir(parents=True, exist_ok=False)
    oldout = out/'original_source_adapter_inputs'
    oldout.mkdir()
    cv2.setNumThreads(1)
    enable_projection_backend()
    original, _, _, _, native, aliases, _, rows, historical = original_prepare(oldout)
    current, _, allaliases, allviews = PhysicalCameraModel.load_npz(a.current)
    cal = current.calibrations[IDENTIFIER]
    views = {p:v for p,v in allviews.items() if v.installation_id == IDENTIFIER}
    if {int(p):allaliases[int(p)] for p in native} != aliases:
        raise ValueError('Current low source exposure aliases changed')
    scene = read(Path(historical['scene']['path']))
    if ref(Path(historical['scene']['path']))['sha256'] != HARD_SCENE_SHA:
        raise ValueError('Hard scene changed')
    stamps = [p for p in sorted(views) if 229945856 <= p < 230369024]
    index = {p:i for i,p in enumerate(stamps)}
    old_index = np.asarray([original.index[p] for p in stamps], int)
    inside = np.zeros(len(original.stamps), bool)
    inside[old_index] = True
    active = np.flatnonzero(inside[original.ia] & inside[original.ib])
    if np.any(inside[original.ia] ^ inside[original.ib]):
        raise ValueError('Original motion graph crosses cuts')
    if len(active) != 319568:
        raise ValueError('Complete original cut5 motion graph required')
    remap = np.full(len(original.stamps), -1, int)
    remap[old_index] = np.arange(len(stamps))
    observed = {p:list(rr) for p,rr in original.abs_groups if p in index}
    additional = []
    if a.new_source:
        import hashlib
        from render_result import advertisement_boundary_paths
        packet = read(a.new_source)
        if packet['scene_sha256'] != HARD_SCENE_SHA or packet['source_sha256'] != SOURCE_SHA:
            raise ValueError('Additional source changed event or scene')
        inventory = {(p['number'],p['role']):p['xyz'] for p in advertisement_boundary_paths(scene)}
        for role in ('fit_rows','held_rows'):
            for source in packet[role]:
                r = convert_row(source)
                identity = r['physical_identity']
                if identity.get('kind') == 'physical_ice_ring_boundary':
                    feature = next(f for f in scene['ice_features'] if f['id'] == identity['feature_id'])
                    ring = next(ring for ring in feature['rings'] if ring['role'] == identity['ring_role'])
                    if identity['visible_material'] != 'gray/blue inner disk external boundary against white middle annulus':
                        raise ValueError('Ice ring physical source material identity changed')
                    if identity['feature_id'] != 'F2' or identity['ring_role'] != 'inner' or float(ring['radius']) != identity['radius_scene_units']:
                        raise ValueError('Only reviewed fixed F2 inner boundary admitted')
                    # Identical sampling to render_result.build_marking_paths.
                    angles = np.linspace(0., 2*np.pi, 181)
                    world = np.asarray(ring['center_xyz']) + np.column_stack([np.cos(angles),np.sin(angles),np.zeros(len(angles))])*ring['radius']
                else:
                    world = inventory[identity['ad_id'],identity['geometry_role']]
                if not np.array_equal(world,r['world']):
                    raise ValueError('Additional source geometry differs from fixed renderer')
                bgr = cv2.imread(r['raw_png'])
                if bgr is None or hashlib.sha256(bgr.tobytes()).hexdigest() != r['native_bgr_sha256']:
                    raise ValueError('Additional source native pixels changed')
                if hashlib.sha256(np.asarray(r['uv'],np.float64).tobytes()).hexdigest() != r['source_uv_float64_sha256']:
                    raise ValueError('Additional measured source UV changed')
                if r['model_projection_used_to_generate_or_select_uv']:
                    raise ValueError('Only native measured source permitted')
                r['pts'] = aliases[int(r['source_native_pts'])]
                if r['pts'] not in index:
                    raise ValueError('Additional source outside cut5')
                if role == 'fit_rows':
                    if r['split'] != 'fit':
                        raise ValueError('Held source cannot be fitted')
                    observed.setdefault(r['pts'],[]).append(r)
                additional.append((role,r))
    groups = [(p,observed[p]) for p in stamps if p in observed]
    ms = original.motion_sigma.reshape(-1,2)[active].ravel()
    absolute_sigma = np.concatenate([np.concatenate([np.repeat(r['sigma'],2) for r in rr]) for p,rr in groups])
    base = SimpleNamespace(stamps=stamps,index=index,a=original.a[active],b=original.b[active],
        ia=remap[original.ia[active]],ib=remap[original.ib[active]],motion_sigma=ms,
        abs_groups=groups,native_residual_sigma=np.r_[ms,absolute_sigma])
    motion_source = next(r for r in historical['source_motion'] if r['cut'] == 5)
    planes, finite_records, finite_input = bind_core_ice_source(a.finite_ice_source,motion_source,base,aliases)
    kernel = SchurMotionKernel(minimum_source_depth_scene_units=a.minimum_unknown_depth_m/SCENE_UNIT_M)
    bundle = FixedCutBundle(base,cal,kernel,reference_pts=stamps[0],
        initial_reference_optical_height_scene=current.camera25(views[stamps[0]])[2],scene=scene,
        finite_planes=planes,finite_records=finite_records)
    q = np.r_[bundle.initial_shared(),np.concatenate([np.r_[views[p].view.rvec_world_to_camera,
        views[p].view.log_f_ratio+np.log(1920.)] for p in stamps])]
    if a.start_state:
        with np.load(a.start_state,allow_pickle=False) as saved:
            if not np.array_equal(saved['canonical_pts'],stamps) or not np.array_equal(saved['q'][:4],q[:4]):
                raise ValueError('Starting checkpoint differs from frozen installation or complete cut')
            q = saved['q'].copy()
    np.savez_compressed(out/'initial_current_state4rf.npz',q=q,canonical_pts=stamps)
    manifest = dict(schema='rinkspace.current-low5-bounded-profiled-motion-Rf-recovery.v1',
        current_complete=ref(a.current),scene=historical['scene'],source_sha256=SOURCE_SHA,
        original_source_absolute=historical['source_authority_rows'],
        original_source_adapter=ref(oldout/'input_manifest.json'),
        only_free='every cut5 actual-exposure R3/effective focal',
        fixed=['current pivotXYZ','current nonzero leverXYZ','PP','k1','hard93c','other8cut R/f'],
        original_motion_rows=len(active),actual_cut5_exposures=len(stamps),
        original_fit_rows=sum(len(rr) for p,rr in original.abs_groups if p in index),
        new_source=ref(a.new_source) if a.new_source else None,
        starting_checkpoint=ref(a.start_state) if a.start_state else None,
        new_source_rows=sum(role == 'fit_rows' for role,r in additional),
        finite_core_ice_source=finite_input,finite_core_ice_fit_motion_rows=len(finite_records),
        finite_source_held_plane_geometry_not_fitted=True,no_global_Lane_id_inferred_from_ice_plane=True,
        original_UV_sigma_roles_unchanged=True,held_pending_not_fitted=True,
        unknown_depth_closure=dict(minimum_source_camera_Z_m=a.minimum_unknown_depth_m,
            engineering_condition_not_measurement=True,source_pixels_not_reassigned_to_plane=True),
        entrypoint=ref(__file__),accuracy_accepted=False,whole_cut_accepted=False,video_generated=False)
    write(out/'input_manifest.json',manifest)
    exact = FixedCutBundle(base,cal,SchurMotionKernel(minimum_source_depth_scene_units=a.minimum_unknown_depth_m/SCENE_UNIT_M,schur=False),
        reference_pts=stamps[0],initial_reference_optical_height_scene=current.camera25(views[stamps[0]])[2],scene=scene,
        finite_planes=planes,finite_records=finite_records)
    write(out/'finite_core_ice_source_binding.json',finite_records)
    r,_ = bundle.evaluate(q)
    if not np.array_equal(r,exact.evaluate(q,False)):
        raise ValueError('Schur original objective replay differs')
    write(out/'source_depth_objective_replay.json',dict(native_objective_bitexact=True,
        source_sigma_unchanged=True,checks=kernel.latest_checks))
    depth = save_conditional_depth_trace(bundle,q,active,a.minimum_unknown_depth_m/SCENE_UNIT_M,out)
    write(out/'initial_depth_summary.json',depth)
    print(dict(stage='low5_prepared',actual=len(stamps),motion=len(active),depth_floor=depth['conditional_depth_floor_hits']),flush=True)
    del original,base,exact
    gc.collect()
    if not a.solve and not a.export_state:
        return
    if a.export_state:
        with np.load(a.export_state,allow_pickle=False) as saved:
            if not np.array_equal(saved['canonical_pts'],stamps) or not np.array_equal(saved['q'][:4],q[:4]):
                raise ValueError('Export checkpoint differs from frozen current graph')
            fitted = saved['q'].copy()
        history = read(a.export_state.parent/'bundle_iterations.json')
        seconds = history[-1]['elapsed_seconds']
        write(out/'bundle_iterations.json',history)
        manifest['export_checkpoint_without_further_solve'] = ref(a.export_state)
    else:
        fitted,history,seconds = fixed_mount_rf_solve(bundle,q,out,require_undamped_precision=a.require_undamped_precision,phases=(True,))
    newviews = dict(views)
    for p in stamps:
        newviews[p] = CanonicalView(IDENTIFIER,bundle.view(fitted,p))
    settings = dict(cal.field_settings)
    settings['low5_depth_profile_recovery'] = dict(input_manifest=ref(out/'input_manifest.json'),
        actual_cut5_only=True,fixed_current_installation=True,accuracy_accepted=False)
    model = PhysicalCameraModel({IDENTIFIER:replace(cal,field_settings=settings)})
    path = out/'physical_low5_depth_profile_candidate_unaccepted.npz'
    model.save_npz(path,native,aliases,newviews,source_sha256=SOURCE_SHA,time_base='1/15360')
    loaded,pts2,aliases2,views2 = PhysicalCameraModel.load_npz(path)
    if not np.array_equal(pts2,native) or aliases2 != aliases:
        raise ValueError('Native source exposure scope changed')
    if not all(np.array_equal(current.camera25(views[p]),loaded.camera25(views2[p])) for p in views if p not in index):
        raise ValueError('Low3/8 must remain bitexact')
    changes = source_changes(rows+additional,current,allviews,loaded,views2,aliases)
    write(out/'all_absolute_source_changes.json',changes)
    np.savez_compressed(out/'optimized_state4rf.npz',q=fitted,canonical_pts=stamps)
    write(out/'report.json',dict(manifest,candidate=ref(path),seconds=seconds,history=history,
        other_low_cuts_numeric_bitexact=True,source_changes=ref(out/'all_absolute_source_changes.json'),
        disposition='real_source_review_pending_not_adopted'))
    print(dict(stage='low5_exported',candidate=str(path),seconds=seconds),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--current',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--minimum-unknown-depth-m',type=float,default=.01)
    p.add_argument('--new-source',type=Path)
    p.add_argument('--finite-ice-source',type=Path)
    p.add_argument('--export-state',type=Path)
    p.add_argument('--start-state',type=Path)
    p.add_argument('--require-undamped-precision',action='store_true')
    p.add_argument('--solve',action='store_true')
    run(p.parse_args())
