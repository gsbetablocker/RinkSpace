"""Recover one current main cut from its unchanged source graph.

The current fixed pivot, nonzero lever, optics and hard scene are immutable.
Source person-envelope hits are outside static-rink motion; all other untyped
rows retain a profiled depth, with an explicit engineering near-depth bound.
No ice identity is inferred from a mask, a small residual or a projection.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np

from continuous_central_brown4_optin import enable_projection_backend
from render_result import PersonMasks, DEFAULT_MASKS, DEFAULT_PEOPLE
from z190_continuous_source_bundle import HARD_SCENE_SHA, SOURCE_SHA, SCENE_UNIT_M
from z190_low_static_foreground_rf_recovery import fixed_mount_rf_solve, sample_foreground
from z190_low8_direct_source_rf_recovery import FixedCutBundle
from z190_moving_center_motion import MovingDepthDomainError, moving_center_motion_value_jacobians
from z190_physical_camera import PhysicalCameraModel, CanonicalView
from z190_profiled_motion_schur import SchurMotionKernel
from z190_source_camera_diagnostic import source_errors
from z190_low_opening_rf_refine import convert_row
from z190_main_zoom_k1_source_information import read, write, ref

OLD = Path('D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00009/RinkSpace_tmp')
CURRENT = OLD / 'z190_full_programme_depth_profile_low8_v1/z190_full_programme_camera_candidate.npz'
CURRENT_SHA = '7625b07752027a857fcc9ea2afa979f2c639152308851958e94bdeee27994e1d'
MANIFEST = OLD / 'z190_main_nonzero_mount_full_source_v1/input_manifest.json'
AUTHORITY = OLD / 'z190_main_source_authority_v1/source_authority_rows.json'


def conditional_depth_state_arrays(bundle,q,*,chunk_points=65536):
    """Replay all original depth rows in order without a full-point kernel call.

    The exported fields are the same row arrays formerly saved by prepare.
    They are filled directly, so chunk intermediates and duplicate lists of
    complete source state never coexist. No pixels, planes, sigma or index
    are changed; scalar kernel metadata was not part of the saved NPZ.
    """
    if chunk_points<=0:raise ValueError('Positive motion chunk size required')
    vv,rr,dr,cc,ds,dv=bundle.cache(q,False)
    cal=bundle.current_calibration(q);focal=np.exp(vv[:,3])
    principal=np.array([1920*cal.principal_x_fraction[0],1080*cal.principal_y_fraction[0]])
    arrays={};total=len(bundle.a);keys=None
    for lo in range(0,total,chunk_points):
        hi=min(lo+chunk_points,total);ia,ib=bundle.ia[lo:hi],bundle.ib[lo:hi]
        try:
            result=moving_center_motion_value_jacobians(
                bundle.a[lo:hi],bundle.b[lo:hi],focal[ia],focal[ib],rr[ia],rr[ib],None,None,
                cc[ia],cc[ib],None,None,None,None,principal_px=principal,k1=cal.k1[0],
                sigma=bundle.motion_sigma[2*lo:2*hi].reshape(-1,2),jac=False,return_state=True,
                finite_normals=bundle.finite_planes[lo:hi,:3],finite_d=-bundle.finite_planes[lo:hi,3],
                finite_mask=bundle.finite_mask[lo:hi])
        except MovingDepthDomainError as exc:
            # The old full call reported active-global row indices. Keep that
            # provenance when a conditional domain fails in a later chunk.
            raise MovingDepthDomainError(exc.reason,np.asarray(exc.indices)+lo)from exc
        state={key:value for key,value in result[-1].items()if isinstance(value,np.ndarray)}
        if keys is None:
            keys=set(state)
            for key,value in state.items():
                if value.ndim==0 or len(value)!=hi-lo:raise ValueError('Depth state must preserve original row arrays')
                arrays[key]=np.empty((total,*value.shape[1:]),dtype=value.dtype)
        elif set(state)!=keys:raise ValueError('Depth state fields changed between motion chunks')
        for key,value in state.items():
            target=arrays[key]
            if value.shape!=(hi-lo,*target.shape[1:])or value.dtype!=target.dtype:
                raise ValueError('Depth state dtype/row shape changed between motion chunks')
            target[lo:hi]=value
        del result,state
    return arrays


def prepare(args):
    if ref(args.current)['sha256'] != CURRENT_SHA:
        raise ValueError('Start from the latest complete camera')
    model, native, aliases, views = PhysicalCameraModel.load_npz(args.current)
    source_manifest = read(MANIFEST)
    scene_path = Path(source_manifest['scene']['path'])
    if ref(scene_path)['sha256'] != HARD_SCENE_SHA:
        raise ValueError('Hard scene changed')
    authority = read(AUTHORITY)
    if ref(AUTHORITY) != source_manifest['source_authority']:
        raise ValueError('Main source authority changed')
    shot = read(Path(__file__).with_name('source_shots.json'))['shots'][args.cut]
    stamps = sorted(p for p in views if shot['start_pts'] <= p < shot['end_pts'])
    if not stamps or len({views[p].installation_id for p in stamps}) != 1:
        raise ValueError('Incomplete single-installation cut')
    index = {p: i for i, p in enumerate(stamps)}
    identifier = views[stamps[0]].installation_id
    cal = model.calibrations[identifier]
    binding = next(b for b in source_manifest['original_rays'] if b['cut'] == args.cut)
    ray_path = Path(binding['path'])
    if ref(ray_path)['sha256'] != binding['sha256']:
        raise ValueError('Original motion source changed')
    admission_cache = getattr(args, 'source_admission_cache', None)
    cache_binding = None
    if admission_cache is not None:
        admission_cache = Path(admission_cache)
        cache_manifest = read(admission_cache.parent/'input_manifest.json')
        for key, expected in [('cut',args.cut),('current_complete',ref(args.current)),
                              ('motion',ref(ray_path)),('source_sha256',SOURCE_SHA),
                              ('source_authority',ref(AUTHORITY)),('scene',ref(scene_path))]:
            if cache_manifest[key] != expected:
                raise ValueError('Existing source-person admission binding changed: '+key)
        if cache_manifest['mask_role'] != 'existing exact-native conservative person envelope, not static rink; no material plane assigned':
            raise ValueError('Source-person cache must retain the exact existing mask admission')
        cache_binding = dict(cache=ref(admission_cache),manifest=ref(admission_cache.parent/'input_manifest.json'),
            original_source_pixels_endpoints_sigmas_aliases_verified=True,
            mask_selection_reused_not_recomputed=True,material_planes_not_inferred_from_mask=True)
    masks = PersonMasks(DEFAULT_MASKS, DEFAULT_PEOPLE) if admission_cache is None else None
    source_masks = {}
    def get_mask(p):
        if p not in source_masks:
            source_masks[p] = masks.get(p, require_sam2=True)[0]
        return source_masks[p]
    with np.load(ray_path, allow_pickle=False) as z:
        a = z['source_uv'].astype(float)
        b = z['target_uv'].astype(float)
        pairs = z['pair_pts'].copy()
        offsets = z['offsets'].copy()
    ia = np.empty(len(a), np.int32)
    ib = np.empty(len(a), np.int32)
    sigma = np.empty((len(a), 2), float)
    foreground = np.zeros(len(a), bool)
    for i, (pa, pb) in enumerate(pairs):
        pa, pb = int(pa), int(pb)
        if aliases[pa] != pa or aliases[pb] != pb or pa == pb:
            raise ValueError('Motion endpoints must retain confirmed actual-exposure identity')
        lo, hi = map(int, offsets[i:i+2])
        ia[lo:hi], ib[lo:hi] = index[pa], index[pb]
        sigma[lo:hi] = max(2., np.sqrt(hi-lo)/3.)
        if admission_cache is None:
            foreground[lo:hi] = sample_foreground(get_mask(pa), a[lo:hi]) | sample_foreground(get_mask(pb), b[lo:hi])
        if admission_cache is None and i % 250 == 0:
            print(json.dumps(dict(stage='main_cut_source_mask', cut=args.cut, pairs=i+1, total=len(pairs))), flush=True)
    source_masks.clear()
    if admission_cache is not None:
        with np.load(admission_cache, allow_pickle=False) as z:
            for key, expected in [('source_uv_a',a),('source_uv_b',b),('endpoint_index_a',ia),
                                  ('endpoint_index_b',ib),('source_original_sigma_xy',sigma),
                                  ('pairs',pairs),('offsets',offsets),('original_row_index',np.arange(len(a)))]:
                if not np.array_equal(z[key],expected):
                    raise ValueError('Existing source-person cache source array changed: '+key)
            static = z['static_factor_active'].copy()
            if static.dtype != np.dtype(bool) or static.shape != (len(a),):
                raise ValueError('Existing source-person admission mask layout changed')
            if not np.array_equal(z['active_original_row_index'],np.flatnonzero(static)):
                raise ValueError('Existing source-person active original row order changed')
            foreground = ~static
    active = np.flatnonzero(~foreground)
    active_lookup = {int(raw): i for i, raw in enumerate(active)}
    scene = read(scene_path)
    planes = np.zeros((len(active), 4))
    finite_records = []
    finite_path = OLD/'z190_main_nonzero_mount_full_source_v1/qualified_finite_source_rows.json'
    for raw in read(finite_path):
        if raw['cut'] != args.cut:
            continue
        original_row = int(offsets[raw['original_pair_index']]) + raw['original_local_index']
        if not np.array_equal(a[original_row], raw['xy0']) or not np.array_equal(b[original_row], raw['xy1']):
            raise ValueError('Previously reviewed finite material source UV changed')
        if original_row not in active_lookup:
            raise ValueError('Previously reviewed finite material contradicts person-envelope exclusion')
        j = active_lookup[original_row]
        if raw['surface_kind'] == 'ice_plane':
            planes[j] = [0., 0., 1., 0.]
        elif raw['surface_kind'] == 'finite_board_face':
            start, end = np.asarray(raw['extent_start_xy']), np.asarray(raw['extent_end_xy'])
            delta = end-start
            normal = np.r_[delta[1], -delta[0], 0.]
            normal /= np.linalg.norm(normal)
            planes[j] = np.r_[normal, -normal@np.r_[start, 0.]]
        else:
            raise ValueError('Unsupported reviewed finite material')
        finite_records.append(dict(raw, global_bundle_index=j, original_cut_row_index=original_row))
    old_finite_count = len(finite_records)
    if args.new_ice_motion_source:
        motion_packet = read(args.new_ice_motion_source)
        if motion_packet['source_graph'] != ref(ray_path):
            raise ValueError('New material review must bind the unchanged original graph')
        if motion_packet['selection_uses_model_or_camera'] or motion_packet['source_raw_rows_changed']:
            raise ValueError('Material qualification must use original source pixels only')
        for raw in motion_packet['records']:
            original_row = int(raw['global_index'])
            pair = int(raw['pair_index'])
            if original_row != int(offsets[pair])+int(raw['local_index']):
                raise ValueError('New material source row identity changed')
            if list(map(int,pairs[pair])) != raw['pair_pts']:
                raise ValueError('New material exposure identity changed')
            if not np.array_equal(a[original_row],raw['source_uv']) or not np.array_equal(b[original_row],raw['target_uv']):
                raise ValueError('New material original source UV changed')
            if raw['surface_id'] != 'ice_z0' or original_row not in active_lookup:
                raise ValueError('New material must be verified ice outside the source person envelope')
            j = active_lookup[original_row]
            if np.any(planes[j]):
                raise ValueError('New material review duplicates a previous finite constraint')
            planes[j] = [0.,0.,1.,0.]
            finite_records.append(dict(raw,global_bundle_index=j,original_cut_row_index=original_row,
                                       origin=str(args.new_ice_motion_source),surface_kind='ice_plane'))
    observed = {}
    for r in authority['fit_rows']:
        p = aliases[int(r['pts'])]
        if p in index:
            observed.setdefault(p, []).append(convert_row(r))
    groups = [(p, observed[p]) for p in stamps if p in observed]
    additional = []
    if args.new_ice_source:
        packet = read(args.new_ice_source)
        if packet['source_video_sha256'] != SOURCE_SHA:
            raise ValueError('New ice source changed source/event')
        source_only = bool(packet.get('source_only_interval_reading')) or (
            'source_reading_uses_camera_or_model' in packet and not packet['source_reading_uses_camera_or_model'])
        if packet['scene_sha256'] != HARD_SCENE_SHA or not source_only:
            raise ValueError('New ice source must bind the hard scene and source-only reading')
        if packet.get('source_profile_reading_uses_camera_or_model', False):
            raise ValueError('Ice observations must come from native source pixels')
        for raw in packet['groups']:
            if not raw['id'].endswith('_outer') or raw['point_correspondence']:
                raise ValueError('Expected a reviewed hard-scene external ring boundary')
            feature = next(f for f in scene['ice_features'] if f['id'] == raw['id'].removesuffix('_outer'))
            center = np.asarray(feature['center_xyz'])
            radius = feature['radius']
            row = dict(raw)
            row['pts'] = aliases[int(raw['source_native_pts'])]
            if row['pts'] not in index:
                raise ValueError('New ice source crosses cut')
            row['normal'] = np.asarray(raw['source_normal_uv'], float)
            row['uv'] = np.asarray(raw['uv'], float)
            row['world'] = np.asarray(raw['world'], float)
            row['normal_interval'] = np.asarray(raw['normal_interval_px'], float)
            if np.max(np.abs(np.linalg.norm(row['world'][:,:2]-center[:2],axis=1)-radius))>1e-7 or np.any(row['world'][:,2] != 0.):
                raise ValueError('Source curve must equal the current hard external ring')
            # Retain the source reader's stated engineering scale at each
            # spatial sample; the measured blur support is applied separately.
            # Do not multiply it by sqrt(N), which suppressed this ice support.
            row['sigma'] = np.full(len(row['uv']),float(raw['sigma_source_choice_px']))
            row['origin'] = str(args.new_ice_source)
            additional.append(row)
            if raw['split'] == 'fit':
                observed.setdefault(row['pts'], []).append(row)
            elif raw['split'] != 'held':
                raise ValueError('New source must have explicit fit/held role')
        groups = [(p, observed[p]) for p in stamps if p in observed]
    absolutes = np.concatenate([np.concatenate([np.repeat(r['sigma'], 2) for r in rr]) for _, rr in groups])
    base = SimpleNamespace(stamps=stamps, index=index, a=a[active], b=b[active], ia=ia[active], ib=ib[active],
                           motion_sigma=sigma[active].ravel(), abs_groups=groups,
                           native_residual_sigma=np.r_[sigma[active].ravel(), absolutes])
    bundle = FixedCutBundle(base, cal, SchurMotionKernel(minimum_source_depth_scene_units=args.minimum_unknown_depth_m/SCENE_UNIT_M),
        reference_pts=stamps[0], initial_reference_optical_height_scene=model.camera25(views[stamps[0]])[2],
        scene=scene, finite_planes=planes, finite_records=finite_records)
    q = np.r_[bundle.initial_shared(), np.concatenate([np.r_[views[p].view.rvec_world_to_camera,
                   views[p].view.log_f_ratio+np.log(1920.)] for p in stamps])]
    np.savez_compressed(args.output/'original_motion_source_admission.npz', source_uv_a=a, source_uv_b=b,
        endpoint_index_a=ia, endpoint_index_b=ib, source_original_sigma_xy=sigma,
        original_row_index=np.arange(len(a)), active_original_row_index=active,
        static_factor_active=~foreground, pairs=pairs, offsets=offsets, active_finite_planes=planes)
    write(args.output/'qualified_finite_source_rows.json', finite_records)
    np.savez_compressed(args.output/'initial_state.npz', q=q, canonical_pts=stamps)
    manifest = dict(schema='rinkspace.current-main-cut-fixed-mount-depth-profile-recovery.v1', cut=args.cut,
        current_complete=ref(args.current), scene=ref(scene_path), source_sha256=SOURCE_SHA,
        motion=ref(ray_path), source_authority=ref(AUTHORITY), actual_exposures=len(stamps),
        original_motion_rows=len(a), source_foreground_rows=int(foreground.sum()), active_motion_rows=len(active),
        original_fit_rows=sum(len(rr) for _, rr in groups), fit_identity_and_UV_and_sigma_unchanged=True,
        previously_reviewed_finite_material_rows=old_finite_count, finite_source_input=ref(finite_path),
        additional_verified_ice_material_rows=len(finite_records)-old_finite_count,
        new_ice_motion_source=None if args.new_ice_motion_source is None else ref(args.new_ice_motion_source),
        new_ice_source=None if args.new_ice_source is None else ref(args.new_ice_source),
        additional_source_fit_points=sum(len(r['uv']) for r in additional if r['split']=='fit'),
        additional_source_held_points=sum(len(r['uv']) for r in additional if r['split']=='held'),
        held_pending_future_not_fit=True, minimum_unknown_depth_m=args.minimum_unknown_depth_m,
        depth_bound_role='explicit engineering near-optical-depth exclusion; not a measured material depth',
        mask_role='existing exact-native conservative person envelope, not static rink; no material plane assigned',
        source_person_admission_cache=cache_binding,
        shared_mechanical_optical_parameters_fixed=True, hard_scene_fixed=True,
        all_other_cuts_bitexact=True, entrypoint=ref(__file__), accuracy_accepted=False)
    write(args.output/'input_manifest.json', manifest)
    # The source depth is conditional on the camera, never a new observed XYZ.
    state = conditional_depth_state_arrays(bundle,q)
    np.savez_compressed(args.output/'current_unbounded_conditional_depth.npz', **state, active_original_row_index=active)
    print(json.dumps(dict(stage='main_cut_prepared', **{k:manifest[k] for k in ('cut','actual_exposures','original_motion_rows','source_foreground_rows','original_fit_rows')})), flush=True)
    return bundle, q, model, native, aliases, views, cal, authority, manifest, additional


def source_report(model, views, bundle, authority, aliases, additional=()):
    records = []
    roles = [(role, authority[key]) for role, key in [('fit','fit_rows'), ('held','held_rows'), ('pending','pending_rows')]]
    roles.append(('future_unfitted_identity_check', [r for g in authority['untouched_future_groups'] for r in g['rows']]))
    roles.append(('held_ad2_41300', read(Path(read(MANIFEST)['new41300held']['path']))))
    roles.extend((role,[r for r in additional if r['split']==split]) for role,split in [('new_ice_fit','fit'),('new_ice_held','held')])
    for role, rows in roles:
        for raw in rows:
            p = aliases[int(raw['pts'])]
            if p not in bundle.index:
                continue
            row = convert_row(raw)
            records.append(dict(role=role, pts=int(raw['pts']), id=raw['id'],
                source_uv_sha256=hashlib.sha256(np.asarray(raw['uv'],float).tobytes()).hexdigest(),
                error=source_errors(model.camera25(views[p]), [row])[0]))
    return records


def run(args):
    args.output = args.output.resolve()
    if not args.output.is_relative_to(Path('D:/CodexData').resolve()):
        raise ValueError('Bulk results stay outside OneDrive')
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    enable_projection_backend()
    bundle,q,model,native,aliases,views,cal,authority,manifest,additional = prepare(args)
    write(args.output/'baseline_source_errors.json', source_report(model, views, bundle, authority, aliases,additional))
    if not args.solve:
        return
    fitted, history, seconds = fixed_mount_rf_solve(bundle, q, args.output)
    if not np.array_equal(fitted[:4], q[:4]):
        raise ValueError('Shared fixed installation changed')
    updated = dict(views)
    updated.update({p: CanonicalView(views[p].installation_id, bundle.view(fitted,p)) for p in bundle.stamps})
    path = args.output/'complete_camera_cut_recovery_UNACCEPTED.npz'
    model.save_npz(path,native,aliases,updated,source_sha256=SOURCE_SHA,time_base='1/15360')
    saved, nn, aa, vv = PhysicalCameraModel.load_npz(path)
    unchanged = [p for p in views if p not in bundle.index]
    if aa != aliases or not np.array_equal(nn,native) or not all(np.array_equal(saved.camera25(vv[p]), model.camera25(views[p])) for p in unchanged):
        raise ValueError('An untouched cut or source alias changed')
    write(args.output/'candidate_source_errors.json', source_report(saved, vv, bundle, authority, aliases,additional))
    write(args.output/'report.json',dict(manifest,candidate=ref(path),solve_seconds=seconds,iterations=len(history),
        unchanged_actual_exposures=len(unchanged),disposition='source_visual_review_required_UNACCEPTED'))
    print(json.dumps(dict(stage='main_cut_recovery_saved',path=str(path),seconds=seconds)),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current',type=Path,default=CURRENT)
    parser.add_argument('--cut',type=int,choices=(1,7,9),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--minimum-unknown-depth-m',type=float,default=1.)
    parser.add_argument('--new-ice-source',type=Path)
    parser.add_argument('--new-ice-motion-source',type=Path)
    parser.add_argument('--solve',action='store_true')
    run(parser.parse_args())
