"""Read-only actual saved B1/Ad4 update decomposition; stdout, no J or files.

The four crossed camera/world values use original source operators.  Single
coordinate replacements explain this saved update only, not identifiability.
"""
from pathlib import Path
from dataclasses import fields, replace
import argparse, hashlib, json, os, sys
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
from calibration.solver_runtime import install, configure_native_runtime
install(); configure_native_runtime(stage='readonly_B1_Ad4_saved_state', native_threads=1)
from rinkspace.vision_camera_probe import FrozenCameraSourceProbe
from structured_programme_source_candidate.current_ice_material_scene import install_source_ice_material_view
from z190_physical_camera import ViewState


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def emit(stage, **value):
    print(json.dumps(dict(stage=stage, **value), allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()), flush=True)


def selected(row):
    c = row['curve']
    return ((c.get('kind') == 'current_ice_segment' and c.get('feature_id') == 'B1')
        or (c.get('kind') == 'advertisement_boundary' and c.get('advertisement_number') == 4))


def summary(raw):
    norms = np.linalg.norm(raw, axis=1)
    return dict(mean=float(norms.mean()), median=float(np.median(norms)),
        p90=float(np.quantile(norms, .90)), maximum=float(norms.max()),
        mean_xy=raw.mean(axis=0))


def response(raw, change):
    norms = np.linalg.norm(raw, axis=1)
    direction = np.divide(raw, norms[:, None], out=np.zeros_like(raw), where=norms[:, None] > 0.)
    return dict(mean_norm_error_delta=float((np.linalg.norm(raw + change, axis=1) - norms).mean()),
        first_order_mean_error_delta=float(np.einsum('ni,ni->n', direction, change).mean()),
        mean_vector_change_xy=change.mean(axis=0),
        maximum_vector_change=float(np.linalg.norm(change, axis=1).max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parent', default='091f7278c5c601208c168fdbf9b560f227503d2d2dfd6fd0f3a10a7568fb16c8')
    parser.add_argument('--child', default='f9d83084051c0c5498ed01646f2fd4b8ef64dd15e46f8ec471eb5fba5ec97496')
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--pts', type=int, nargs='+', help='Existing actual exposures only')
    args = parser.parse_args()
    options = read(Path(str(args.run) + '.bootstrap/input_config.json'))['source_options']
    probes = []
    for role, token in [('initial_states', args.parent), ('accepted_states', args.child)]:
        directory = args.run / role / token
        identity = read(directory / 'state_identity.json')
        path = directory / 'complete_state_UNACCEPTED.npz'
        if sha(path) != identity['state_npz_sha256'] or sha(identity['camera_path']) != identity['camera_sha256']:
            raise ValueError('Actual saved state/camera file identity changed')
        probe = FrozenCameraSourceProbe(path, identity['camera_path'], options['--world-init'], options)
        if probe.provenance['q_sha256'] != token:
            raise ValueError('Actual q identity changed')
        install_source_ice_material_view(probe.binder)
        probes.append(probe)
        emit('authenticated_state', q=token, public_camera_bitexact=True)
    p0, p1 = probes
    if p0.binder.current.parameter_names != p1.binder.current.parameter_names:
        raise ValueError('The physical parameter chart changed')
    indices = p0.native['native_ice'].world_indices
    if not np.array_equal(indices, p0.native['native_board'].world_indices):
        raise ValueError('Native geometry owners differ')
    q0 = p0.q.copy(); q1 = q0.copy(); q1[indices] = p1.world
    changes = [{'world_index': i, 'evaluation_index': int(indices[i]), 'name': name,
                'unit': p0.binder.current.parameter_units[i], 'before': p0.world[i],
                'after': p1.world[i], 'delta': p1.world[i] - p0.world[i]}
        for i, name in enumerate(p0.binder.current.parameter_names) if p0.world[i] != p1.world[i]]
    emit('actual_world_changes', changes=changes if not args.compact else [r for r in changes
        if r['world_index'] in (6, 14, 44, 45, 46, 48, 49, 54, 55, 56, 57)])
    rows = []
    for family, factor in p0.native.items():
        for role, source in [('FIT', factor.rows), ('HELD', factor.held)]:
            for row in source:
                if selected(row):
                    rows.append((family, factor, role, row))
    if args.compact:
        dense = [x for x in rows if x[3]['id'].startswith('assistant_same_print_actual_')]
        if dense:
            keep = {min(dense, key=lambda x: x[3]['actual_pts'])[3]['id'],
                    max(dense, key=lambda x: x[3]['actual_pts'])[3]['id']}
            rows = [x for x in rows if not x[3]['id'].startswith('assistant_same_print_actual_') or x[3]['id'] in keep]
            emit('compact_dense_interval', original_rows=len(dense), displayed_endpoint_ids=sorted(keep),
                source_rows_or_FIT_objective_changed=False)
    for family, factor, role, original in rows:
        if args.pts is not None and int(original['actual_pts']) not in args.pts:
            continue
        row = dict(original)
        row.pop('_source_optical_support', None)
        at = int(row['actual_pts']); i0, i1 = p0.actual_index[at], p1.actual_index[at]
        cameras = [p0.camera['camera25'][i0], p1.camera['camera25'][i1]]
        def evaluate(q, camera):
            kwargs = dict(cameras={at: camera})
            if family == 'native_board':
                kwargs['apply_optical_support'] = False
            return factor._residual(q, [row], **kwargs).reshape(-1, 2) * np.asarray(row['sigma'])[:, None]
        values = {(g, c): evaluate([q0, q1][g], cameras[c]) for g in (0, 1) for c in (0, 1)}
        nominal = values[0, 0]
        geo = .5 * (values[1, 0] - nominal + values[1, 1] - values[0, 1])
        cam = .5 * (values[0, 1] - nominal + values[1, 1] - values[1, 0])
        per_world = []; sum_world = np.zeros_like(nominal)
        coordinate_fraction = 1e-3
        for change in changes:
            q = q0.copy(); q[change['evaluation_index']] += coordinate_fraction * change['delta']
            delta = (evaluate(q, cameras[0]) - nominal) / coordinate_fraction
            if np.any(delta):
                sum_world += delta
                per_world.append(dict(name=change['name'], delta=change['delta'], **response(nominal, delta)))
        iid = str(p0.camera['installation_id'][i0])
        if str(p1.camera['installation_id'][i1]) != iid:
            raise ValueError('A physical installation changed identity')
        cal0, cal1 = p0.calibrations[iid], p1.calibrations[iid]
        view0, view1 = p0.camera['view_state4'][i0], p1.camera['view_state4'][i1]
        offset0, offset1 = p0.camera['effective_pp_offset_px'][i0], p1.camera['effective_pp_offset_px'][i1]
        per_camera = []; sum_camera = np.zeros_like(nominal)
        for axis in range(4):
            if view0[axis] == view1[axis]:
                continue
            view = view0.copy(); view[axis] += coordinate_fraction * (view1[axis] - view0[axis])
            camera = cal0.camera25(ViewState.from_state4(view, effective_pp_offset_px=offset0))
            delta = (evaluate(q0, camera) - nominal) / coordinate_fraction; sum_camera += delta
            per_camera.append(dict(name=('RF4_rvec_x', 'RF4_rvec_y', 'RF4_rvec_z', 'RF4_logf')[axis],
                delta=view1[axis] - view0[axis], **response(nominal, delta)))
        for field in ('pivot_xyz', 'lever_xyz', 'principal_x_fraction', 'principal_y_fraction', 'k1', 'k2', 'axial', 'tangential_p1_p2'):
            old, new = getattr(cal0, field), getattr(cal1, field)
            if len(old) != len(new):
                raise ValueError('Calibration chart changed')
            for axis in range(len(old)):
                if old[axis] == new[axis]:
                    continue
                replacement = list(old); replacement[axis] += coordinate_fraction * (new[axis] - old[axis])
                cal = replace(cal0, **{field: tuple(replacement)})
                camera = cal.camera25(ViewState.from_state4(view0, effective_pp_offset_px=offset0))
                delta = (evaluate(q0, camera) - nominal) / coordinate_fraction; sum_camera += delta
                per_camera.append(dict(name=field + ':' + str(axis), delta=new[axis] - old[axis],
                    **response(nominal, delta)))
        if not np.array_equal(offset0, offset1):
            raise ValueError('Unaccounted effective pixel offset changed')
        sort = lambda items: sorted(items, key=lambda x: abs(x['first_order_mean_error_delta']), reverse=True)
        if args.compact:
            compact = lambda items: [[r['name'], r['first_order_mean_error_delta']] for r in sort(items)[:5]]
            emit('compact_row', id=row['id'], role=role, pts=at, installation=iid,
                edge=row['curve'].get('edge', 'B1'), points=len(row['uv']),
                original_sigma_px=row['sigma'], coordinate_fraction=coordinate_fraction,
                mean_00_01_10_11=[summary(values[g,c])['mean'] for g,c in values],
                p90_00_01_10_11=[summary(values[g,c])['p90'] for g,c in values],
                before_after_mean_xy=[nominal.mean(axis=0), values[1,1].mean(axis=0)],
                geometry_camera_first_order_mean_error=[response(nominal, geo)['first_order_mean_error_delta'],
                    response(nominal, cam)['first_order_mean_error_delta']],
                world=compact(per_world), camera=compact(per_camera),
                additive_error_world_camera=[float(abs(sum_world-(values[1,0]-nominal)).max()),
                    float(abs(sum_camera-(values[0,1]-nominal)).max())])
            continue
        emit('row', family=family, role=role, id=row['id'], actual_pts=at, source_seconds=at/15360.,
            physical_curve=row['curve'], installation=iid, source_points=len(row['uv']),
            original_sigma_px=row['sigma'], crossed_raw_px={str(g)+str(c): summary(values[g,c]) for g,c in values},
            exact_geometry_response=response(nominal, geo), exact_camera_response=response(nominal, cam),
            exact_component_sum_error=float(abs(geo+cam-(values[1,1]-nominal)).max()),
            original_point_uv=row['uv'], nominal_raw_vectors=nominal, final_raw_vectors=values[1,1],
            top_world=sort(per_world)[:6], top_camera=sort(per_camera)[:7],
            coordinate_tangent_stencil_fraction_of_saved_update=coordinate_fraction,
            all_world_replacement_additivity_error=float(abs(sum_world-(values[1,0]-nominal)).max()),
            all_camera_replacement_additivity_error=float(abs(sum_camera-(values[0,1]-nominal)).max()),
            crossed_states_and_coordinate_replacements_are_diagnostics_not_candidates=True,
            sigma_is_original_engineering_width_not_metric_accuracy=True)


if __name__ == '__main__':
    main()
