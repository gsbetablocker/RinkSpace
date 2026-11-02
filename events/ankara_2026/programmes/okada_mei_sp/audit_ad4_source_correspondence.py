"""CPU-only Ad4 audit: source-photograph correspondence versus saved fit station.

Photographic H navigates an inferred source-edge correspondence; it is not a
pixel measurement or a material-tracking certificate. Actual UV stays untouched.
No optimizer, GPU resident problem, or station update is created by this script.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import numpy as np

from rinkspace.core.native import NativeCore
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.domain.parameters import ParameterState, SceneParameterModel
from rinkspace.domain.print_assets import evaluate_print_path
from rinkspace.path_policy import guard_project_write
from export_native_six_review import _manifest_for_library, _verify_native_library_binding

DEFAULT_PTS = (2388480, 2406400, 2414080)
PREFIX = 'print_instance/Ad4::'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=digest(path))


def closed_vertices(points, closed):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('Invalid finite source/template polyline')
    if closed and not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    return points


def inverse_photo(H, uv):
    value = np.linalg.inv(np.asarray(H, dtype=float)) @ np.r_[uv, 1.]
    if not np.isfinite(value).all() or abs(value[2]) <= np.finfo(float).tiny:
        raise ValueError('Photographic inverse has no finite source coordinate')
    return value[:2] / value[2]


def source_station(vertices, point, closed, local_window):
    """Nearest source edge plus a competing, nonlocal source-arc diagnostic."""
    delta = np.diff(vertices, axis=0)
    length2 = np.sum(delta * delta, axis=1)
    fraction = np.clip(np.divide(np.sum((point-vertices[:-1])*delta, axis=1), length2,
                                  out=np.zeros_like(length2), where=length2 > 0), 0., 1.)
    projected = vertices[:-1] + fraction[:, None]*delta
    distance = np.linalg.norm(projected-point, axis=1)
    distance[length2 == 0] = np.inf
    segment = int(np.argmin(distance))
    if not np.isfinite(distance[segment]):
        raise ValueError('Source polyline has no nondegenerate edge')
    lengths = np.sqrt(length2)
    arc = np.r_[0., np.cumsum(lengths)]
    along = arc[:-1] + fraction*lengths
    separation = np.abs(along-along[segment])
    if closed:
        separation = np.minimum(separation, arc[-1]-separation)
    alternatives = np.flatnonzero((separation > local_window) & np.isfinite(distance))
    other = None if not len(alternatives) else int(alternatives[np.argmin(distance[alternatives])])
    return dict(station=float((segment+fraction[segment])/(len(vertices)-1)),
                source_segment=segment, source_segment_fraction=float(fraction[segment]),
                source_edge_projection=projected[segment].tolist(),
                source_edge_distance_px=float(distance[segment]),
                competing_nonlocal_source_segment=other,
                competing_nonlocal_source_distance_px=None if other is None else float(distance[other]),
                competing_nonlocal_source_margin_px=None if other is None else float(distance[other]-distance[segment]),
                competing_source_arc_exclusion_window_px=float(local_window),
                competing_edge_margin_is_diagnostic_not_an_admission_threshold=True)


def error_summary(rows, field):
    values = np.asarray([row[field] for row in rows if row.get(field) is not None], dtype=float)
    if not len(values):
        return dict(valid_rows=0)
    errors = np.linalg.norm(values, axis=1)
    return dict(valid_rows=len(values), signed_mean_px=values.mean(axis=0).tolist(),
                signed_median_px=np.median(values, axis=0).tolist(),
                norm_quantiles_px=dict(zip(('p50', 'p95', 'max'), map(float, np.percentile(errors, [50, 95, 100])))))


def summaries(rows):
    return dict(same_source_edge=error_summary(rows, 'same_source_edge_residual_px'),
                saved_fit_station=error_summary(rows, 'saved_fit_station_residual_px'))


def photo_evidence(folder, pt, audit):
    result = dict(decoded_BGR_sha256=audit['source_BGR_sha256'],
                  decoded_hash_authority='existing_source_audit_record',
                  H=np.asarray(audit['word_registration']['H'], dtype=float).tolist(),
                  H_float64_sha256=hashlib.sha256(np.asarray(audit['word_registration']['H'], dtype='<f8').tobytes()).hexdigest(),
                  correlation=audit['word_registration'].get('correlation'),
                  photographic_H_is_navigation_not_a_pixel_measurement=True)
    image_path = folder / f'{pt}_original.png'
    if image_path.is_file():
        import cv2
        image = cv2.imread(str(image_path))
        if image is None or hashlib.sha256(image.tobytes()).hexdigest() != audit['source_BGR_sha256']:
            raise ValueError('Archived original photo differs from source audit: '+str(pt))
        result['archived_original_png'] = reference(image_path)
        result['decoded_original_BGR_verified'] = True
    else:
        result['decoded_original_BGR_verified'] = False
        result['archived_original_png'] = None
    return result


def audit(args):
    output = guard_project_write(args.output)
    if output.suffix.lower() != '.json':
        output = guard_project_write(output / 'report.json')
    if output.exists():
        raise FileExistsError(output)
    if args.local_source_arc_window_px < 0:
        raise ValueError('Source arc diagnostic window cannot be negative')
    manifest_path = args.prepared / 'joint_manifest.json'
    mapping_path = args.prepared / 'mapping.json'
    point_path = args.prepared / 'point_observations.npz'
    manifest = _manifest_for_library(read(manifest_path), args.native_library)
    mapping = read(mapping_path)
    state_bytes = args.state.read_bytes()
    with np.load(io.BytesIO(state_bytes), allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    _verify_native_library_binding(manifest, state)
    point_hash = digest(point_path)
    if point_hash != manifest['point_input_sha256'] or point_hash != str(state['point_input_sha256']):
        raise ValueError('Prepared observations differ from the selected state')
    scene_path = Path(manifest['scene_path'])
    scene = read(scene_path)
    model = SceneParameterModel(scene, scene_path, optimize_scene=True)
    parameter_state = ParameterState(model.initial_state.layout, state['scene_values'])
    dependencies = json.loads(str(state['joint_numeric_dependencies_json']))
    expected = dict(scene_revision=model.initial_state.layout['scene_content_revision'],
                    layout_revision=model.initial_state.layout['layout_revision'],
                    point_input_sha256=point_hash,
                    paths_sha256=hashlib.sha256(json.dumps(mapping['path_ids'], separators=(',', ':')).encode()).hexdigest(),
                    shots=manifest['shots'], cameras=manifest['cameras'], control_dt=manifest['control_dt'],
                    first_control_time=manifest['first_control_time'], max_readout=manifest['max_readout'])
    if any(dependencies.get(key) != value for key, value in expected.items()):
        raise ValueError('State scene/layout/path/shot/motion dependencies differ from prepared input')
    if dependencies.get('camera_parameterization') != manifest.get('camera_parameterization'):
        raise ValueError('State camera parameterization differs')
    ref = next(value for value in scene['advertisement_assets'] if value['advertisement_number'] == 4)
    asset_path = (scene_path.parent / ref['path']).resolve(strict=True)
    assets = read(asset_path)
    if digest(asset_path) != ref['sha256']:
        raise ValueError('Selected Ad4 asset changed')
    template = next(value for value in assets['templates'] if value['id'] == ref['template_id'])
    all_template_paths = {value['id']: value for value in template['paths']}
    paths = {value['id']: value for value in template['paths']
             if '.ANKARA.' in value['id'] and not value.get('source_support_parent')}
    source_reference_path = args.source_reference or args.source_audit.parent / 'word_reference.json'
    atlas = read(source_reference_path)
    atlas_paths = {value['path_id']: value for value in atlas['paths']}
    source_polylines = {}
    source_images = {}
    for name, path in paths.items():
        origin = path['shape_provenance']
        raw = origin['source_native_boundary']
        vertices = closed_vertices(raw, path['closed'])
        if not np.array_equal(vertices, np.asarray(atlas_paths[name]['pixels'])):
            raise ValueError('Photographic reference and selected asset source vertex order differ: '+name)
        if len(vertices) != len(closed_vertices(path['material_uv'], path['closed'])):
            raise ValueError('Source/template station vertex counts differ: '+name)
        source_polylines[name] = vertices
        image_ref = origin['source_frame']
        if image_ref['sha256'] != atlas['image_sha256']:
            raise ValueError('Photographic reference image differs from path source: '+name)
        if image_ref['path'] not in source_images:
            actual = reference(image_ref['path'])
            if actual['sha256'] != image_ref['sha256']:
                raise ValueError('Original source-reference photo changed')
            source_images[image_ref['path']] = actual
    records = {}
    with args.source_audit.open('rb') as stream:
        for line in stream:
            record = json.loads(line)
            pt = int(record['pts'])
            if pt in records:
                raise ValueError('Duplicate source audit PTS')
            records[pt] = (record, hashlib.sha256(line).hexdigest())
    with np.load(point_path, allow_pickle=False) as observations:
        pts = observations['pts']
        path_indices = observations['path_indices']
        uv = observations['uv']
        source_rows = observations['source_rows']
    if not np.array_equal(state['source_rows'], source_rows) or state['stations'].shape != pts.shape:
        raise ValueError('State observation row/station identity differs')
    chosen = args.pts
    if args.first is not None or args.last is not None:
        if chosen is not None or args.first is None or args.last is None or args.first > args.last:
            raise ValueError('Use --pts or both ordered --first/--last source seconds')
        chosen = [pt for pt in sorted(records) if args.first <= pt/12800 <= args.last]
    if chosen is None:
        chosen = list(DEFAULT_PTS)
    chosen = sorted(set(chosen))
    if not chosen:
        raise ValueError('No selected source exposures')
    names = mapping['path_ids']
    selected_indices = {i: name[len(PREFIX):] for i, name in enumerate(names)
                        if name.startswith(PREFIX) and
                        (name[len(PREFIX):] in paths or
                         all_template_paths.get(name[len(PREFIX):], {}).get('source_support_parent') in paths)}
    timeline = ShotMotionTimeline(manifest['shots'], manifest['control_dt'], manifest['first_control_time'])
    board = parameter_state.apply_scene(scene)['board']
    image_size = mapping.get('image_size', [1920, 1080])
    if image_size != [1920, 1080]:
        raise ValueError('Source audit requires the original native 1920x1080 pixel domain')
    core = NativeCore(manifest['native_library'])
    instance = ref.get('instance_id') or ref['instance_ids'][0]
    all_rows = []
    frames = []
    for pt in chosen:
        indices = np.flatnonzero((pts == pt) & np.isin(path_indices, list(selected_indices)))
        frame = dict(pts=int(pt), source_seconds=pt/12800, observation_rows=len(indices), rows=[])
        frames.append(frame)
        source_record, record_hash = records.get(pt, (None, None))
        if source_record is None or source_record.get('word_registration', {}).get('status') != 'source_pattern_registered':
            frame['status'] = 'photographic_registration_unavailable'
            continue
        shots = [shot for shot in manifest['shots'] if shot['first_pts'] <= pt < shot['stop_pts_exclusive']]
        if len(shots) != 1:
            raise ValueError('Source PTS does not identify exactly one prepared shot')
        shot = shots[0]
        clock = ((pt-int(shot['reference_pts']))*int(shot['time_base_num'])/int(shot['time_base_den'])
                 + timeline.offset(shot))
        frame.update(status='audited', shot_id=shot['id'], motion_storage_time=clock,
                     native_time_base=[shot['time_base_num'], shot['time_base_den']],
                     source_audit_record_sha256=record_hash,
                     photographic_registration=photo_evidence(args.source_audit.parent, pt, source_record))
        H = np.asarray(source_record['word_registration']['H'], dtype=float)
        profile_path = Path(source_record['word_registration']['profile_path'])
        frame['source_profile'] = reference(profile_path)
        with np.load(profile_path, allow_pickle=False) as profiles:
            profile_uv = profiles['actual_source_uv']
            profile_paths = profiles['query_path_labels']
        for index, name in selected_indices.items():
            current = indices[path_indices[indices] == index]
            if not len(current):
                continue
            parent_name = all_template_paths[name].get('source_support_parent', name)
            rows = []
            for row in current:
                source_coordinate = inverse_photo(H, uv[row])
                correspondence = source_station(source_polylines[parent_name], source_coordinate,
                                                paths[parent_name]['closed'], args.local_source_arc_window_px)
                matches = np.flatnonzero((profile_paths == parent_name) & np.all(profile_uv == uv[row], axis=1))
                rows.append(dict(point_row=int(row), source_row=int(source_rows[row]), path_id=PREFIX+name,
                                 source_parent_path_id=PREFIX+parent_name,
                                 measured_uv=uv[row].tolist(), source_coordinate_from_inverse_photo=source_coordinate.tolist(),
                                 photographic_correspondence=correspondence,
                                 saved_fit_station=float(state['stations'][row]),
                                 exact_same_source_profile_readout_matches=matches.tolist()))
            correspondence_stations = [row['photographic_correspondence']['station'] for row in rows]
            fit_stations = [row['saved_fit_station'] for row in rows]
            xyz = np.concatenate([
                evaluate_print_path(assets, instance, parent_name, board, core, correspondence_stations,
                                    parameter_state=parameter_state, jacobian=False)['xyz'],
                evaluate_print_path(assets, instance, name, board, core, fit_stations,
                                    parameter_state=parameter_state, jacobian=False)['xyz']])
            projection = core.project_motion(state['rig'][shot['camera_index']], state['controls'],
                                             timeline.first, timeline.dt, image_size[1], clock, xyz)
            for j, row in enumerate(rows):
                for label, position in (('same_source_edge', j), ('saved_fit_station', len(rows)+j)):
                    status = int(projection['status'][position])
                    row[label+'_projection_status'] = status
                    valid = status == 0 and np.isfinite(projection['uv'][position]).all()
                    row[label+'_predicted_uv'] = projection['uv'][position].tolist() if valid else None
                    row[label+'_residual_px'] = (projection['uv'][position]-uv[row['point_row']]).tolist() if valid else None
                row['station_changed_by_audit'] = False
            frame['rows'].extend(rows)
        frame['summary'] = summaries(frame['rows'])
        frame['paths'] = {name: summaries([row for row in frame['rows'] if row['source_parent_path_id'] == PREFIX+name])
                          for name in paths if any(row['source_parent_path_id'] == PREFIX+name for row in frame['rows'])}
        all_rows.extend(frame['rows'])
    report = dict(schema='rinkspace.ad4_source_correspondence_audit.v1',
                  prepared=str(args.prepared.resolve()), selected_pts=chosen,
                  inputs=dict(manifest=reference(manifest_path), mapping=reference(mapping_path),
                              observations=dict(path=str(point_path.resolve()), sha256=point_hash),
                              state=dict(path=str(args.state.resolve()), sha256=hashlib.sha256(state_bytes).hexdigest()),
                              native_library=reference(manifest['native_library']), scene=reference(scene_path),
                              selected_ad4_asset=reference(asset_path), source_audit=reference(args.source_audit),
                              photographic_reference_atlas=reference(source_reference_path),
                              photographic_reference_images=list(source_images.values()), script=reference(__file__)),
                  frames=frames, summary=summaries(all_rows),
                  method=dict(projection='evaluate_print_path + NativeCore.project_motion, CPU native calls only',
                              correspondence='inverse photographic H of actual source UV, nearest original source polyline edge, unchanged vertex-index station',
                              ordinary_residual='selected state saved fit station; no nearest-point reoptimization in this audit',
                              conditional_on_existing_photographic_registration=True,
                              photographic_H_is_not_a_pixel_measurement_or_material_tracking_certificate=True,
                              source_measured_UV_unchanged=True, fitted_stations_unchanged=True,
                              optimizer_started=False, gpu_resident_problem_created=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), selected_frames=len(frames), rows=len(all_rows),
                          summary=report['summary']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'state', 'native-library', 'source-audit', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-reference', type=Path, help='Default: word_reference.json next to source audit')
    parser.add_argument('--pts', type=int, nargs='+', help='Original native integer PTS; default 186.6/188.0/188.6 seconds')
    parser.add_argument('--first', type=float, help='First source second; select every available source-audit PTS in range')
    parser.add_argument('--last', type=float, help='Last source second; requires --first')
    parser.add_argument('--local-source-arc-window-px', type=float, default=8.,
                        help='Exclude this local source-arc neighborhood only when reporting competing-edge margin')
    audit(parser.parse_args())
