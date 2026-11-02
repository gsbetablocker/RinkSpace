"""Bind measured print edges to finite source-supported arcs, without changing UV.

The original whole contours remain the geometry/drawing authority. Child paths
contain exact contiguous parent vertices and share the same print parameters.
Photographic search queries identify arcs; they are not fixed material points.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import collections
import json
import numpy as np
import cv2

import prepare_native_multicamera as prepared
from prepare_spatial_refinement import _write_prepared
from build_ad3_completion import layout_without_content_identity
from audit_ad4_source_correspondence import closed_vertices, inverse_photo, source_station
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.print_assets import evaluate_print_path
from rinkspace.session.gpu_geometry import NativeSceneGPU
from rinkspace.path_policy import guard_project_write


def split_arcs(path, epsilon):
    raw = closed_vertices(path['shape_provenance']['source_native_boundary'], path['closed'])
    material = closed_vertices(path['material_uv'], path['closed'])
    if len(raw) != len(material) or not path['closed']:
        raise ValueError('Expected matching closed source/template vertex arrays')
    count = len(raw)-1
    corners = cv2.approxPolyDP(raw[:-1].astype(np.float32), epsilon, True).reshape(-1, 2)
    cuts = sorted({int(np.argmin(np.linalg.norm(raw[:-1]-q, axis=1))) for q in corners})
    if len(cuts) < 3:
        raise ValueError('Source contour has insufficient corners for branch support')
    result = []
    for i, start in enumerate(cuts):
        stop = cuts[(i+1) % len(cuts)]
        if stop <= start:
            stop += count
        indices = np.arange(start, stop+1) % count
        identity = path['id'] + '.source_arc_' + str(i)
        child = dict(id=identity, closed=False, role='printed_ink_open_boundary_arc',
                     material_uv=material[indices].tolist(),
                     source_support_parent=path['id'],
                     source_support=dict(parent_vertex_indices=indices.tolist(),
                         parent_segment_begin=start, parent_segment_stop_unwrapped=stop,
                         parent_segment_count=count, source_corner_partition_tolerance_px=epsilon,
                         partition_only_not_geometry_simplification=True,
                         source_original_vertices=raw[indices].tolist(),
                         original_role=path['role'], original_hole_parent=path.get('parent_contour_key')))
        result.append(child)
    return result


def containing_arc(arcs, position):
    for arc in arcs:
        support = arc['source_support']
        start, stop, count = (support[k] for k in
            ('parent_segment_begin', 'parent_segment_stop_unwrapped', 'parent_segment_count'))
        unwrapped = position if position >= start else position+count
        if start <= unwrapped < stop or abs(unwrapped-stop) < 1e-10:
            return arc, float(np.clip((unwrapped-start)/(stop-start), 0., 1.))
    raise ValueError('Source query lies outside all parent arc supports')


def main(a):
    root = a.prepared.resolve(); out = guard_project_write(a.output.resolve())
    for target in (out, a.asset_output, a.scene_output):
        guard_project_write(target.resolve())
        if target.exists():
            raise FileExistsError(target)
    if not np.isfinite(a.corner_tolerance_px) or a.corner_tolerance_px <= 0:
        raise ValueError('Invalid source corner partition tolerance')
    manifest = prepared.read(root/'joint_manifest.json'); mapping = prepared.read(root/'mapping.json')
    points = prepared.npz(root/'point_observations.npz'); state = prepared.npz(a.state)
    old_scene_path = Path(manifest['scene_path']); scene = prepared.read(old_scene_path)
    old_model = SceneParameterModel(scene, old_scene_path, optimize_scene=True)
    prepared.validate_snapshot(state, old_model.initial_state.layout, mapping, manifest, points)
    if prepared.digest(root/'point_observations.npz') != manifest['point_input_sha256']:
        raise ValueError('Prepared point identity changed')
    dll_sha = prepared.digest(a.native_library)
    if dll_sha != str(state['native_library_provenance_sha256']):
        raise ValueError('Frozen native library changed')
    ref = next(r for r in scene['advertisement_assets'] if r['advertisement_number'] == 4)
    asset_path = (old_scene_path.parent / ref['path']).resolve(strict=True)
    if prepared.digest(asset_path) != ref['sha256']:
        raise ValueError('Parent template identity changed')
    old_asset = prepared.read(asset_path); asset = deepcopy(old_asset)
    template = next(t for t in asset['templates'] if t['id'] == ref['template_id'])
    parents = {p['id']:p for p in template['paths'] if '.ANKARA.' in p['id']}
    arcs = {name:split_arcs(p, a.corner_tolerance_px) for name,p in parents.items()}
    source = prepared.read(a.source_audit.parent/'word_reference.json')
    source_paths = {p['path_id']:p for p in source['paths']}
    for name, parent in parents.items():
        if not np.array_equal(closed_vertices(parent['shape_provenance']['source_native_boundary'], True),
                              np.asarray(source_paths[name]['pixels'])):
            raise ValueError('Source reference/template vertex order differs')
    records = {}
    for line in a.source_audit.read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        if record['pts'] in records:
            raise ValueError('Duplicate source exposure')
        records[record['pts']] = record
    ids = mapping['path_ids']; index_to_name = {ids.index('print_instance/Ad4::'+name):name for name in parents}
    selected = np.flatnonzero(np.isin(points['path_indices'], list(index_to_name)))
    by_pts = collections.defaultdict(list)
    for row in selected:
        by_pts[int(points['pts'][row])].append(int(row))
    bindings = []; unresolved = collections.Counter(); profile_refs = {}; max_query_distance = 0.
    for pt, rows in sorted(by_pts.items()):
        record = records.get(pt, {}); registration = record.get('word_registration', {})
        if registration.get('status') != 'source_pattern_registered':
            unresolved['no_source_registration'] += len(rows); continue
        profile_path = Path(registration['profile_path'])
        profile = prepared.npz(profile_path); H = np.asarray(registration['H'], dtype=float)
        lookup = collections.defaultdict(list)
        for j, (label, uv) in enumerate(zip(profile['query_path_labels'], profile['actual_source_uv'])):
            if np.isfinite(uv).all():
                lookup[(str(label), *uv)].append(j)
        profile_refs[str(pt)] = dict(path=str(profile_path), sha256=prepared.digest(profile_path),
                                    source_BGR_sha256=record['source_BGR_sha256'], H=H.tolist())
        for row in rows:
            name = index_to_name[int(points['path_indices'][row])]
            matches = lookup.get((name, *points['uv'][row]), [])
            if len(matches) != 1:
                unresolved['no_unique_exact_source_readout'] += 1; continue
            j = matches[0]
            query = inverse_photo(H, profile['search_uv'][j])
            raw = closed_vertices(parents[name]['shape_provenance']['source_native_boundary'], True)
            correspondence = source_station(raw, query, True, 0.)
            distance = correspondence['source_edge_distance_px']; max_query_distance = max(max_query_distance, distance)
            if distance > 1e-7:
                raise ValueError('Archived query is not on its declared source contour')
            position = correspondence['station']*(len(raw)-1)
            child, child_station = containing_arc(arcs[name], position)
            if points['kind'][row] != 1:
                raise ValueError('A fixed observation cannot acquire a free arc station')
            bindings.append(dict(point_row=row, source_row=int(points['source_rows'][row]), pts=pt,
                parent_path_id=name, child_path_id=child['id'], source_profile_index=j,
                source_query_uv=profile['search_uv'][j].tolist(), source_query_reference_uv=query.tolist(),
                parent_query_station=correspondence['station'], child_query_station=child_station,
                previous_numeric_station=float(state['stations'][row]), measured_uv=points['uv'][row].tolist()))
    if not bindings:
        raise ValueError('No exact source-supported arc correspondences')
    used = {b['child_path_id'] for b in bindings}
    children = [c for group in arcs.values() for c in group if c['id'] in used]
    internal = template['internal_geometry']
    for child in children:
        parent_binding = internal['path_bindings'][child['source_support_parent']]
        if parent_binding['kind'] != 'shape_group':
            raise ValueError('Only existing shared print-shape paths may gain finite aliases')
        internal['path_bindings'][child['id']] = deepcopy(parent_binding)
        group = next(g for g in internal['groups'] if g['id'] == parent_binding['group_id'])
        group['path_ids'].append(child['id'])
        if internal.get('layout_policy'):
            declaration = next(g for g in internal['layout_policy']['groups']
                               if g['id'] == group['component_identity'])
            declaration['path_ids'].append(child['id'])
    template['paths'].extend(children)
    template['finite_source_support_revision'] = dict(source_audit=str(a.source_audit),
        source_audit_sha256=prepared.digest(a.source_audit), parent_asset_sha256=ref['sha256'],
        corner_partition_tolerance_px=a.corner_tolerance_px, original_whole_paths_preserved=True,
        exact_contiguous_parent_vertices=True, no_new_geometry_parameters=True,
        source_queries_select_support_only=True, actual_source_UV_unchanged=True)
    prepared.write(a.asset_output, asset)
    ref.update(path=str(a.asset_output.resolve()), sha256=prepared.digest(a.asset_output))
    prepared.write(a.scene_output, scene)
    model = SceneParameterModel(scene, a.scene_output, optimize_scene=True)
    if layout_without_content_identity(model.initial_state.layout) != layout_without_content_identity(old_model.initial_state.layout):
        raise ValueError('Finite support aliases changed the joint parameter layout')
    core = NativeCore(a.native_library)
    parameters = ParameterState(model.initial_state.layout, state['scene_values'])
    old_parameters = ParameterState(old_model.initial_state.layout, state['scene_values'])
    board = parameters.apply_scene(scene)['board']; geometry_error = 0.; derivative_error = 0.
    for child in children:
        support = child['source_support']; u = np.linspace(0., 1., 7)
        parent_position = (support['parent_segment_begin'] + u*(support['parent_segment_stop_unwrapped']-support['parent_segment_begin']))
        parent_u = (parent_position % support['parent_segment_count']) / support['parent_segment_count']
        for instance in ref.get('instance_ids', [ref.get('instance_id', 'print_instance/Ad4')]):
            old_eval = evaluate_print_path(old_asset, instance, child['source_support_parent'], board, core,
                                          parent_u, parameter_state=old_parameters)
            new_eval = evaluate_print_path(asset, instance, child['id'], board, core, u, parameter_state=parameters)
            geometry_error = max(geometry_error, float(np.max(abs(old_eval['xyz']-new_eval['xyz']))))
            for block in old_eval['jacobians_by_block']:
                derivative_error = max(derivative_error, float(np.max(abs(old_eval['jacobians_by_block'][block]-new_eval['jacobians_by_block'][block]))))
    if geometry_error > 1e-12 or derivative_error > 1e-10:
        raise ValueError('Finite child support changed physical geometry or its shared derivatives')
    mapped = deepcopy(mapping); mapped['path_ids'] += ['print_instance/Ad4::'+c['id'] for c in children]
    path_index = {name:i for i,name in enumerate(mapped['path_ids'])}
    updated_points = {k:v.copy() for k,v in points.items()}; numeric = state['stations'].copy()
    for b in bindings:
        row = b['point_row']; updated_points['path_indices'][row] = path_index['print_instance/Ad4::'+b['child_path_id']]
        updated_points['stations'][row] = b['child_query_station']; numeric[row] = b['child_query_station']
    native = NativeSceneGPU(a.scene_output, core, parameter_state=parameters)
    if not set(mapped['path_ids']) <= set(native.paths):
        raise ValueError('Prepared support alias not available in native geometry')
    mapped['scene_revision'] = native.scene_revision
    target = deepcopy(manifest); target.update(scene_path=str(a.scene_output.resolve()), scene_file_sha256=prepared.digest(a.scene_output),
        scene_revision=native.scene_revision, native_library=str(a.native_library.resolve()), native_library_sha256=dll_sha)
    receipt = dict(schema='rinkspace.finite_print_source_support.v1',
        parent_prepared=str(root), parent_state=dict(path=str(a.state),sha256=prepared.digest(a.state)),
        parent_point_input_sha256=manifest['point_input_sha256'], rows=len(points['pts']), changed_rows=len(bindings),
        retained_legacy_unmodified=dict(unresolved), source_frames=len({b['pts'] for b in bindings}),
        new_finite_support_aliases=len(children), no_new_geometry_parameters=True, no_new_observations=True,
        source_UV_PTS_observation_identity_unchanged=True, parent_full_contours_preserved=True,
        whole_programme_rig_scene_motion_unchanged=True, max_source_query_inverse_distance_px=max_query_distance,
        max_native_parent_child_xyz_difference=geometry_error, max_shared_derivative_difference=derivative_error,
        source_profile_references=profile_refs, bindings=bindings, fit_executed=False,
        interpretation='Observed source transitions keep their UV; source-photograph queries choose a finite contour arc. Free station remains within that arc, not fixed to the query.')
    updated, mapped = _write_prepared(out, updated_points, numeric, manifest['shots'], state, target, mapped, model, receipt)
    if 'spatial_refinement_preparation' in manifest:
        updated['spatial_refinement_preparation'] = manifest['spatial_refinement_preparation']
    else:
        updated.pop('spatial_refinement_preparation', None)
    updated['finite_print_source_support'] = {k:v for k,v in receipt.items() if k not in ('bindings','source_profile_references')}
    updated['finite_print_source_support']['binding_receipt'] = str(out/'source_support_bindings.json')
    prepared.write(out/'joint_manifest.json', updated); prepared.write(out/'source_support_bindings.json', receipt)
    saved = prepared.npz(out/'initial_pixels.npz')
    prepared.validate_snapshot(saved, model.initial_state.layout, mapped, updated, updated_points)
    for key in ('uv','pts','source_rows','kind','observation_ids'):
        if not np.array_equal(points[key], updated_points[key]):
            raise ValueError('Source measurement changed: '+key)
    for key in ('rig','controls','scene_values','board','layout'):
        if not np.array_equal(state[key], saved[key]):
            raise ValueError('Global numeric state changed: '+key)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('bindings','source_profile_references')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','state','native-library','source-audit','asset-output','scene-output','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--corner-tolerance-px', type=float, default=1.25,
                   help='Source-raster corner partition only; original geometry vertices remain exact')
    main(p.parse_args())
