"""Rebuild a versioned scene from a joint fit and its declared corner relation.

No optimization runs here. Source scenes are never overwritten. Legacy
schematic advertising anchors may be reattached as uncalibrated references;
user-confirmed advertising topology remains separate from physical placement.
"""
from __future__ import annotations
from rinkspace.storage import resolve_asset as _resolve_asset

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from rinkspace.calibration_companion import Scene
from rinkspace.calibration_companion.joint_scene import CORNERS, JointSceneProblem
from rinkspace.calibration_companion.structural_constraints import (
    LINE_PAIRS, hard_geometry_checks, normalize_hard_geometry_constraints,
    project_scene_to_hard_constraints,
)
from build_rink_3d_geometry import make_ice, validate, inside_convex
from build_rink_3d_data import grid_data, clip_segment, clip_polygon, area


BASE = _resolve_asset(ROOT, ROOT / 'outputs/rink_model/ankara_2026_shared/v0.14-scene3d/scene.json')
DEFAULT_OUT = _resolve_asset(ROOT, ROOT / 'outputs/rink_model/ankara_2026_shared/v0.15-joint-video')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def rounded_boundary(bounds, radii, spacing, arc_steps):
    """CCW convex boundary with each quarter-circle parameter independent."""
    x0, y0, x1, y1 = map(float, bounds)
    if x1 <= x0 or y1 <= y0 or spacing <= 0 or arc_steps < 2:
        raise ValueError('invalid boundary/grid parameters')
    if any(not np.isfinite(radii[name]) or radii[name] <= 0 for name in CORNERS):
        raise ValueError('individual corner radius must be finite and positive')
    for first, second, dimension in (
        ('upper_left', 'upper_right', x1-x0),
        ('lower_left', 'lower_right', x1-x0),
        ('upper_left', 'lower_left', y1-y0),
        ('upper_right', 'lower_right', y1-y0),
    ):
        if radii[first] + radii[second] >= dimension:
            raise ValueError('adjacent corner radii overlap or remove a straight side')
    corners = []
    for name, first in zip(CORNERS, (0., 90., 180., 270.)):
        radius = float(radii[name])
        cx = x1 - radius if 'right' in name else x0 + radius
        cy = y1 - radius if 'upper' in name else y0 + radius
        corners.append((cx, cy, radius, first))
    points = []
    for index, (cx, cy, radius, first) in enumerate(corners):
        count = max(arc_steps, int(math.ceil(math.pi * radius / (2 * spacing))))
        theta = np.deg2rad(np.linspace(first, first + 90, count + 1))
        arc = [cx, cy] + radius * np.column_stack([np.cos(theta), np.sin(theta)])
        for point in arc:
            if not points or np.linalg.norm(points[-1] - point) > 1e-9:
                points.append(point)
        nx, ny, nr, angle = corners[(index + 1) % 4]
        target = [nx + nr * np.cos(np.deg2rad(angle)), ny + nr * np.sin(np.deg2rad(angle))]
        start = points[-1]
        count = max(1, int(math.ceil(np.linalg.norm(target - start) / spacing)))
        points.extend(np.linspace(start, target, count + 1)[1:])
    if np.linalg.norm(points[-1] - points[0]) < 1e-9:
        points.pop()
    return np.asarray(points)


def boards_mesh(boundary, height, thickness):
    """Extrude the shared faceted perimeter; mitered outer offset is watertight."""
    if height <= 0 or thickness < 0:
        raise ValueError('invalid board height or thickness')
    n = len(boundary)
    positions = [np.column_stack([boundary, np.zeros(n)]),
                 np.column_stack([boundary, np.full(n, height)])]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend([[i, n + j, j], [i, n + i, n + j]])
    if thickness:
        edges = np.roll(boundary, -1, axis=0) - boundary
        tangents = edges / np.linalg.norm(edges, axis=1)[:, None]
        normals = np.column_stack([tangents[:, 1], -tangents[:, 0]])
        bisectors = normals + np.roll(normals, 1, axis=0)
        bisectors /= np.linalg.norm(bisectors, axis=1)[:, None]
        distances = thickness / np.sum(bisectors * normals, axis=1)
        outer = boundary + bisectors * distances[:, None]
        positions.extend([np.column_stack([outer, np.zeros(n)]),
                          np.column_stack([outer, np.full(n, height)])])
        for i in range(n):
            j = (i + 1) % n
            ob, ob_next, ot, ot_next = 2*n+i, 2*n+j, 3*n+i, 3*n+j
            faces.extend([[ob, ob_next, ot_next], [ob, ot_next, ot]])
            faces.extend([[n+i, ot, ot_next], [n+i, ot_next, n+j]])
            faces.extend([[i, ob_next, ob], [i, j, ob_next]])
    return dict(name='rink_boards', positions=np.vstack(positions).tolist(), triangles=faces,
        color=[.89, .90, .92], group='boards', role='vertical_boards_at_the_same_ice_perimeter',
        inner_bottom_vertex_indices=list(range(n)), inner_top_vertex_indices=list(range(n, 2*n)),
        closed_volume=bool(thickness), topology='closed_annular_solid' if thickness else 'closed_perimeter_zero_thickness_wall',
        construction='independent_corner_radii; polygon_normal_mitered_outer_offset')


def rebuild_board_surface(boundary, advertisements, height):
    edges = np.roll(boundary, -1, axis=0) - boundary
    lengths = np.linalg.norm(edges, axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    panels = []
    for index, (edge, length) in enumerate(zip(edges, lengths)):
        tangent = edge / length; n = len(boundary)
        panels.append(dict(id=f'board_panel_{index:03}', mesh_id='rink_boards',
            bottom_vertex_indices=[index, (index+1) % n], top_vertex_indices=[index+n, (index+1) % n+n],
            s_range=[float(cumulative[index]), float(cumulative[index+1])],
            tangent_xyz=[*tangent, 0.], normal_into_rink_xyz=[-tangent[1], tangent[0], 0.], z_range=[0., height]))
    updated = []
    for original in advertisements:
        advert = copy.deepcopy(original)
        anchor_xyz = advert.get('anchor_xyz')
        anchor_unknown = (anchor_xyz is None or
                          isinstance(anchor_xyz, (list, tuple)) and len(anchor_xyz) == 3 and
                          all(value is None for value in anchor_xyz))
        if (advert.get('attachment_s') is None and advert.get('board_panel_id') is None
                and anchor_unknown):
            # The ring order and content identity are known, but no physical
            # attachment has been measured. Even a historical schematic XY in
            # this record must not populate its live placement fields.
            updated.append(advert)
            continue
        if advert.get('relative_position_status') == 'user_confirmed_side_order_and_schematic_layout':
            raise ValueError(
                f"advertisement {advert['id']} has a physical attachment; "
                'user-confirmed layout needs an explicit video-backed geometry update')
        if not isinstance(anchor_xyz, (list, tuple)) or len(anchor_xyz) != 3:
            raise ValueError(f"advertisement {advert['id']} has no usable anchor_xyz")
        source_xy = advert.get('reference_schematic_xy')
        if source_xy is None:
            source_xy = anchor_xyz[:2]
        anchor = np.asarray(source_xy, float)
        if anchor.shape != (2,) or not np.isfinite(anchor).all():
            raise ValueError(f"advertisement {advert['id']} has no finite source XY anchor")
        along = np.clip(np.sum((anchor - boundary) * edges, axis=1) / lengths**2, 0., 1.)
        feet = boundary + along[:, None] * edges
        index = int(np.argmin(np.linalg.norm(feet - anchor, axis=1)))
        advert['previous_geometry_attachment'] = {
            'attachment_s': original.get('attachment_s'), 'board_panel_id': original.get('board_panel_id'),
            'anchor_xyz': original.get('anchor_xyz'), 'placement_status': original.get('placement_status')}
        advert['anchor_xyz'] = [*feet[index].tolist(), anchor_xyz[2]]
        advert['attachment_s'] = float(cumulative[index] + along[index] * lengths[index])
        advert['board_panel_id'] = panels[index]['id']
        advert['placement_status'] = 'original_schematic_XY_reprojected_to_revised_boundary_NOT_video_calibrated'
        advert['attachment_reconstruction'] = {
            'method': 'closest point to unchanged source schematic XY on revised faceted boundary',
            'reference_xy': anchor.tolist(), 'old_s_reused': False,
            'physical_advert_location_verified': False,
            'note': 'Identity and original source preserved. Changed s is a geometry representation update, not a measured advert displacement.'}
        updated.append(advert)
    surface = dict(surface_id='boards_inner_surface', mesh_id='rink_boards', panels=panels,
        local_coordinates={'s': 'CCW cumulative arc length from revised boundary[0]; not compatible with previous revision s',
                           'z': 'height above ice in unchanged normalized scene units'},
        perimeter_length=float(cumulative[-1]), closed=True)
    return surface, updated


def semantic_checks(scene, boundary):
    failures, ring_order, clipped, open_scaffolds = [], [], [], []
    for feature in scene['ice_features']:
        points = []
        if 'center_xyz' in feature:
            theta = np.linspace(0., 2*np.pi, 257)
            center = np.asarray(feature['center_xyz'])
            points.extend(center + feature['radius'] * np.column_stack([np.cos(theta), np.sin(theta), np.zeros(len(theta))]))
            rings = feature.get('rings', [])
            if any(r['radius'] > feature['radius'] + 1e-8 for r in rings):
                ring_order.append(feature['id'])
        if feature.get('open_edges'):
            # Its four rectangle vertices parameterize lane width/centerline;
            # only the three individually clipped painted sides are visible.
            for segment in feature.get('paint_segments_xyz', []):
                points.extend(segment)
            open_scaffolds.append(feature['id'])
        else:
            points.extend(feature.get('polygon_xyz', []))
        for segment in feature.get('segments_xyz', []):
            points.extend(segment)
        points = np.asarray(points).reshape(-1, 3)
        if len(points) and (not np.isfinite(points).all() or not np.allclose(points[:, 2], 0.) or
                            not inside_convex(points[:, :2], boundary).all()):
            failures.append(feature['id'])
        if feature.get('joint_export_boundary_clipped'):
            clipped.append(feature['id'])
    grid_area = sum(cell['area'] for cell in scene['ice_grid']['cells'])
    perimeter_area = area(boundary)
    return dict(all_ice_features_at_z0_inside_rink=not failures, features_outside_rink_or_invalid=failures,
        inner_rings_larger_than_outer=ring_order, feature_segments_clipped_at_revised_boundary=clipped,
        grid_cell_area_equals_shared_boundary=bool(np.isclose(grid_area, perimeter_area, atol=1e-6, rtol=1e-10)),
        grid_total_area=grid_area, boundary_area=perimeter_area,
        open_polygon_parameter_scaffolds_checked_via_painted_segments=open_scaffolds,
        physical_accuracy_verified=False,
        note='Geometry representation checks only. Out-of-bound or inconsistent circle candidates remain visible and are not silently modified.')


def clip_open_polygon_paint(feature, boundary):
    """Clip painted sides separately; keep the open rectangle's hard semantics.

    Filling/clipping the polygon would manufacture a left closing edge and may
    replace its four parameter vertices with asymmetric corner intersections.
    """
    polygon = np.asarray(feature['polygon_xyz'], float)
    open_edges = set(feature.get('open_edges', []))
    if (polygon.ndim != 2 or polygon.shape[1] != 3 or len(polygon) < 2 or
            any(not isinstance(i, int) or i < 0 or i >= len(polygon) for i in open_edges)):
        raise ValueError(f"{feature['id']}: invalid open polygon paint topology")
    painted, edge_ids, changed = [], [], False
    for index in range(len(polygon)):
        if index in open_edges:
            continue
        original = polygon[[index, (index + 1) % len(polygon)]]
        clipped = clip_segment(original[0, :2], original[1, :2], boundary)
        if clipped is None:
            changed = True
            continue
        painted.append(np.column_stack([clipped, np.zeros(2)]).tolist())
        edge_ids.append(index)
        changed |= not np.allclose(original[:, :2], clipped, atol=1e-9, rtol=0.)
    feature['paint_segments_xyz'] = painted
    feature['paint_segment_source_edge_indices'] = edge_ids
    feature['paint_export_boundary_rule'] = 'clip each painted side; preserve four parameter vertices, open edge identities and exact row centerline'
    feature['paint_segments_authoritative_for_boundary_clipped_display'] = True
    if changed:
        feature['joint_export_boundary_clipped'] = True
    return feature


def clip_hard_mirrored_features(scene, boundary):
    """Clip each left member to the common mirrored footprint, then reflect it.

    Independent corner radii make per-side clipping asymmetric, including the
    small chord approximation at analytic arc endpoints. Intersecting the real
    faceted boundary with its reflection keeps both exported members inside the
    same mesh without changing any corner radius or a pair's X distance.
    """
    cx = (scene['parameters']['bounds'][0] + scene['parameters']['bounds'][2]) / 2.
    reflected_boundary = np.asarray(boundary, float).copy()
    reflected_boundary[:, 0] = 2 * cx - reflected_boundary[:, 0]
    reflected_boundary = reflected_boundary[::-1]  # clipping boundary stays CCW
    features = {feature['id']: feature for feature in scene['ice_features']}
    protected = set()
    for left_id, right_id in LINE_PAIRS:
        left, right = features[left_id], features[right_id]
        original_left = copy.deepcopy(left['segments_xyz'])
        original_right = copy.deepcopy(right['segments_xyz'])
        clipped = []
        for segment in original_left:
            result = np.asarray(segment, float)[:, :2]
            for perimeter in (boundary, reflected_boundary):
                result = clip_segment(result[0], result[1], perimeter)
                if result is None:
                    raise ValueError(f'hard mirrored line {left_id} has no common in-rink segment')
            clipped.append(np.column_stack([result, np.zeros(len(result))]).tolist())
        left['segments_xyz'] = clipped
        mirrored = np.asarray(clipped, float).copy()
        mirrored[..., 0] = 2 * cx - mirrored[..., 0]
        right['segments_xyz'] = mirrored.tolist()
        for feature, original in ((left, original_left), (right, original_right)):
            if not np.allclose(feature['segments_xyz'], original, atol=1e-9, rtol=0.):
                feature['joint_unclipped_segments_xyz'] = original
                feature['joint_export_boundary_clipped'] = True
            feature['hard_export_boundary_rule'] = 'common actual/reflected faceted footprint; clip left once and mirror right exactly'
        protected.update((left_id, right_id))
    left, right = features['goal_crease_left'], features['goal_crease_right']
    original_left = copy.deepcopy(left['polygon_xyz'])
    original_right = copy.deepcopy(right['polygon_xyz'])
    polygon = np.asarray(original_left, float)[:, :2]
    for perimeter in (boundary, reflected_boundary):
        polygon = clip_polygon(polygon, perimeter)
        if len(polygon) < 3:
            raise ValueError('hard mirrored goal crease has no common in-rink area')
    left['polygon_xyz'] = np.column_stack([polygon, np.zeros(len(polygon))]).tolist()
    mirrored = np.asarray(left['polygon_xyz']).copy()
    mirrored[:, 0] = 2 * cx - mirrored[:, 0]
    right['polygon_xyz'] = mirrored.tolist()
    for feature, original in ((left, original_left), (right, original_right)):
        if len(original) != len(feature['polygon_xyz']) or not np.allclose(feature['polygon_xyz'], original, atol=1e-9, rtol=0.):
            feature['joint_unclipped_polygon_xyz'] = original
            feature['joint_export_boundary_clipped'] = True
        feature['hard_export_boundary_rule'] = 'common actual/reflected faceted footprint; clip left once and mirror right exactly'
    protected.update(('goal_crease_left', 'goal_crease_right'))
    return protected


def rebuild_scene(semantic, fit=None, revision='v0.15-joint-video', comparison=None):
    hard_profile = normalize_hard_geometry_constraints(semantic)
    scene = (project_scene_to_hard_constraints(semantic, hard_profile)
             if hard_profile else copy.deepcopy(semantic))
    hard_structure_only = bool(hard_profile and fit is None and 'hard_structure_update' in scene)
    params = scene['parameters']
    radii = {c: params.get('corner_radii', {}).get(c, params['corner_radius']) for c in CORNERS}
    boundary = rounded_boundary(params['bounds'], radii, params['grid_spacing'], params.get('arc_steps_per_quarter', 24))
    ice = make_ice(boundary, params['bounds'], params['grid_spacing']); ice['group'] = 'ice'
    boards = boards_mesh(boundary, params['board_height'], params['board_thickness'])
    equal_corners = bool(hard_profile and hard_profile.get('equal_corner_radii'))
    if equal_corners:
        boards['construction'] = 'shared_corner_radius; polygon_normal_mitered_outer_offset'
    checks = validate([ice, boards], boundary, params['board_height'], params['board_thickness'])
    scene['revision'] = revision
    scene['status'] = ('user_hard_relation_projection_not_video_refitted_NOT_physically_validated'
                       if hard_structure_only else 'joint_video_candidate_with_rebuilt_geometry_NOT_physically_validated')
    scene['meshes'] = [ice, boards]
    scene['parameters']['corner_radii'] = radii
    scene['parameters']['corner_radius_status'] = (
        'one fitted physical radius defines all four corners' if equal_corners else
        'legacy scalar retained for provenance only; corner_radii defines this revision')
    scene['parameters']['geometry_source'] = (
        'user_hard_relation_projection_from_prior_video_candidate' if hard_structure_only else
        'semantic_joint_video_candidate; unchanged parameters remain schematic')
    scene['ice_boundary'] = {**scene.get('ice_boundary', {}), 'points_xyz': np.column_stack([boundary, np.zeros(len(boundary))]).tolist(),
        'closed': True, 'closure': 'last connects to first', 'ice_mesh_vertex_indices': ice['boundary_vertex_indices'],
        'boards_inner_bottom_vertex_indices': boards['inner_bottom_vertex_indices'],
        'identity': 'single rebuilt perimeter shared by ice, grid and boards'}
    scene['ice_grid'] = grid_data(boundary, params, ice)
    scene['board_surface'], scene['advertisements'] = rebuild_board_surface(boundary, scene['advertisements'], params['board_height'])
    protected = clip_hard_mirrored_features(scene, boundary) if hard_profile else set()
    for feature in scene['ice_features']:
        if feature['id'] in protected:
            continue
        if 'polygon_xyz' in feature and feature.get('open_edges'):
            clip_open_polygon_paint(feature, boundary)
            continue
        if 'segments_xyz' in feature:
            original = copy.deepcopy(feature['segments_xyz']); clipped = []
            for segment in original:
                result = clip_segment(np.asarray(segment)[0, :2], np.asarray(segment)[1, :2], boundary)
                if result is not None:
                    clipped.append(np.column_stack([result, np.zeros(len(result))]).tolist())
            if len(original) != len(clipped) or not np.allclose(original, clipped, atol=1e-9, rtol=0.):
                feature['joint_unclipped_segments_xyz'] = original
                feature['segments_xyz'] = clipped
                feature['joint_export_boundary_clipped'] = True
        if 'polygon_xyz' in feature:
            original = copy.deepcopy(feature['polygon_xyz'])
            clipped = clip_polygon(np.asarray(original)[:, :2], boundary)
            changed = len(original) != len(clipped) or not np.allclose(np.asarray(original)[:, :2], clipped, atol=1e-9, rtol=0.)
            if changed:
                feature['joint_unclipped_polygon_xyz'] = original
                feature['polygon_xyz'] = np.column_stack([clipped, np.zeros(len(clipped))]).tolist()
                feature['joint_export_boundary_clipped'] = True
    scene.setdefault('joint_refinement', {})['boundary_mesh_rebuild_required'] = False
    scene['joint_refinement']['meshes_rebuilt_from_shared_semantic_parameters'] = True
    confirmed_advertisements = any(
        ad.get('relative_position_status') == 'user_confirmed_side_order_and_schematic_layout'
        for ad in scene['advertisements'])
    advertisement_uncertainty = (
        'User-confirmed advertisement identity, side, order and content pairs are preserved; '
        'this export does not fit physical placement or size. Unknown fields stay null, and '
        'historical schematic attachments are provenance only.'
        if confirmed_advertisements else
        'identity/source preserved; XY reattachment is schematic, vertical placement and size remain unknown')
    scene.setdefault('uncertainty', {}).update(physical_accuracy_verified=False,
        ice_feature_geometry='listed parameters jointly fitted; remaining parameters retain schematic values',
        advertisements=advertisement_uncertainty)
    if any(change['kind'] == 'board_height' for change in scene['joint_refinement'].get('parameter_changes', [])):
        scene['uncertainty']['board_height'] = 'conditional joint image fit; not independently measured'
        params['board_height_status'] = 'conditional_joint_image_fit_not_measured'
    scene.setdefault('build_scope', {}).update(video_fitting_run=fit is not None or 'base_scene_sha256' in scene['joint_refinement'], visualization_created=False,
        independent_corner_geometry_rebuilt=True)
    if hard_structure_only:
        scene['build_scope']['video_fitting_run'] = False
        scene['uncertainty']['ice_feature_geometry'] = (
            'user-declared relations projected exactly from the prior video candidate; '
            'this revision did not run video fitting or establish new image accuracy')
    if fit is not None:
        scene['joint_refinement']['optimizer'] = copy.deepcopy(fit.get('optimizer', {}))
        scene['joint_refinement']['local_numerical_identifiability'] = copy.deepcopy(fit.get('local_numerical_identifiability', {}))
        scene['uncertainty']['numerical_solution'] = (
            'optimizer_termination_reported_success; not a physical accuracy or uniqueness claim'
            if fit.get('optimizer', {}).get('success') else
            'optimizer_did_not_report_convergence; retain this output as a candidate')
        scene['cameras'] = [dict(id=row['id'], installation_id=row['id'], physical_camera_id=row.get('physical_camera_id'),
            position_xyz=row['center_xyz'], position_status=row.get('position_status', 'conditional_joint_candidate_NOT_measured'),
            evidence_refs=[{'frame_id': frame['id'], 'source_id': frame.get('source_id'), 'source_pts': frame.get('source_pts'),
                            'time_base': frame.get('time_base')} for frame in fit['frames'] if frame['installation_id'] == row['id']],
            radial_k1_image=row.get('radial_k1_image'), radial_scale_image=row.get('radial_scale_image'),
            physical_position_verified=False) for row in fit['installations']]
        scene['frame_cameras'] = copy.deepcopy(fit['frames'])
        for identity in scene.get('camera_identity_reference', []):
            matches = [c['id'] for c in scene['cameras'] if c.get('physical_camera_id') == identity['id']]
            if matches:
                identity['position_status'] = 'conditional_candidates_available_not_measured'
                identity['conditional_installation_ids'] = matches
    if comparison is not None:
        scene['joint_refinement']['training_candidate_comparison'] = copy.deepcopy(comparison)
        scene['uncertainty']['candidate_comparison'] = (
            'Training alternatives and ambiguous feature branches are preserved in '
            'joint_refinement.training_candidate_comparison; this comparison is not held-out validation.')
    semantics = semantic_checks(scene, boundary)
    if hard_profile:
        structure = hard_geometry_checks({f['id']: f for f in scene['ice_features']}, params, radii, hard_profile)
        if not structure['all_constraints_satisfied']:
            raise ValueError('mesh rebuild broke declared hard geometry constraints')
        scene['hard_geometry_checks'] = structure
        semantics['hard_geometry_checks'] = copy.deepcopy(structure)
    if not semantics['grid_cell_area_equals_shared_boundary']:
        raise ValueError('grid cells do not cover the rebuilt ice perimeter')
    combined = {'mesh_and_grid_checks_passed': checks['all_checks_passed'] and semantics['grid_cell_area_equals_shared_boundary'],
                'mesh_checks': checks, 'semantic_checks': semantics, 'physical_accuracy_verified': False}
    geometry = dict(schema_version='rinkspace.scene-geometry.v1', model_id=scene['model_id'], revision=revision,
        status=scene['status'], units=dict(name=scene['coordinate_system']['unit'],
            metres_per_unit=scene['coordinate_system'].get('metres_per_unit'), ice_z=0.,
            axes='right-handed X right in user plan, Y up in plan, Z above ice'),
        parameters=copy.deepcopy(params), boundary_xy=boundary.tolist(), boundary_closed=True,
        boundary_closure_convention='last vertex connects to first; first vertex not repeated',
        meshes=copy.deepcopy(scene['meshes']), checks=checks,
        source={'joint_refinement': scene['joint_refinement'], 'script_path': str(Path(__file__).resolve()), 'script_sha256': sha(__file__)},
        exports={}, scope={'physical_accuracy_verified': False, 'independent_corner_radii': not equal_corners,
                          'ice_mesh': True, 'closed_perimeter_vertical_boards': True})
    if hard_profile:
        geometry['hard_geometry_constraints'] = copy.deepcopy(scene['hard_geometry_constraints'])
        geometry['hard_geometry_checks'] = copy.deepcopy(scene['hard_geometry_checks'])
    return scene, geometry, combined


def export(problem, fit, dest, *, semantic=None, scene_path=BASE, revision='v0.15-joint-video',
           comparison=None, source_files=None):
    """Export dictionaries or file paths. Existing model outputs are not replaced."""
    if isinstance(problem, (str, Path)):
        problem = load(problem)
    if isinstance(fit, (str, Path)):
        fit = load(fit)
    if isinstance(semantic, (str, Path)):
        semantic = load(semantic)
    if isinstance(comparison, (str, Path)):
        comparison = load(comparison)
    destination = Path(dest).resolve()
    source_path = Path(scene_path).resolve()
    if destination == source_path.parent or destination == BASE.resolve().parent:
        raise ValueError('refusing to overwrite a source model revision')
    if any((destination / name).exists() for name in ('scene.json', 'scene_geometry.json', 'scene_checks.json')):
        raise FileExistsError('output model already exists; select a new version directory')
    if semantic is None:
        scene = Scene(source_path)
        if problem is None or fit is None:
            raise ValueError('need semantic scene or both problem and fit')
        if fit['scene_sha256'] != scene.sha256:
            raise ValueError('fit/base scene SHA mismatch')
        joint = JointSceneProblem(scene, problem)
        if joint.parameter_names != fit['parameter_names']:
            raise ValueError('fit parameter ordering does not match problem')
        semantic = joint.export_scene_data(np.asarray(fit['parameter_values']), revision=revision)
    elif fit is not None and semantic.get('joint_refinement', {}).get('base_scene_sha256') != fit['scene_sha256']:
        raise ValueError('semantic/fit base scene SHA mismatch')
    result, geometry, checks = rebuild_scene(semantic, fit, revision, comparison)
    destination.mkdir(parents=True, exist_ok=True)
    provenance = {str(Path(p).resolve()): sha(p) for p in source_files or [] if Path(p).exists()}
    result.setdefault('source', {}).update(joint_export_inputs=provenance,
        joint_export_script_sha256=sha(__file__), geometry_json_sha256=None)
    save(destination / 'scene_geometry.json', geometry)
    result['source']['geometry_json_sha256'] = sha(destination / 'scene_geometry.json')
    save(destination / 'scene.json', result)
    save(destination / 'scene_checks.json', dict(schema_version='rinkspace.scene3d-checks.v1',
        scene_path='scene.json', scene_sha256=sha(destination / 'scene.json'), checks=checks))
    return {'output': str(destination), 'scene_sha256': sha(destination / 'scene.json'), **checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-dir', type=Path)
    parser.add_argument('--semantic', type=Path)
    parser.add_argument('--scene', type=Path, default=BASE)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--revision', default='v0.15-joint-video')
    parser.add_argument('--comparison', type=Path, help='Optional training candidate/ambiguity report to preserve in the model')
    args = parser.parse_args()
    if args.fit_dir is None and args.semantic is None:
        parser.error('supply --fit-dir or --semantic')
    semantic = args.semantic
    if semantic is None and (args.fit_dir / 'scene_semantic.json').exists():
        semantic = args.fit_dir / 'scene_semantic.json'
    problem = args.fit_dir / 'problem.json' if args.fit_dir and (args.fit_dir / 'problem.json').exists() else None
    fit = args.fit_dir / 'fit.json' if args.fit_dir and (args.fit_dir / 'fit.json').exists() else None
    result = export(problem, fit, args.out, semantic=semantic, scene_path=args.scene, revision=args.revision,
                    comparison=args.comparison,
                    source_files=[p for p in (semantic, problem, fit, args.scene, args.comparison) if p is not None])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
