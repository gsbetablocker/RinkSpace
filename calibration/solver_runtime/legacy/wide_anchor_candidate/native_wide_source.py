"""Immutable original native56218880 source and current-world target binding.

The observer remains the historical high_56218880 / women_sp_high installation.
The user's circled long_side_secondary camera is a different, visible target;
its physical-camera observation belongs to a separate factor, never this pose.
This module neither solves nor changes any active factor graph or source file.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
from rinkspace.storage import AssetStore

REPO = Path('D:/OneDrive/codex_projs/RinkSpace')
SOURCE_PTS = 56218880
SOURCE_VIDEO_SHA256 = 'f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'
SOURCE_IMAGE_SHA256 = 'ccd92c70eaa4a8b410808ad0372dba4ce0fb7ba30de4a75ef0368459f32a86e5'
SCENE_UNIT_M = .31353729994266055
FILES = {
    'v5_problem': ('outputs/calibration/ankara_2026_wide_joint_20260923/A_pair_symmetry_A1_upper_A2_balance_v5/problem.json', '5b139c6a53b1926f8df20cf0810d5c45886fc7d5b47c8c9928a401438accce12'),
    'v5_additional': ('outputs/calibration/ankara_2026_wide_joint_20260923/complete_observations_v5_A_pair_arc_balance.json', 'a50c1e4391d782d0cbe920fbf80296426a45e19ea11e0c287efea1c16a5313e8'),
    'step02': ('three_frame_refit/inputs/step02_observations.json', '2ebaa742175967477f9bd1b40a1d819a3e6e4ae3c55fb671b136f6e19c4d39f3'),
    'annotation_inventory': ('three_frame_refit/inputs/annotation_inventory.json', '043f02756f09ac91ea999c49f6a8e59cc834b09ad4c8413944d3eb15f67a8c63'),
    'user_ad_digitization': ('outputs/rink_model/ankara_2026_shared/v0.23-high-view-ad-corners/high_view_ad_corners_56218880.json', 'c482a57576d7c641fe6f2a474767033ac1621bef24c3bbe314c1e5054ccd0aa0'),
}

def _require(condition, message):
    if not condition:
        raise ValueError(message)

def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

def _read_authenticated(root, key):
    relative, expected = FILES[key]
    path = AssetStore(root).resolve(relative)
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == expected, f'Original wide source bytes changed: {path}')
    return json.loads(raw.decode('utf-8-sig')), dict(path=str(path.resolve()), sha256=expected)

def _uv(observation):
    a = np.asarray(observation['pixels_uv'], np.float64)
    _require(a.ndim == 2 and a.shape[1] == 2 and len(a) > 0 and np.isfinite(a).all(), 'Finite nonempty original native UV required')
    _require(np.all((a[:, 0] >= 0) & (a[:, 0] < 1920) & (a[:, 1] >= 0) & (a[:, 1] < 1080)), 'Original UV outside native1920x1080')
    return a

@dataclass(frozen=True)
class NativeWideSourcePacket:
    """JSON bytes are immutable; properties return independent fresh values.

    Both historical numerical representations are retained. A caller must
    choose one; originalv5 and step02 must never both contribute to FIT.
    """
    frame_json: str
    observations_json: tuple[str, ...]
    step02_observations_json: tuple[str, ...]
    advertisement_observations_json: tuple[str, ...]
    contract_json: str

    @property
    def frame(self): return json.loads(self.frame_json)
    @property
    def observations(self): return tuple(json.loads(v) for v in self.observations_json)
    @property
    def step02_observations(self): return tuple(json.loads(v) for v in self.step02_observations_json)
    @property
    def advertisement_observations(self): return tuple(json.loads(v) for v in self.advertisement_observations_json)
    @property
    def contract(self): return json.loads(self.contract_json)

def load_native_wide_source_packet(root=REPO, *, verify_source_image=True):
    """45 originalv5 FIT curves/2144UV, plus 26 separately owned user corners.

    No observation comes from the old fitted world or historical projection.
    The original numeric tolerances are preserved, with their stated source
    basis; they are not promoted to independently measured accuracy bounds.
    """
    root = Path(root)
    problem, p_ref = _read_authenticated(root, 'v5_problem')
    extra, x_ref = _read_authenticated(root, 'v5_additional')
    step, s_ref = _read_authenticated(root, 'step02')
    inventory, i_ref = _read_authenticated(root, 'annotation_inventory')
    _, a_ref = _read_authenticated(root, 'user_ad_digitization')
    _require(problem['schema_version'] == 'rinkspace.real-joint-video-problem.v1' and problem['synthetic'] is False, 'Real original wide problem required')
    _require(step['schema_version'] == 'rinkspace.three_frame_refit.observations.v2', 'Historical step02 source representation changed')
    frame = next(f for f in problem['frames'] if int(f['source_pts']) == SOURCE_PTS)
    _require(frame['id'] == 'women_sp_high_56218880' and frame['installation_id'] == 'women_sp_high', 'Independent high observer identity changed')
    _require(frame['source_sha256'] == SOURCE_VIDEO_SHA256 and frame['time_base'] == '1/15360' and frame['image_sha256'] == SOURCE_IMAGE_SHA256, 'Actual original wide source/clock changed')
    image = AssetStore(root).resolve('outputs/rink_model/ankara_2026_shared/v0.13-women-sp/evidence/survey/survey_3660.png')
    if verify_source_image:
        _require(hashlib.sha256(image.read_bytes()).hexdigest() == SOURCE_IMAGE_SHA256, 'Original wide native image changed')
    incoming = [o for o in problem['observations'] if o['frame_id'] == frame['id']]
    _require(len(incoming) == 45 and sum(len(_uv(o)) for o in incoming) == 2144 and len({o['id'] for o in incoming}) == 45, 'Original45/2144 source count or IDs changed')
    by_id = {o['id']: o for o in incoming}
    extra_by_id = {o['id']: o for o in extra['observations']}
    for o in incoming:
        _require(o['use'] == 'fit' and o['synthetic'] is False and o['measurement_origin'] in ('direct_image_measurement', 'direct_native_image_measurement', 'inherited_direct_image_measurement'), 'Original FIT/image-source role changed')
        _require(np.isfinite(o['sigma_px']) and o['sigma_px'] > 0 and o['source_frame']['source_pts'] == SOURCE_PTS, 'Original native sigma/PTS invalid')
        if o['id'] in extra_by_id:
            e = extra_by_id[o['id']]
            _require(o['pixels_uv'] == e['pixels_uv'] and o['sigma_px'] == e['sigma_px'], 'Merged v5 differs from its original image measurement')
            if o['target'] != e['target']:
                _require(o['id'] == 'women_sp_high_56218880_near_lower_right_rail_ice_facing_top' and
                    e['target'] == dict(kind='board_side', side='right_end', edge='top') and
                    o['target'] == dict(kind='board_side', side='right_end', edge='top', continue_corner='lower_right') and
                    isinstance(o.get('target_revision'), dict), 'Unqualified merged target identity/domain change')
    sampled = [o for o in step['observations'] if o['frame_id'] == 'high_56218880' and o['use'] == 'fit' and o['category'] != 'ads']
    ads = [o for o in step['observations'] if o['frame_id'] == 'high_56218880' and o['use'] == 'fit' and o['category'] == 'ads']
    _require(len(sampled) == 45 and sum(len(_uv(o)) for o in sampled) == 2271 and len(ads) == 26, 'Historical high representation/26 user corners changed')
    for o in sampled:
        source = by_id[o['id']]
        _require(o['source_pixels_uv'] == source['pixels_uv'] and o['target'] == source['target'], 'Historical resampling no longer represents originalv5 UV/identity')
    corner_by_id = {o['id']: o for o in inventory['advertisement_corners']}
    for o in ads:
        _require(o['pixels_uv'] == [corner_by_id[o['id']]['source_frame_uv_1920x1080']] and o['uncertainty']['sigma_px'] is None, 'Original user corner UV/null sigma changed')
    contract = dict(schema='rinkspace.original-native-wide56218880-source-packet.v1', source_pts=SOURCE_PTS,
        source_time='01:01:00.083', source_time_base='1/15360', source_video_sha256=SOURCE_VIDEO_SHA256,
        native_image=dict(path=str(image.resolve()), sha256=SOURCE_IMAGE_SHA256), pixel_domain='native1920x1080',
        observer_installation_id='women_sp_high', observer_alias='high_56218880',
        visible_circled_camera_target_installation='long_side_secondary', observer_is_visible_camera_target=False,
        observer_physical_xyz_not_measured=True, original_observer_registry=frame.get('camera_identity_evidence'),
        source_files=dict(v5_problem=p_ref, v5_additional=x_ref, step02=s_ref, annotation_inventory=i_ref, user_ad_digitization=a_ref),
        original_v5_groups=45, original_v5_UV=2144, original_v5_residual_rows=4288,
        historical_step02_curve_groups=45, historical_step02_curve_UV=2271, historical_step02_full_groups=71,
        original_user_ad_corners=26, advertisement_roles='FIT_original_user_corner_marking; owned by separate advertisement adapter',
        default_bound_advertisements=False, default_representation='v5', alternative_representations_must_not_be_double_fitted=True,
        advertisement_sigma=None, advertisement_sigma_basis='not_a_calibrated_accuracy_bound; explicit engineering scale required if used',
        original_native_sigmas=[o['sigma_px'] for o in incoming], original_v5_near_right_weight_experiment=problem.get('near_right_weight_experiment'),
        historical_weight_fields_preserved=True, additional_strengthening_weight='caller_explicit_configuration_only_not_a_new_measurement_sigma',
        historical_merged_target_revision=[dict(id=o['id'], original_additional_target=extra_by_id[o['id']]['target'], merged_problem_target=o['target'], revision=o['target_revision']) for o in incoming if o['id'] in extra_by_id and o['target'] != extra_by_id[o['id']]['target']],
        no_50Hz_alias_guess=True, no_new_source_UV=True, no_world_backprojection_measurement=True,
        old_fit_camera_role='optional_initializer_never_observation_or_prior',
        current_shared_world_binding_required=True, graph_installed=False, accuracy_accepted=False)
    return NativeWideSourcePacket(_json(frame), tuple(map(_json, incoming)), tuple(map(_json, sampled)), tuple(map(_json, ads)), _json(contract))

def _board_piece(scene, side, samples, edge):
    p = scene['parameters']; x0, y0, x1, y1 = map(float, p['bounds'])
    c = p['corner_curve']
    _require(c['kind'] == 'four_congruent_symmetric_superellipse_quarters', 'Current authenticated superellipse course required')
    rx, ry = map(float, c['tangent_extent_xy_scene_units']); exponent = float(c['exponent'])
    _require(rx > 0 and ry > 0 and 1 < exponent < 8 and x1-x0 > 2*rx and y1-y0 > 2*ry, 'Current course structural domain invalid')
    z = float(p['board_height']) if edge == 'top' else 0.
    _require(edge in ('top', 'contact'), 'Original top/contact physical identity required')
    if side in ('upper_right', 'upper_left', 'lower_left', 'lower_right'):
        start = dict(upper_right=0., upper_left=90., lower_left=180., lower_right=270.)[side]
        angle = np.deg2rad(start + 90*np.linspace(0., 1., samples))
        cs = np.column_stack((np.cos(angle), np.sin(angle)))
        cs[np.abs(cs) < 1e-14] = 0.
        center = [x1-rx if 'right' in side else x0+rx, y1-ry if 'upper' in side else y0+ry]
        xy = center + np.array([rx, ry])*np.sign(cs)*np.abs(cs)**(2/exponent)
    else:
        endpoints = dict(upper_long_side=([x0+rx, y1], [x1-rx, y1]), lower_long_side=([x0+rx, y0], [x1-rx, y0]),
            left_end=([x0, y0+ry], [x0, y1-ry]), right_end=([x1, y0+ry], [x1, y1-ry]))
        _require(side in endpoints, 'Original named finite course part required')
        xy = np.linspace(*endpoints[side], samples)
    return np.column_stack((xy, np.full(len(xy), z)))

def world_curve(scene, target, *, samples=256, scene_unit_m=SCENE_UNIT_M):
    """Current semantic world curve from original target identity only.

    Numerical world sampling is the originalv5 count256, never UV evidence.
    Original polygon_supporting_line measures line position, extended only
    to original current rink bounds; it never measures a template endpoint.
    The rendered Lane remains its original finite edge. No domain is silently
    changed to a nearest generic mark or an unobserved contact surface.
    """
    _require(isinstance(samples, (int, np.integer)) and samples >= 8 and np.isfinite(scene_unit_m) and scene_unit_m > 0, 'Original finite numerical sample count/domain required')
    kind = target['kind']
    if kind == 'board_side':
        side, edge = target['side'], target.get('edge', 'contact')
        out = _board_piece(scene, side, samples, edge)
        corner = target.get('continue_corner')
        if corner is not None:
            neighbours = dict(upper_long_side=('upper_left', 'upper_right'), lower_long_side=('lower_left', 'lower_right'), left_end=('lower_left', 'upper_left'), right_end=('lower_right', 'upper_right'))
            _require(side in neighbours and corner in neighbours[side], 'Original continuation must be physically adjacent')
            arc = _board_piece(scene, corner, samples, edge)
            distance = np.array([[np.linalg.norm(a-b) for b in (arc[0], arc[-1])] for a in (out[0], out[-1])])
            i, j = np.unravel_index(np.argmin(distance), distance.shape)
            _require(distance[i, j] <= 1e-7, 'Named current course pieces have no shared finite endpoint')
            if i == 0: out = out[::-1]
            if j == 1: arc = arc[::-1]
            out = np.vstack((out, arc[1:]))
        _require(not target.get('allow_adjacent_corners'), 'No unqualified extra corner domain')
        return out
    if kind == 'advertisement_corner':
        number = int(target['user_ad_number']); role = target.get('corner_role', target.get('role'))
        advert = next(a for a in scene['advertisements'] if int(a['user_number']) == number)
        out = np.asarray(advert['corners_xyz_scene_units'][role], float).reshape(1,3)
        _require(out.shape == (1, 3) and np.isfinite(out).all(), 'Current named print corner required')
        return out.copy()
    feature = next(f for f in scene['ice_features'] if f['id'] == target['feature_id'])
    if kind == 'circle_outline':
        _require(feature['kind'] == 'circle' and np.array_equal(feature['plane_normal_xyz'], [0., 0., 1.]), 'Current named planar circle required')
        circle = feature
        if 'ring_role' in target:
            circle = next(r for r in feature['rings'] if r['role'] == target['ring_role'])
        angle = np.linspace(0., 2*np.pi, samples+1)
        return np.asarray(circle['center_xyz'], float) + float(circle['radius'])*np.column_stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)))
    if kind == 'line_segment':
        _require(feature['kind'] == 'polyline', 'Current named transverse line required')
        a, b = np.asarray(feature['segments_xyz'][int(target.get('segment', 0))], float)
        return np.linspace(a, b, samples)
    if kind in ('polygon_supporting_line', 'polygon_outline'):
        _require(feature['kind'] == 'polygon', 'Current named polygon required')
        polygon = np.asarray(feature['polygon_xyz'], float)
        open_edges = set(feature.get('open_edges', []))
        if kind == 'polygon_supporting_line':
            _require(feature['id'] in ('lane1', 'lane2', 'lane3', 'lane4'), 'Original Lane supporting-line identity required')
            index = int(target['index'])
            _require(0 <= index < len(polygon) and index not in open_edges, 'Cannot fit an unobserved open Lane edge')
            a, b = polygon[[index, (index+1) % len(polygon)]].copy()
            difference = abs(b[:2]-a[:2])
            _require(np.sum(difference > 1e-9) == 1 and np.allclose([a[2], b[2]], 0.), 'Original axis-aligned Lane line required')
            axis = int(np.argmax(difference))
            a[axis], b[axis] = scene['parameters']['bounds'][axis], scene['parameters']['bounds'][axis+2]
            return np.linspace(a, b, samples)
        if feature.get('closed') is False and not open_edges: open_edges.add(len(polygon)-1)
        indices = [i for i in range(len(polygon)) if i not in open_edges]
        _require(indices and all(b == a+1 for a, b in zip(indices[:-1], indices[1:])), 'Original polygon must have one connected finite chain')
        count = max(2, int(np.ceil(samples/len(indices))))
        pieces = [np.linspace(polygon[i], polygon[(i+1) % len(polygon)], count, endpoint=False) for i in indices]
        return np.vstack((*pieces, polygon[[(indices[-1]+1) % len(polygon)]]))
    raise ValueError(f'Original wide target is not supported: {target}')

def bind_native_wide_source_rows(packet, shared_world, world_q=None, *, sampling='v5', include_ads=False):
    """Bind current scene/world to exactly one original numerical representation.

    Returned dicts are owned copies, so no live source or immutable packet is
    mutated. No loss/weight choice or implicit null-sigma substitution occurs.
    """
    _require(isinstance(packet, NativeWideSourcePacket), 'Authenticated immutable native wide packet required')
    if isinstance(shared_world, dict):
        _require(world_q is None, 'A direct scene already represents one complete current world event')
        scene = shared_world
    else:
        _require(world_q is not None and (hasattr(shared_world, 'source_scene') or hasattr(shared_world, 'scene')), 'Current shared world.source_scene(q) or scene(q) required')
        scene = shared_world.source_scene(world_q) if hasattr(shared_world, 'source_scene') else shared_world.scene(world_q)
    _require(sampling in ('v5', 'step02'), 'Choose exactly one original source representation')
    observations = packet.observations if sampling == 'v5' else packet.step02_observations
    original_by_id = {o['id']: o for o in packet.observations}
    if include_ads: observations = (*observations, *packet.advertisement_observations)
    rows = []
    for observation in observations:
        row = dict(observation)
        row['uv'] = _uv(row).copy()
        row['world'] = world_curve(scene, row['target'])
        row['source_pts'] = SOURCE_PTS
        row['source_role'] = 'FIT'
        row['observer_installation_id'] = 'women_sp_high'
        row['source_representation'] = sampling if row['target']['kind'] != 'advertisement_corner' else 'original_user_corner26'
        row['world_source_identity'] = 'current shared scene semantic target; not old fit/world inverse projection'
        row['original_v5_sigma_px'] = original_by_id[row['id']]['sigma_px'] if row['id'] in original_by_id else None
        if 'sigma_px' not in row: row['sigma_px'] = row.get('uncertainty', {}).get('sigma_px')
        _require(row['world'].ndim == 2 and row['world'].shape[1] == 3 and np.isfinite(row['world']).all(), 'Current original semantic world is not finite')
        rows.append(row)
    return rows

def observation_world_curve(observation, scene, *, samples=256):
    """Convenience API for an unchanged original source-row dictionary."""
    return world_curve(scene, observation['target'], samples=samples)

def export_native_wide_source_packet(packet, path):
    """Write traceable source bytes, never old derived XYZ or camera targets."""
    path = Path(path)
    path.write_text(json.dumps(dict(contract=packet.contract, frame=packet.frame,
        original_v5_observations=packet.observations, historical_step02_observations=packet.step02_observations,
        original_user_advertisement_corner_observations=packet.advertisement_observations), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
