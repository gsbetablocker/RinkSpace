"""Use source-proven ice pixels in the existing complete programme factors.

Only explicitly reviewed FIT point pixels/support change on a copied active
row. Original packet bytes, physical material, roles, weights and scalar slots
remain intact. A corrected observation is never an accepted camera solution.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import re

import numpy as np

SOURCE_SHA = 'f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'
PACKET_SCHEMA = 'rinkspace.original-native-ice-pixel-correction.v1'


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _plain(value):
    return json.loads(json.dumps(value, default=lambda x: x.tolist() if isinstance(x, np.ndarray)
                                else x.item() if isinstance(x, np.generic) else str(x)))


def _reference(ref):
    path = Path(ref['path']).resolve()
    if _sha(path) != ref['sha256']:
        raise ValueError('Original ice correction source evidence changed')
    return path


def _same_runtime_source(row, original):
    for key in ('id', 'pts', 'split', 'origin', 'uv', 'sigma', 'normal', 'normal_interval',
                'point_correspondence', 'curve_observation_kind', 'scanline_branch',
                'scanline_fixed_axis', 'extent_side'):
        if _plain(row.get(key)) != _plain(original.get(key)):
            raise ValueError('Ice correction does not match original runtime source: ' + key)


def _source_only_reading(record):
    reading = record.get('source_only_reading')
    if reading is True:
        return True
    if not isinstance(reading, list) or not reading:
        return False
    support = record.get('corrected_normal_interval_px')
    support_matches = (all(r.get('corrected_normal_interval_px') is None for r in reading)
                       if support is None else
                       [r.get('corrected_normal_interval_px') for r in reading] == support)
    return (len(reading) == len(record.get('point_indices', []))
            and [r.get('original_point_index') for r in reading] == record.get('point_indices')
            and all(r.get('model_used_for_navigation_or_peak_selection') is False
                    or (r.get('model_used_for_navigation_or_peak_selection') is True
                        and r.get('observed_UV_generated_from_model') is False
                        and r.get('raw_source_pixels_read') is True) for r in reading)
            and [r.get('corrected_uv') for r in reading] == record.get('corrected_uv')
            and support_matches)


def _qualified_named_lane(row, old, record):
    """Correct pixels of an already bound lane, never infer a new identity."""
    def declared(value):
        identity = value.get('canonical_physical_geometry_id', value.get('physical_curve_id'))
        if identity is not None:
            return identity
        binding = value.get('world_scene_binding', {})
        expected = f"{binding.get('feature')}_edge{binding.get('edge')}"
        if binding.get('kind') == 'finite_known_lane_edge' and value.get('curve_id') == expected:
            return expected
        return None

    identity = declared(old)
    if not isinstance(identity, str) or re.fullmatch(r'lane[1-4]_edge[02]', identity) is None:
        return False
    if (declared(row) != identity
            or not str(old['id']).startswith(identity)
            or record.get('physical_curve_id') != identity
            or record.get('physical_identity_changed') is not False
            or not np.array_equal(np.asarray(row['world']), np.asarray(old['world']))):
        raise ValueError('Named lane correction must preserve the exact existing physical identity and WORLD')
    required = ('original_pts', 'original_id', 'original_split', 'original_collection_role',
                'original_uv', 'original_sigma', 'original_full_uv_float64_sha256')
    if any(key not in record for key in required):
        raise ValueError('Named lane correction requires explicit original row and pixel identities')
    return True


def _qualified_original_contact(state, row, old, record):
    """Keep the already certified original contact material, not a new rail."""
    if old.get('id') not in ('far_contact', 'ice_board_contact'):
        return False
    from shared_board_course_candidate.course_absolute_material import declared_absolute_course_identity
    if (declared_absolute_course_identity(old) is None
            or declared_absolute_course_identity(old)[1] != 'contact'
            or declared_absolute_course_identity(row) is None
            or declared_absolute_course_identity(row)[1] != 'contact'
            or not np.array_equal(np.asarray(row['world']), np.asarray(old['world']))):
        raise ValueError('Original contact WORLD and declared physical identity must remain exact')
    _, binding = state.geometry._binding(row)
    material = binding.get('material')
    if (binding.get('kind') != 'course' or material is None
            or material.contract.get('z_policy') != 'contact'
            or not np.array_equal(np.asarray(material.old_world), np.asarray(old['world']))):
        raise ValueError('Existing finite current-course contact certificate required')
    required = ('original_pts', 'original_id', 'original_split', 'original_collection_role',
                'original_uv', 'original_sigma', 'original_full_uv_float64_sha256')
    if any(key not in record for key in required):
        raise ValueError('Contact correction requires explicit original row and pixel identities')
    if record.get('physical_identity_changed') is not False:
        raise ValueError('Contact correction may only revise pixels of the same physical material')
    return True


def _qualified_original_top(state, row, old, record):
    """Revise the certified vertical board's top pixels, never an ad edge."""
    if record.get('measurement_role') != 'board_top_boundary':
        return False
    from shared_board_course_candidate.course_absolute_material import declared_absolute_course_identity
    original_identity = declared_absolute_course_identity(old)
    if (original_identity is None or original_identity[1] != 'top'
            or declared_absolute_course_identity(row) != original_identity
            or not np.array_equal(np.asarray(row['world']), np.asarray(old['world']))):
        raise ValueError('Original board-top WORLD and declared physical identity must remain exact')
    _, binding = state.geometry._binding(row)
    material = binding.get('material')
    identity = binding.get('contract', {}).get('physical_identity')
    # Old thin rows and later typed rows already share this certified finite
    # material in course_absolute_material. The cached certificate's name or
    # exposure need not be this source row's declaration. Authenticate the
    # existing explicit alias, never infer a new facet from similar pixels.
    material_contract = {} if material is None else material.contract
    same_identity = (identity == original_identity[0] or
                     ({identity, original_identity[0]} ==
                      {'full_board_top', 'original_named_board_top'}
                      and material_contract.get('schema') ==
                      'rinkspace.original-declared-finite-absolute-course-material.v1'
                      and material_contract.get('identity') == identity))
    if (binding.get('kind') != 'course' or material is None
            or material.contract.get('z_policy') != 'top'
            or not same_identity
            or not np.array_equal(np.asarray(material.old_world), np.asarray(old['world']))):
        raise ValueError('Existing finite current-course board-top certificate required')
    required = ('original_pts', 'original_id', 'original_split', 'original_collection_role',
                'original_actual_pts', 'original_block', 'original_group_order', 'original_row_order',
                'original_uv', 'original_sigma', 'original_full_uv_float64_sha256',
                'original_geometry_identity', 'original_course_z_policy')
    if (any(key not in record for key in required)
            or record.get('physical_identity_changed') is not False
            or record.get('original_geometry_identity') != original_identity[0]
            or record.get('original_course_z_policy') != 'top'
            or record.get('physical_edge_role') != 'vertical_board_blue_upper_boundary'):
        raise ValueError('Board-top correction requires exact original source identity and vertical board blue-upper boundary')
    return True


def _qualified_advertisement_corner(state, row, old, record):
    """Revise a measured print corner while retaining its existing identity."""
    if record.get('measurement_role') != 'advertisement_print_corner':
        return False
    _, binding = state.geometry._binding(row)
    identities = binding.get('contract', {}).get('physical_identities')
    roles = {'left_top', 'right_top', 'right_bottom', 'left_bottom'}
    required = ('original_pts', 'original_id', 'original_split', 'original_collection_role',
                'original_actual_pts', 'original_block', 'original_group_order', 'original_row_order',
                'original_uv', 'original_sigma', 'original_full_uv_float64_sha256',
                'original_geometry_identity')
    typed = (isinstance(identities, (list, tuple)) and len(identities) == 4
             and all(isinstance(i, (list, tuple)) and len(i) == 2
                     and isinstance(i[0], (int, np.integer)) and i[0] > 0
                     and i[1] in roles for i in identities))
    physical_ids = [f'ad{n}:{role}' for n, role in identities] if typed else None
    if (any(key not in record for key in required)
            or binding.get('kind') != 'ad' or not typed
            or len({int(n) for n, _ in identities}) != 1
            or {role for _, role in identities} != roles
            or _plain(identities) != _plain(record['original_geometry_identity'])
            or old.get('physical_ids') != physical_ids or row.get('physical_ids') != physical_ids
            or old.get('point_correspondence') is not True or row.get('point_correspondence') is not True
            or np.asarray(old['uv']).shape != (4, 2) or np.asarray(old['world']).shape != (4, 3)
            or record.get('physical_identity_changed') is not False
            or record.get('physical_edge_role') != 'intersection_of_two_observed_print_boundaries'
            or record.get('corrected_normal_interval_px') is not None
            or record.get('remove_original_normal_interval')
            or any(key in old or key in row for key in ('normal', 'normal_interval', 'normal_interval_px'))
            or not np.array_equal(np.asarray(row['world']), np.asarray(old['world']))):
        raise ValueError('Print-corner correction requires the exact registered original four-corner material identity')
    return True


def _qualified_target_ring_boundary(state, row, old, record):
    """Correct an existing target's outer-boundary UV without inventing support.

    Older hand-read target rows have no normal/support fields. Their registered
    outer material boundary remains the observation; a new RGB reading does
    not turn the layered target into a single stroke or change its radius.
    """
    if record.get('measurement_role') != 'target_ring_outer_material_boundary':
        return False
    _, binding = state.geometry._binding(row)
    identity = binding.get('contract', {}).get('physical_identity')
    feature = getattr(state.geometry, 'features', {}).get(binding.get('name'))
    rings = feature.get('rings') if isinstance(feature, dict) else None
    required = ('original_pts', 'original_id', 'original_split', 'original_collection_role',
                'original_actual_pts', 'original_block', 'original_group_order', 'original_row_order',
                'original_uv', 'original_sigma', 'original_full_uv_float64_sha256',
                'original_geometry_identity')
    if (any(key not in record for key in required)
            or binding.get('kind') != 'circle'
            or not isinstance(identity, (tuple, list)) or len(identity) != 3
            or identity[0] not in ('circle', 'fixed_circle') or identity[2] != 'outer'
            or identity[1] != binding.get('name')
            or not isinstance(rings, list) or not any(r.get('role') == 'outer' for r in rings)
            or _plain(identity) != _plain(record['original_geometry_identity'])
            or record.get('physical_identity_changed') is not False
            or record.get('physical_edge_role') != 'target_ring_outer_material_boundary'
            or record.get('corrected_normal_interval_px') is not None
            or record.get('remove_original_normal_interval')
            or any(key in old or key in row for key in ('normal', 'normal_interval', 'normal_interval_px'))
            or not np.array_equal(np.asarray(row['world']), np.asarray(old['world']))):
        raise ValueError('Target UV-only correction needs its unchanged registered outer material boundary and original absence of support')
    return True


def _explicit_single_stroke_center_revision(old, record):
    """Allow an evidenced correction of previously untyped circle samples.

    The physical single/unfilled/no-rings check remains in the caller. This
    authority corrects the source reading, never the physical circle identity.
    """
    ref = record.get('painted_center_role_revision_reference')
    if ref is None:
        return False
    if not isinstance(ref, dict):
        raise ValueError('Explicit painted-centre role revision reference required')
    authority = json.loads(_reference(ref).read_text(encoding='utf8'))
    if (authority.get('schema') != 'rinkspace.original-single-stroke-center-role-revision.v1'
            or authority.get('source_video_sha256') != SOURCE_SHA
            or authority.get('time_base') != '1/15360'
            or authority.get('previous_measurement_role') != 'untyped_approximate_curve_points'
            or authority.get('revised_measurement_role') != 'painted_line_center'
            or authority.get('single_unfilled_painted_stroke') is not True
            or authority.get('authority_kind') != 'explicit_user_observation_correction'
            or not isinstance(authority.get('user_message'), str)
            or not authority['user_message'].strip()
            or authority.get('model_projection_used_for_measurement') is not False
            or old.get('measurement_role') is not None
            or old.get('source_uv_role') is not None
            or old.get('samples')):
        raise ValueError('Source-only correction authority for previously untyped circle samples required')
    for key in ('original_packet_row_index', 'original_actual_pts', 'original_id',
                'original_full_uv_float64_sha256', 'original_geometry_identity'):
        if key not in record or _plain(authority.get(key)) != _plain(record[key]):
            raise ValueError('Painted-centre role revision original identity differs: ' + key)
    frame = authority.get('source_frame')
    if not isinstance(frame, dict) or frame.get('pts') != record['original_actual_pts']:
        raise ValueError('Painted-centre role revision needs the exact original exposure')
    _reference(frame)
    return True


def _qualified_painted_line_center(state, row, old, record):
    """Revise pixels/paint semantics of one already registered ice carrier."""
    if record.get('measurement_role') != 'painted_line_center':
        if record.get('remove_original_normal_interval') is True:
            raise ValueError('Removing paint support requires an explicit painted-line centre measurement')
        return False
    required = ('original_pts', 'original_id', 'original_split', 'original_collection_role',
                'original_actual_pts', 'original_block', 'original_group_order', 'original_row_order',
                'original_uv', 'original_sigma', 'original_full_uv_float64_sha256',
                'original_geometry_identity')
    if (any(key not in record for key in required)
            or record.get('physical_identity_changed') is not False
            or record.get('paint_footprint_is_not_location_uncertainty') is not True
            or record.get('remove_original_normal_interval') is not True
            or record.get('corrected_normal_interval_px') is not None
            or not np.array_equal(np.asarray(row['world']), np.asarray(old['world']))):
        raise ValueError('Painted-line centre correction requires exact source/carrier identity and no paint-width location band')
    _, binding = state.geometry._binding(row)
    identity = binding.get('contract', {}).get('physical_identity')
    typed = isinstance(identity, (list, tuple)) and bool(identity)
    ordinary = (typed and binding.get('kind') in ('line', 'carrier', 'lane', 'crease')
                and identity[0] == binding['kind'])
    # A single unfilled painted circular stroke is an ordinary line. A layered
    # or filled target has distinct actual outer/inner boundaries, never a
    # midpoint radius. Use registered feature structure AND original source
    # semantics; a caller's label, name pattern or apparent width cannot qualify.
    feature = getattr(state.geometry, 'features', {}).get(binding.get('name'))
    style = feature.get('style', {}) if isinstance(feature, dict) else {}
    unfilled = (isinstance(feature, dict)
                and ('fill_srgb' in feature or 'fill_srgb' in style)
                and feature.get('fill_srgb') is None and style.get('fill_srgb') is None)
    original_center_role = (
        old.get('measurement_role') == 'painted_line_center'
        or old.get('source_uv_role') == 'painted_line_center'
        or old.get('source_uv_role') == 'source paint-support reading center; interval is geometry-center uncertainty envelope, not a perfect annotated center'
        or (isinstance(old.get('samples'), list) and bool(old['samples'])
            and all(str(sample.get('boundary_definition', '')).startswith('world line represents paint centerline;')
                    for sample in old['samples'])))
    single_circle = (typed and binding.get('kind') == 'circle'
                     and identity[0] in ('circle', 'fixed_circle')
                     and isinstance(feature, dict) and not feature.get('rings')
                     and unfilled
                     and (original_center_role
                          or _explicit_single_stroke_center_revision(old, record)))
    if (not (ordinary or single_circle)
            or _plain(identity) != _plain(record['original_geometry_identity'])):
        raise ValueError('Painted-line centre correction requires a registered ordinary ice line; ring boundaries retain outer/inner semantics')
    return True


def _verify_record_original(record, item, old, points):
    expected = dict(original_pts=old['pts'], original_id=old['id'], original_split=old['split'],
                    original_collection_role=item['collection_role'],
                    original_actual_pts=int(item['group_actual_pts']),
                    original_block=item['block'], original_group_order=int(item['group_order']),
                    original_row_order=int(item['row_order']))
    for key, value in expected.items():
        if key in record and _plain(record[key]) != _plain(value):
            raise ValueError('Source correction original locator differs: ' + key)
    uv = np.asarray(old['uv'], dtype=np.float64)
    sigma = np.broadcast_to(np.asarray(old['sigma'], float), (len(uv),))
    for key, value in (('original_uv', uv[points]), ('original_sigma', sigma[points])):
        if key in record and not np.array_equal(np.asarray(record[key]), value):
            raise ValueError('Source correction original point values differ: ' + key)
    digest = hashlib.sha256(uv.tobytes()).hexdigest()
    if ('original_full_uv_float64_sha256' in record
            and record['original_full_uv_float64_sha256'] != digest):
        raise ValueError('Source correction original full UV identity differs')


def _verify_painted_row_coverage(record, item, old, points, qualifications, parent_sha):
    """A whole paint row consists only of measured or RGB-retired points."""
    retired = record.get('retired_original_point_indices', [])
    if (not isinstance(retired, list) or any(type(p) is not int for p in retired)
            or len(set(retired)) != len(retired)
            or any(p < 0 or p >= len(old['uv']) for p in retired)
            or set(points) & set(retired)
            or sorted(points + retired) != list(range(len(old['uv'])))):
        raise ValueError('Painted-line centre semantics must review every original point as measured or explicitly retired')
    ref = record.get('source_point_qualification_reference')
    if retired:
        if not isinstance(ref, dict):
            raise ValueError('Explicit retired paint points require exact source point qualification reference')
        path = _reference(ref)
        key = (str(path), ref['sha256'])
        if key not in qualifications:
            from calibration.solver_runtime.original_source_point_qualification import OriginalSourcePointQualification
            qualifications[key] = OriginalSourcePointQualification(path)
        authority = qualifications[key]
        rows = [r for r in authority.records if r['item']['original_packet_row_index'] == record['original_packet_row_index']]
        if (authority.contract['original_packet']['sha256'] != parent_sha
                or len(rows) != 1 or set(rows[0]['record']['excluded_point_indices']) != set(retired)
                or _plain(rows[0]['original']) != _plain(old)
                or _plain({k: rows[0]['item'][k] for k in ('block', 'collection_role', 'group_actual_pts', 'group_order', 'row_order')})
                   != _plain({k: item[k] for k in ('block', 'collection_role', 'group_actual_pts', 'group_order', 'row_order')})):
            raise ValueError('Retired painted points must exactly match this immutable original row qualification')
        return retired, dict(path=str(path), sha256=ref['sha256'])
    if ref is not None:
        raise ValueError('A paint row without retired points cannot claim an unused qualification reference')
    return [], None


def install_original_ice_pixel_correction(state, packet_path):
    """Install before actors/source capture; keep the complete row layout."""
    path = Path(packet_path).resolve()
    packet = json.loads(path.read_text(encoding='utf8'))
    if (packet.get('schema') != PACKET_SCHEMA or packet.get('source_video_sha256') != SOURCE_SHA
            or packet.get('time_base') != '1/15360'
            or type(packet.get('model_projection_used_for_selection')) is not bool):
        raise ValueError('Explicit same-video source-only ice pixel correction required')
    guided = (packet.get('model_projection_used_for_selection') is True
              or packet.get('model_guided_navigation') is True)
    if guided:
        if (packet.get('model_guided_navigation') is not True
                or packet.get('observed_UV_generated_from_model') is not False
                or packet.get('raw_source_pixels_read') is not True
                or not isinstance(packet.get('model_guidance_reference'), dict)):
            raise ValueError('Model navigation requires source-only RGB measurements, never projected UV')
        _reference(packet['model_guidance_reference'])
    authority = _reference(packet['original_absolute_packet'])
    original = pickle.loads(authority.read_bytes())['programme_absolute_rows']
    evidence = packet.get('source_evidence', [])
    if not evidence:
        raise ValueError('Original RGB evidence required for ice pixel correction')
    evidence_pts = set()
    for ref in evidence:
        _reference(ref)
        pts = ref.get('pts')
        if type(pts) is not int or pts not in state.original.aliases:
            raise ValueError('Ice correction evidence requires an authenticated source PTS')
        evidence_pts.add(int(state.original.aliases[pts]))
    corrections = packet.get('records', [])
    if not corrections:
        raise ValueError('At least one reviewed source correction required')
    plans, history, seen, qualifications = [], [], set(), {}
    blocks = {block.name: block for block in state.blocks}
    for record in corrections:
        index = record.get('original_packet_row_index')
        if type(index) is not int or not 0 <= index < len(original) or index in seen:
            raise ValueError('One exact original packet row per ice correction required')
        seen.add(index)
        item = original[index]
        old = item['source_row']
        actual = int(item['group_actual_pts'])
        if (item['collection_role'] != 'FIT' or old.get('split') not in ('fit', 'development')
                or actual not in evidence_pts
                or not record.get('correction_reason') or not record.get('physical_edge_role')
                or not _source_only_reading(record)):
            raise ValueError('Only evidenced existing FIT ice material pixels may be corrected')
        block = blocks[item['block']]
        group = int(item['group_order']); position = int(item['row_order'])
        pts, rows = block.groups[group]
        if int(pts) != actual:
            raise ValueError('Original ice source group order changed')
        row = rows[position]
        _same_runtime_source(row, old)
        painted_center = _qualified_painted_line_center(state, row, old, record)
        course_top = False if painted_center else _qualified_original_top(state, row, old, record)
        contact = False if painted_center or course_top else _qualified_original_contact(state, row, old, record)
        named_lane = False if painted_center else _qualified_named_lane(row, old, record)
        target_boundary = False if painted_center else _qualified_target_ring_boundary(state, row, old, record)
        ad_corner = False if painted_center else _qualified_advertisement_corner(state, row, old, record)
        if not contact and not course_top and not named_lane and not painted_center and not target_boundary and not ad_corner and not str(old['id']).startswith(('F', 'R')):
            raise ValueError('Only original named ice markings or certified ice-board contact/top may be corrected')
        points = record.get('point_indices')
        if (not isinstance(points, list) or not points or any(type(p) is not int for p in points)
                or len(set(points)) != len(points) or min(points) < 0 or max(points) >= len(row['uv'])):
            raise ValueError('Explicit unique original point indices required')
        _verify_record_original(record, item, old, points)
        retired, qualification_ref = [], None
        if painted_center:
            retired, qualification_ref = _verify_painted_row_coverage(record, item, old, points,
                qualifications, packet['original_absolute_packet']['sha256'])
        elif record.get('retired_original_point_indices'):
            raise ValueError('Explicit row retirement coverage is only used for whole painted-line centre review')
        uv = np.asarray(record['corrected_uv'], dtype=float)
        support_literal = record.get('corrected_normal_interval_px')
        update_support = support_literal is not None
        support = np.asarray(support_literal, dtype=float) if update_support else None
        normal = np.asarray(row.get('normal'), dtype=float)
        if (uv.shape != (len(points), 2) or not np.isfinite(uv).all() or np.any(uv < 0)
                or np.any(uv >= [1920., 1080.])
                or (not contact and not course_top and not named_lane and not painted_center and not target_boundary and not ad_corner and not update_support)
                or (update_support and (support.shape != (len(points), 2)
                    or not np.isfinite(support).all() or np.any(support[:, 0] > 0)
                    or np.any(support[:, 1] < 0) or 'normal_interval' not in row
                    or normal.shape != np.asarray(row['uv']).shape
                    or not np.allclose(np.linalg.norm(normal, axis=1), 1., rtol=0, atol=1e-6)))):
            raise ValueError('Real native pixels and original unit-normal optical support required')
        revised = deepcopy(row)
        revised['uv'] = np.asarray(row['uv'], dtype=float).copy()
        revised['uv'][points] = uv
        if update_support:
            revised['normal_interval'] = np.asarray(row['normal_interval'], dtype=float).copy()
            revised['normal_interval'][points] = support
        if painted_center:
            footprint = {key: _plain(row[key]) for key in ('normal_interval', 'normal_interval_px') if key in row}
            revised.pop('normal_interval', None)
            revised.pop('normal_interval_px', None)
            revised['source_paint_footprint_evidence'] = dict(
                original_source_fields=footprint,
                measured_paint_footprint_evidence=deepcopy(record.get('paint_footprint_evidence')),
                paint_footprint_is_not_location_uncertainty=True)
            revised['measurement_role'] = 'painted_line_center'
        for alias in ('source_uv',):
            if alias in row and np.array_equal(row[alias], row['uv']):
                revised[alias] = revised['uv'].copy()
        revised['active_source_pixel_correction'] = dict(packet_path=str(path), packet_sha256=_sha(path),
            original_packet_row_index=index, point_indices=points,
            original_uv=_plain(row['uv']), original_normal_interval=_plain(row.get('normal_interval')),
            original_contact_material_preserved=contact, optical_support_updated=update_support,
            original_top_material_preserved=course_top,
            original_course_z_policy=record.get('original_course_z_policy'),
            original_named_lane_identity_preserved=named_lane,
            original_target_ring_boundary_preserved=target_boundary,
            original_advertisement_corner_identity_preserved=ad_corner,
            measurement_role=record.get('measurement_role'),
            original_geometry_identity=deepcopy(record.get('original_geometry_identity')),
            original_normal_interval_removed_from_operator=painted_center,
            paint_footprint_is_not_location_uncertainty=painted_center,
            retired_original_point_indices=retired,
            source_point_qualification_reference=qualification_ref,
            retired_points_require_verified_physical_input_compaction=bool(retired),
            physical_edge_role=record['physical_edge_role'], correction_reason=record['correction_reason'],
            old_source_profiles_are_history_not_current_pixel_evidence=True)
        # Prepare every row first; a later failed record cannot partially alter state.
        plans.append((block, group, position, revised))
        history.append(dict(original_packet_row_index=index,
            locator={key: item[key] for key in ('block', 'collection_role', 'group_actual_pts', 'group_order', 'row_order')},
            original_source_row=_plain(old), point_indices=points, corrected_uv=uv.tolist(),
            corrected_normal_interval_px=None if support is None else support.tolist(),
            optical_support_updated=update_support, original_contact_material_preserved=contact,
            original_top_material_preserved=course_top,
            original_course_z_policy=record.get('original_course_z_policy'),
            original_named_lane_identity_preserved=named_lane,
            original_target_ring_boundary_preserved=target_boundary,
            original_advertisement_corner_identity_preserved=ad_corner,
            measurement_role=record.get('measurement_role'),
            original_geometry_identity=deepcopy(record.get('original_geometry_identity')),
            original_normal_interval_removed_from_operator=painted_center,
            paint_footprint_is_not_location_uncertainty=painted_center,
            retired_original_point_indices=retired,
            source_point_qualification_reference=qualification_ref,
            retired_points_require_verified_physical_input_compaction=bool(retired),
            paint_footprint_evidence=deepcopy(record.get('paint_footprint_evidence')),
            physical_edge_role=record['physical_edge_role'],
            correction_reason=record['correction_reason'], original_sigma_and_material_and_role_preserved=True))
    for block, group, position, revised in plans:
        pts, rows = block.groups[group]
        copied = list(rows); copied[position] = revised
        groups = list(block.groups); groups[group] = (pts, copied)
        block.groups = groups
    contract = dict(schema=PACKET_SCHEMA, packet_path=str(path), packet_sha256=_sha(path),
        original_absolute_packet=packet['original_absolute_packet'], source_video_sha256=SOURCE_SHA,
        time_base='1/15360', source_evidence=evidence, corrected_rows=history,
        model_guided_navigation=guided,
        model_guidance_reference=packet.get('model_guidance_reference') if guided else None,
        observed_UV_generated_from_model=False,
        original_packet_bytes_untouched=True, original_HELD_not_fitted=True,
        original_world_sigma_roles_source_order_masks_and_scalar_slots_unchanged=True,
        active_FIT_source_UV_and_optical_support_explicitly_revised=True,
        fresh_complete_source_profiles_and_J_required=True, accuracy_accepted=False)
    state.original_ice_pixel_correction = contract
    # Feedback is a post-transaction observer. Its derived index must never
    # invalidate rows already installed by the authenticated transaction above.
    try:
        from rinkspace.feedback_outbox import record_source_installation
        project = next(parent for parent in Path(__file__).resolve().parents if (parent/'configs/storage.json').is_file())
        feedback = record_source_installation(project, path, deepcopy(contract))
    except Exception as error:
        feedback = dict(status='retry_pending', error=type(error).__name__+': '+str(error),
            runtime_rows_installed=True, packet_reference=dict(path=str(path),sha256=contract['packet_sha256']),
            retry='replay record_source_installation with the preserved contract',
            observation_dataset_promoted=False, visual_accuracy_verified=False)
    state.original_ice_pixel_feedback = feedback
    return contract


def review_corrected_ice_pixels(state, q):
    """Evaluate the actual corrected rows; history errors are not fit accuracy."""
    contract = state.original_ice_pixel_correction
    blocks = {block.name: block for block in state.blocks}
    results = []
    for entry in contract['corrected_rows']:
        locator = entry['locator']; block = blocks[locator['block']]
        pts, rows = block.groups[int(locator['group_order'])]
        row = rows[int(locator['row_order'])]
        world = q[block.geometry_parameter_indices]
        bound = block.geometry.rebind(row, world)
        camera = block.bundle.camera(block.local(q), int(pts))
        outside = np.asarray(block.absolute(camera, [bound], False)).reshape(-1, 2) * row['sigma'][:, None]
        raw_row = dict(bound); raw_row.pop('normal_interval', None)
        raw = np.asarray(block.absolute(camera, [raw_row], False)).reshape(-1, 2) * row['sigma'][:, None]
        points = entry['point_indices']
        from calibration.solver_runtime.original_absolute_input_compaction import original_points_to_active
        active_points=original_points_to_active(row,points)
        if not np.array_equal(row['uv'][active_points],np.asarray(entry['corrected_uv'])):
            raise ValueError('Current active corrected source pixels differ from their original point authority')
        results.append(dict(locator=locator, point_indices=points,actual_active_point_indices=active_points.tolist(),
            source_uv=row['uv'][active_points].tolist(),
            raw_vector_distance_px=raw[active_points].tolist(), outside_optical_support_vector_px=outside[active_points].tolist(),
            mean_raw_distance_px=float(np.mean(np.linalg.norm(raw[active_points], axis=1))),
            max_outside_optical_support_px=float(np.max(np.linalg.norm(outside[active_points], axis=1))),
            source_pixels_are_not_accepted_camera_accuracy=True))
    return dict(contract=contract, actual_corrected_FIT_source_metrics=results,
                original_error_records_preserved_not_counted_as_corrected_alignment=True, accuracy_accepted=False)


def require_painted_retirement_compaction(contract, qualification_path, compaction_path):
    """Reject an omitted production installer, even if the row cites a proof."""
    declared = [r for r in contract['corrected_rows']
                if r.get('retired_points_require_verified_physical_input_compaction')]
    if not declared:
        return
    if qualification_path is None or compaction_path is None:
        raise ValueError('Retired painted source points require selected qualification and physical input compaction')
    selected = dict(path=str(Path(qualification_path).resolve()), sha256=_sha(qualification_path))
    if any(r['source_point_qualification_reference'] != selected for r in declared):
        raise ValueError('Selected painted source qualification differs from the exact row retirement authority')
