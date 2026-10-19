"""Collect the live qualified FIT pixel input, never historical packets or q.

The input is the assembled production state after correction/compaction and
the installed typed factors. Interval-only constraints have no invented pixel
centre. A disabled or nonpixel family still has an explicit receipt.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np


PIXEL_FACTORS = ('native_ice_factor', 'native_board_factor',
                'native_board_material_point_factor', 'circle_factor',
                'native_board_track_factor', 'persistent_ice_factor')


def collect_current_active_source_points(state, source, wide_factor):
    aliases = {int(k): int(v) for k, v in state.original.aliases.items()}
    allowed = set(int(p) for p in state.bypts)
    if not allowed:
        raise ValueError('An actual programme exposure inventory is required')
    lo, hi = min(allowed), max(allowed)
    groups, receipts = defaultdict(list), []

    def receipt(name, **detail):
        value = dict(family=name, complete_active_input=True, pixel_input_count=0,
                     outside_programme_pixel_count=0, excluded_nonpixel_rows=0,
                     excluded_retired_pixel_count=0, excluded_HELD_rows=0, **detail)
        receipts.append(value)
        return value

    def actual(row, native_key='pts', actual_key='actual_pts'):
        declared = row.get(actual_key)
        native = row.get(native_key)
        if native is not None and int(native) in aliases:
            resolved = aliases[int(native)]
            if declared is not None and int(declared) != resolved:
                raise ValueError('Active pixel source disagrees with its original exposure alias')
            return resolved
        if declared is not None:
            return int(declared)
        if native is not None:
            return int(native)
        raise ValueError('Active pixel row has no original source clock')

    def add(rec, pts, uv):
        values = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        if (not np.isfinite(values).all() or np.any(values < 0)
                or np.any(values >= [1920, 1080])):
            raise ValueError('An active FIT source contains invalid native pixel coordinates')
        pts = int(pts)
        if pts not in allowed:
            if lo <= pts <= hi:
                raise ValueError('Source PTS inside programme has no actual camera exposure')
            rec['outside_programme_pixel_count'] += len(values)
            return
        rec['pixel_input_count'] += len(values)
        if len(values):
            groups[pts].append(values)

    def nonpixel(row):
        return (row.get('source_uv_is_image_boundary_context_only') is True
                or row.get('no_visible_tangent_UV_created') is True
                or row.get('curve_observation_kind') in ('horizontal_extent', 'clipped_horizontal_extent'))

    def row_pixels(rec, row, pts=None, mask=None):
        if row.get('split', 'fit') not in ('fit', 'FIT', 'development', 'development_source'):
            raise ValueError('A typed active FIT view contains a HELD or unknown split')
        if nonpixel(row):
            rec['excluded_nonpixel_rows'] += 1
            return
        uv = np.asarray(row['uv'], np.float64).reshape(-1, 2)
        if mask is not None:
            rec['excluded_retired_pixel_count'] += len(uv)-int(mask.sum())
            uv = uv[mask]
        add(rec, actual(row) if pts is None else pts, uv)

    # These scalar masks are actual runtime neutralizations when a legacy run
    # retains slots. Compacted runs expose empty masks and already removed UV.
    retired = set()
    for attr, key in (('original_role_retirement', 'active_global_rows'),
                      ('static_motion_role_retirement', 'global_rows')):
        factor = getattr(source, attr, None)
        if factor is not None:
            retired.update(map(int, np.asarray(getattr(factor, key)).ravel()))
    qualification = getattr(source, 'original_point_qualification', None)
    if qualification is not None:
        retired.update(map(int, qualification.contract['active_runtime_global_scalar_slots']))
    cursor = 0
    for block in state.blocks:
        bundle = block.bundle
        motion = receipt('programme:' + block.name + ':motion', input='live_compacted_bundle_a_b')
        motion_keep = np.ones(len(bundle.a), bool)
        local_retired = np.asarray([p-cursor for p in retired if cursor <= p < cursor+2*len(bundle.a)], int)
        if len(local_retired):
            # A source point remains real if either pixel component is active.
            inactive = np.zeros(2*len(bundle.a), bool)
            inactive[local_retired] = True
            motion_keep = ~inactive.reshape(-1, 2).all(1)
        motion['excluded_retired_pixel_count'] = 2*int((~motion_keep).sum())
        stamps = np.asarray(bundle.stamps, np.int64)
        for field, endpoint in (('a', 'ia'), ('b', 'ib')):
            uv = np.asarray(getattr(bundle, field))[motion_keep]
            index = np.asarray(getattr(bundle, endpoint))[motion_keep]
            if len(index) != len(uv) or (len(index) and (index.min() < 0 or index.max() >= len(stamps))):
                raise ValueError('Live motion endpoint clock arrays differ')
            order = np.argsort(index, kind='stable')
            sorted_index = index[order]
            starts = np.r_[0, np.flatnonzero(np.diff(sorted_index))+1, len(order)]
            for a, b in zip(starts[:-1], starts[1:]):
                if a < b:
                    add(motion, stamps[sorted_index[a]], uv[order[a:b]])
        absolute = receipt('programme:' + block.name + ':absolute', input='live_corrected_compacted_block_groups')
        start = cursor + int(bundle.motion_rows)
        for pts, rows in block.groups:
            for row in rows:
                n = len(row['uv'])
                mask = np.array([not {start+2*i, start+2*i+1}.issubset(retired) for i in range(n)], bool)
                row_pixels(absolute, row, int(pts), mask)
                start += 2*n
        cursor += len(block.quadratic)

    collected_factors = {}
    for name in PIXEL_FACTORS:
        factor = getattr(source, name, None)
        rec = receipt(name, installed=factor is not None)
        if factor is None:
            continue
        if id(factor) in collected_factors:
            rec.update(shared_object_alias=True, alias_of=collected_factors[id(factor)])
            continue
        collected_factors[id(factor)] = name
        rec['excluded_HELD_rows'] = len(getattr(factor, 'held', []))
        rec['contract_schema'] = factor.contract.get('schema')
        if name == 'native_ice_factor':
            qualified = factor._qualified_active_fit_view()
            active = factor if qualified is None else qualified
            rec['qualified_active_fit_view_used'] = True
            rec['excluded_retired_pixel_count'] = sum(len(r['uv']) for r in factor.rows)-sum(len(r['uv']) for r in active.rows)
            for row in active.rows:
                row_pixels(rec, row)
        elif name == 'native_board_track_factor':
            for row in factor.fit_evaluation_rows:
                if row.get('_anchor'):
                    row_pixels(rec, row)
                else:
                    add(rec, row['source_actual_pts'], row['uv0'])
                    add(rec, row['target_actual_pts'], row['uv1'])
            rec['explicit_current_anchor_rows_included'] = len(factor.anchor_rows)
        elif name == 'persistent_ice_factor':
            for row in factor.rows:
                add(rec, row['anchor_endpoint']['actual_pts'], row['anchor_uv'])
                add(rec, row['target_endpoint']['actual_pts'], row['target_uv'])
        else:
            for row in factor.rows:
                row_pixels(rec, row)

    patterns = getattr(state, 'patterns', None)
    rec = receipt('shared_board_patterns', installed=patterns is not None)
    if patterns is not None:
        rec['excluded_HELD_rows'] = len(patterns.held)
        for frame in patterns.fit:
            pts = frame['source_actual_pts']
            key = 'source_curve_support' if frame['physical_ad_id'] in (13, 26) else 'source_pattern_points'
            for row in frame[key]:
                add(rec, pts, row['uv'])

    external = getattr(state, 'external', None)
    rec = receipt('external_main_low', installed=external is not None)
    if external is not None:
        for row in external.track_rows:
            if row['split'] == 'fit':
                add(rec, row['actual_pts'], row['target_uv_native'])
            else:
                rec['excluded_HELD_rows'] += 1
        for anchor in external.track_sources:
            add(rec, anchor['anchor_actual_pts'], anchor['anchor_uv_native'])
        for name in ('upper_rows', 'head_rows', 'ad14_rows'):
            # These producers use measured scalar intervals/rectangles rather
            # than an exact UV. In particular nominal_uv_search_only is never
            # a displayed measurement and no support midpoint is invented.
            for row in getattr(external, name):
                if row['split'] == 'fit':
                    rec['excluded_nonpixel_rows'] += 1
                else:
                    rec['excluded_HELD_rows'] += 1
    rec = receipt('physical_cross_target_support')
    rec['excluded_nonpixel_rows'] = len(state.cross_fit)
    if any('native_support_bounds_xy_px' not in row['factor_source'] for row in state.cross_fit):
        raise ValueError('Unknown cross-target source needs an explicit pixel collector')

    rec = receipt('wide_source', input='installed_original_wide_and_user_source_pixels')
    pts = int(wide_factor.contract['source_pts'])
    for uv in wide_factor.uvs:
        add(rec, pts, uv)
    for row in wide_factor.ad_packet['rows']:
        add(rec, pts, row['source_frame_uv'])
    for uv in wide_factor.trace_uv:
        add(rec, pts, uv)
    rec['excluded_nonpixel_rows'] = 1  # fixed-camera entity support rectangle
    rec = receipt('reference_geometry_and_camera_priors',
                  derived_full_stroke_guard_queries_are_not_additional_pixel_inputs=True)
    rec['excluded_nonpixel_rows'] = 18
    # The manual-centre secondary objective reuses current sparse packet UV.
    # Its separate factor view must still be enumerated, including qualified
    # native ice samples. Frozen storage deduplicates identical coordinates.
    manual = receipt('direct_manual_centre_secondary', installed=False)
    pixel_fields = {'uv', 'uv0', 'uv1', 'anchor_uv', 'target_uv',
                    'uv_native', 'target_uv_native', 'source_frame_uv'}
    # Fail when a future wrapper installs another pixel-bearing factor;
    # adding a source schema requires an explicit collector above.
    owner, seen = source, set()
    while owner is not None and id(owner) not in seen:
        seen.add(id(owner))
        for factor in vars(owner).get('factors', []):
            if id(factor) in collected_factors:
                manual.setdefault('shared_object_aliases', []).append(collected_factors[id(factor)])
                continue
            if not factor.contract.get('direct_manual_centre_secondary_objective'):
                raise ValueError('Unenumerated source factor collection')
            manual['installed'] = True
            collected_factors[id(factor)] = manual['family']
            qualified = factor._qualified_active_fit_view() if hasattr(factor, '_qualified_active_fit_view') else None
            active = factor if qualified is None else qualified
            for row in active.rows:
                row_pixels(manual, row)
        for name, value in vars(owner).items():
            if name.endswith('_factor') and name not in PIXEL_FACTORS and name != 'reference_factor':
                if id(value) in collected_factors:
                    receipt(name, installed=True, shared_object_alias=True,
                            alias_of=collected_factors[id(value)])
                    continue
                if any(pixel_fields.intersection(row) for row in getattr(value, 'rows', []) if isinstance(row, dict)):
                    raise ValueError('Unenumerated installed pixel source family: '+name)
        owner = vars(owner).get('base', vars(owner).get('parent'))
    point_groups = [(pts, np.concatenate(groups[pts])) for pts in sorted(groups)]
    return point_groups, receipts, np.asarray(sorted(allowed), np.int64)
