"""Read-only source quantity, exact duplication and material weight inventory."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import numpy as np

from rinkspace.source_sampling import source_support_masses
from .strict_absolute_source import _target_key
from .strict_source_representatives import _declared_paths


def source_working_set_inventory(pack):
    """Describe raw evidence and a separate uninstalled per-material budget.

    No pixel is removed here. Exact duplicate eligibility follows the public
    merge identity and prefers measured working support over missing support.
    A group budget of one is diagnostic; it does not silently change the
    current consumer's loss, source sigma, HELD membership or fit admission.
    """
    groups = defaultdict(lambda: dict(rows=0, points=0, finite_points=0,
        raw_inverse_working_sigma_squared_sum=0., raw_path_mass=0.,
        raw_support_weighted_inverse_sigma_squared_sum=0., splits=set(), roles=set()))
    material = defaultdict(lambda: dict(points=0, finite_points=0, rows=0, exposures=set()))
    choices = {}
    duplicates = 0
    conflicting_split_duplicates = 0
    held = fit = missing = 0
    raw_sigma_min, raw_sigma_max = np.inf, -np.inf
    payload_order = hashlib.sha256(b'rinkspace.exact-source-point-payload-order.v1')
    for name in ('row_offsets', 'uv', 'sigma_px', 'source_sides_uv',
            'raw_candidate_uv', 'source_side_quantiles_uv', 'original_point_indices'):
        value = np.ascontiguousarray(pack.values[name])
        payload_order.update(name.encode('ascii'))
        payload_order.update(str(value.dtype).encode('ascii'))
        payload_order.update(str(value.shape).encode('ascii'))
        payload_order.update(value.data)
    per_row = []
    for ri, h in enumerate(pack.rows):
        lo, hi = map(int, pack.values['row_offsets'][ri:ri+2])
        uv = pack.values['uv'][lo:hi]
        sigma = pack.values['sigma_px'][lo:hi]
        valid = np.isfinite(sigma) & (sigma > 0)
        n = len(uv)
        if h['split']=='held':
            held += n
        else:
            fit += n
        missing += int((~valid).sum())
        if valid.any():
            raw_sigma_min = min(raw_sigma_min, float(sigma[valid].min()))
            raw_sigma_max = max(raw_sigma_max, float(sigma[valid].max()))
        group = (h['shot_id'], int(h['actual_pts']), h['scope'], h['physical_material_id'])
        target = _target_key(h['model_target'])
        identity = (h['actual_pts'], h['scope'], h['physical_material_id'], target, h['observation_semantics'])
        payload_order.update(json.dumps((ri, h['shot_id'], h['native_pts'], identity,
            h['split'], h['decoded_BGR_sha256']), separators=(',', ':')).encode('utf8'))
        for i, point in enumerate(uv):
            key = identity, np.asarray(point, np.float64).tobytes()
            if key in choices:
                duplicates += 1
                if choices[key][1] != h['split']:
                    conflicting_split_duplicates += 1
                if valid[i] and not choices[key][0]:
                    choices[key] = bool(valid[i]), h['split']
            else:
                choices[key] = bool(valid[i]), h['split']
        paths, basis = _declared_paths(dict(h, uv=uv))
        # A missing working sigma divides a path; no integration mass bridges
        # a missing/occluded station and no unmeasured station is fabricated.
        supported_paths = []
        for part in paths:
            current = []
            for i in part:
                if valid[i]:
                    current.append(i)
                elif current:
                    supported_paths.append(current)
                    current = []
            if current:
                supported_paths.append(current)
        local = np.flatnonzero(valid)
        if len(local):
            translate = {int(i): j for j, i in enumerate(local)}
            mass = source_support_masses(uv[local], [[translate[i] for i in p] for p in supported_paths])
            _, inverse, repeats = np.unique(uv[local], axis=0, return_inverse=True, return_counts=True)
            mass /= repeats[inverse]
            inverse_sigma = 1/sigma[local]**2
        else:
            mass = inverse_sigma = np.empty(0)
        g = groups[group]
        g['rows'] += 1
        g['points'] += n
        g['finite_points'] += len(local)
        g['raw_inverse_working_sigma_squared_sum'] += float(inverse_sigma.sum())
        g['raw_path_mass'] += float(mass.sum())
        g['raw_support_weighted_inverse_sigma_squared_sum'] += float(mass @ inverse_sigma)
        g['splits'].add(h['split'])
        g['roles'].add(h.get('vision_training_role', 'unspecified'))
        m = material[h['physical_material_id']]
        m['points'] += n
        m['finite_points'] += len(local)
        m['rows'] += 1
        m['exposures'].add((h['shot_id'], h['actual_pts']))
        per_row.append(dict(source_row_index=ri, physical_material_id=h['physical_material_id'],
            shot_id=h['shot_id'], actual_pts=h['actual_pts'], split=h['split'], points=n,
            finite_working_support_points=len(local), path_basis=basis,
            supported_source_paths=len(supported_paths), raw_source_path_mass=float(mass.sum())))
    records = []
    for key, value in sorted(groups.items()):
        record = dict(shot_id=key[0], actual_pts=key[1], scope=key[2], material=key[3],
            **{k:v for k,v in value.items() if k not in ('splits', 'roles')},
            splits=sorted(value['splits']), source_roles=sorted(value['roles']))
        record['proposed_complete_material_exposure_integration_budget'] = 1. if value['raw_path_mass'] > 0 else None
        record['proposed_normalized_support_weighted_inverse_sigma_squared_sum'] = (
            value['raw_support_weighted_inverse_sigma_squared_sum']/value['raw_path_mass']
            if value['raw_path_mass'] > 0 else None)
        records.append(record)
    return dict(schema='rinkspace.readonly-source-working-set-quantity-and-weight-inventory.v1',
        raw_points=len(pack.values['uv']), raw_rows=len(pack.rows),
        source_declared_fit_points=fit, source_declared_held_points=held,
        missing_working_support_points=missing,
        positive_finite_working_sigma_range=None if not np.isfinite(raw_sigma_min) else [raw_sigma_min, raw_sigma_max],
        exact_duplicate_source_pixel_and_physical_target_occurrences=duplicates,
        unique_source_pixels_and_physical_targets=len(choices),
        duplicate_source_pixels_with_conflicting_fit_held_membership=conflicting_split_duplicates,
        exact_deduplication_is_diagnostic_only=True,
        physical_material_exposure_groups=len(groups),
        material_summary={key:dict(**{k:v for k,v in value.items() if k!='exposures'},
            actual_exposures=len(value['exposures'])) for key,value in sorted(material.items())},
        material_exposure_records=records, source_row_records=per_row,
        source_clock_exposures=len(pack.metadata['expected_actual_pts']),
        source_clock_processed_exposures=len(pack.metadata['processed_actual_pts']),
        source_point_order_and_original_pixel_payload_sha256=payload_order.hexdigest(),
        original_source_arrays_changed=False, time_or_held_subsampling=False,
        proposed_weights_installed=False, no_point_count_cap_or_solver_started=True)
