"""Candidate-scene checks recomputed from exported geometry only.

Historical base checks remain traceable and are not candidate validation.
"""
from __future__ import annotations

import copy
import numpy as np


_GEOMETRY_TOLERANCE_SCENE = 1e-8
_RING_ROLES = ('outer', 'middle', 'inner', 'center')


def audit_exported_circle_geometry(candidate: dict, baseline: dict,
                                   base_hash: str) -> dict:
    """Check only facts decidable from the exported candidate geometry.

    The old shared_geometry_checks includes topology and historical claims
    about the *ring-wrapper operation*. Those claims are not silently reused
    as current-scene validation after a programme ring calibration changes
    centres and radii. Equality of the eight F/R radii is reported as a strict
    geometric fact, not an acceptance gate: small differences are permitted,
    with no numeric acceptance tolerance specified. Only a failed hard
    geometric invariant makes the aggregate false; incomplete review remains
    unknown (JSON null).
    """
    old = copy.deepcopy(baseline.get('shared_geometry_checks', {}))
    features = {row['id']: row for row in candidate['ice_features']}
    circles = {name: row for name, row in features.items()
               if row.get('kind') == 'circle'}
    tolerance = _GEOMETRY_TOLERANCE_SCENE

    def centre(row):
        try:
            value = np.asarray(row['center_xyz'], float)
        except (KeyError, TypeError, ValueError):
            return None
        return value if value.shape == (3,) and np.isfinite(value).all() else None

    def radius(row):
        try:
            value = float(row['radius'])
        except (KeyError, TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def outer_radius(name):
        feature = circles.get(name)
        if feature is None:
            return None
        rings = {r.get('role'): r for r in feature.get('rings', [])}
        return radius(rings['outer']) if 'outer' in rings else radius(feature)

    def equal_radii(names):
        values = [outer_radius(name) for name in names]
        return (all(value is not None for value in values) and
                max(values)-min(values) <= tolerance)

    all_radii = [radius(feature) for feature in circles.values()]
    all_radii.extend(radius(ring) for feature in circles.values()
                     for ring in feature.get('rings', []))
    circle_centres = [centre(feature) for feature in circles.values()]
    ring_centres = [centre(ring) for feature in circles.values()
                    for ring in feature.get('rings', [])]
    fr_names = [f'F{i}' for i in range(1, 5)]+[f'R{i}' for i in range(1, 5)]
    fr_outer_radii = {name: outer_radius(name) for name in fr_names}
    values = list(fr_outer_radii.values())
    valid_fr = all(value is not None and value > 0 for value in values)
    min_radius = min(values) if valid_fr else None
    max_radius = max(values) if valid_fr else None
    radius_span = max_radius-min_radius if valid_fr else None
    mean_radius = float(np.mean(values)) if valid_fr else None
    ring_nesting = {}
    for name in fr_names:
        feature = circles.get(name)
        rings = {r.get('role'): r for r in feature.get('rings', [])} if feature else {}
        for outer_role, inner_role in zip(_RING_ROLES, _RING_ROLES[1:]):
            outer, inner = rings.get(outer_role), rings.get(inner_role)
            co, ci = centre(outer) if outer else None, centre(inner) if inner else None
            ro, ri = radius(outer) if outer else None, radius(inner) if inner else None
            ring_nesting[f'{name}:{inner_role}_inside_{outer_role}'] = bool(
                co is not None and ci is not None and ro is not None and ri is not None
                and ro > 0 and ri > 0 and abs(co[2]) <= tolerance
                and abs(ci[2]) <= tolerance
                and np.linalg.norm(co[:2]-ci[:2])+ri <= ro+tolerance)

    evaluated = {
        'circle_radii_remain_positive': bool(
            all_radii and all(value is not None and value > 0 for value in all_radii)),
        'circle_centers_remain_on_ice_z0': bool(
            circle_centres and all(value is not None and abs(value[2]) <= tolerance
                                   for value in circle_centres)),
        'eight_F_R_outer_radii_remain_equal': bool(equal_radii(fr_names)),
        'A1_A2_outer_radii_remain_equal': bool(equal_radii(['A1', 'A2'])),
        'each_ring_nested_without_crossing': bool(
            len(ring_nesting) == 24 and all(ring_nesting.values())),
        'ring_centers_remain_on_ice_z0': bool(
            ring_centres and all(value is not None and abs(value[2]) <= tolerance
                                 for value in ring_centres)),
    }
    old_names = list(old.get('checks', {}))
    checks = {name: evaluated.get(name) for name in old_names}
    checks.update({name: value for name, value in evaluated.items()
                   if name not in checks})
    unknown = {
        name: ('Historical ring-wrapper operation, not a present-scene constraint'
               if 'ring_wrapper' in name else
               'Needs separate topology or source validation; not recomputed here')
        for name, value in checks.items() if value is None
    }
    hard_names = ('circle_radii_remain_positive',
                  'circle_centers_remain_on_ice_z0',
                  'each_ring_nested_without_crossing',
                  'ring_centers_remain_on_ice_z0')
    hard_geometry_valid = all(checks[name] is True for name in hard_names)
    # Strict F/R equality is a measured fact; the user allows small variation
    # without specifying a numeric threshold. It is therefore not a hard gate.
    result = False if not hard_geometry_valid else None
    return {
        'profile': 'programme_candidate_partial_geometry_audit',
        'all_constraints_satisfied': result,
        'validation_complete': False,
        'hard_geometry_valid': hard_geometry_valid,
        'hard_geometry_check_names': list(hard_names),
        'checks': checks,
        'eight_F_R_outer_radius_audit': {
            'radius_by_feature_scene_units': fr_outer_radii,
            'minimum_scene_units': min_radius,
            'maximum_scene_units': max_radius,
            'range_scene_units': radius_span,
            'relative_range_to_mean': (radius_span/mean_radius if valid_fr else None),
            'strict_equal_with_numeric_tolerance':
                checks['eight_F_R_outer_radii_remain_equal'],
            'numeric_equality_tolerance_scene_units': tolerance,
            'user_policy': 'small_variation_permitted_without_numeric_threshold',
            'tolerance_acceptance': 'not_evaluated_unspecified',
            'not_a_hard_geometry_gate': True,
        },
        'unknown_check_reasons': unknown,
        'ring_nesting': ring_nesting,
        'tolerance_scene_units': tolerance,
        'baseline_historical': {
            'base_model_sha256': base_hash,
            'shared_geometry_checks': old,
            'not_revalidated_as_candidate': True,
        },
    }
