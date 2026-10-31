"""Bounded, native ice-construction deviations around one nominal scene.

The option preserves primitive families, shared attachment nodes and ordering.
All deviations are dimensionless and shared by every frame and physical camera.
"""
from copy import deepcopy
import numpy as np

NOMINAL_SIZE = 39
RELATIVE_LIMIT = .01
# Offset order is part of the finite native layout146 contract.
DEVIATION_GROUPS = (
    ('row_gap', 3, 'nominal row pitch'),
    ('row_centre', 1, 'nominal top/bottom clearance'),
    ('F_column_position', 4, 'nominal row pitch'),
    ('R_column_position', 4, 'nominal row pitch'),
    ('FR_row_alignment', 4, 'nominal row pitch'),
    ('house_ring_radius', 32, 'corresponding nominal ring radius'),
    ('lane_width_at_F', 4, 'nominal Lane width'),
    ('lane_width_at_R', 4, 'nominal Lane width'),
    ('lane_right_position', 4, 'nominal Lane width'),
    ('lane_right_slope', 4, 'dx/dy relative to nominal short-side direction'),
    ('lane_transverse_position', 4, 'nominal Lane width'),
    ('lane_transverse_slope', 4, 'dx/dy relative to nominal short-side direction'),
    ('hockey_carrier_position', 5, 'C: common circle radius; B: blue offset; goals: goal offset'),
    ('hockey_carrier_slope', 5, 'dx/dy relative to nominal short-side direction'),
    ('faceoff_x', 4, 'nominal common hockey circle radius'),
    ('faceoff_y', 4, 'nominal common hockey circle radius'),
    ('hockey_circle_radius', 5, 'nominal common hockey circle radius'),
    ('crease_semiaxes', 4, 'corresponding nominal common semiaxis'),
    ('trapezoid_goal_span', 4, 'nominal goal half span'),
    ('trapezoid_board_span', 4, 'nominal board half span'),
)
DEVIATION_COUNT = sum(n for _, n, _ in DEVIATION_GROUPS)
LAYOUT_SIZE = NOMINAL_SIZE + DEVIATION_COUNT


def deviation_coordinates():
    result = []
    for group, count, reference in DEVIATION_GROUPS:
        for i in range(count):
            result.append(dict(index=len(result), name=f'{group}/{i}', group=group,
                               reference=reference, lower=-RELATIVE_LIMIT, upper=RELATIVE_LIMIT))
    return result


def enable_ice_tolerances(scene):
    """Extend the selected nominal scene with ZERO deviations, without fitting."""
    result = deepcopy(scene)
    if result.get('schema') != 'rinkspace.field_model.v1':
        raise ValueError('Ice tolerances require the current typed native field model')
    if result.get('shared_geometry_options') != 63 or len(result['shared_layout']) != NOMINAL_SIZE:
        raise ValueError('Explicit current layout39 / options63 nominal geometry required')
    if any(b['id'] == 'ice_tolerance' for b in result['parameter_blocks']):
        raise ValueError('Ice tolerance block already exists')
    result['shared_layout'] += [0.] * DEVIATION_COUNT
    result['shared_geometry_options'] = 127
    result['native_family'] = 'shared_geometry_v7_bounded_ice_tolerance'
    result['parameter_blocks'].append(dict(id='ice_tolerance', offset=NOMINAL_SIZE,
        size=DEVIATION_COUNT, tangent_size=DEVIATION_COUNT,
        independent_local_indices=list(range(DEVIATION_COUNT)), positive_local_indices=[],
        values_meaning=[v['name'] for v in deviation_coordinates()]))
    result['ice_tolerance'] = dict(schema='rinkspace.ice_tolerance.v1', native_option=64,
        relative_limit=RELATIVE_LIMIT, deviation_count=DEVIATION_COUNT,
        coordinates=deviation_coordinates(),
        reference_policy='same current common nominal values; not previous iteration; no cumulative rebasing',
        dimension_policy='actual dimension = corresponding nominal dimension * (1 + deviation)',
        position_policy='signed normal displacement divided by declared local nominal length',
        direction_policy='signed transverse drift/run relative to the original rink axis',
        shared_by='entire event scene, all shots and cameras',
        nominal_constraints_ref='constraints describe the nominal template; effective metric ties have the declared tolerance',
        preserved=['line/circle/ellipse primitive family', 'Z=0 ice plane',
            'concentric rings inside each house', 'F/R centres on their connection centreline',
            'Lane centreline is the mean of its two long sides', 'three-sided left-open Lane inventory',
            'right and transverse endpoints on the same Lane long sides',
            'carrier endpoints on the same continuous board',
            'crease centre and diameter on its actual goal line',
            'trapezoid endpoints on their actual goal line and board',
            'nested ring order, row order, C/B/goal order and finite support branches'],
        compound_relations='native signed margins also bound row gaps, clearances, side slopes, widths and collinear-carrier endpoints',
        penalties_added=False, observations_changed=False, fit_executed=False)
    return result


def scene_physical_bounds(state):
    """Intersect caller bounds with these native mandatory deviation bounds."""
    lower = np.full(len(state.values), -np.inf)
    upper = np.full(len(state.values), np.inf)
    for block in state.blocks.values():
        if block['available'] and block['kind'] == 'shared_geometry' and block['id'].endswith('/ice_tolerance'):
            if block['size'] != DEVIATION_COUNT:
                raise ValueError('Native ice deviation block size differs')
            span = slice(block['offset'], block['offset'] + block['size'])
            lower[span] = -RELATIVE_LIMIT
            upper[span] = RELATIVE_LIMIT
    return lower, upper


def migrate_scene_state(old, new):
    """Copy every existing block by identity; the new block remains exactly zero."""
    from .parameters import ParameterState
    values = new.values.copy()
    for identity, block in old.blocks.items():
        if identity not in new.blocks:
            raise ValueError('Scene migration lost a parameter block: ' + identity)
        target = new.blocks[identity]
        if block['available'] != target['available'] or block['size'] != target['size']:
            raise ValueError('Scene migration changed an existing block: ' + identity)
        if block['available']:
            values[target['offset']:target['offset'] + target['size']] = old[identity]
    added = set(new.blocks) - set(old.blocks)
    if len(added) != 1 or not next(iter(added)).endswith('/ice_tolerance'):
        raise ValueError('Migration must add only the bounded ice deviations')
    result = ParameterState(new.layout, values)
    if np.any(result[next(iter(added))] != 0.):
        raise ValueError('New ice deviations must initialize at exact zero')
    return result
