"""Current continuous-board coordinates; frozen legacy branches remain aliases.

No source pixel/PTS, observation ID, statistical group or measurement definition
is rewritten here. The one-time coordinate conversion is evaluated by native.
"""
from copy import deepcopy
import numpy as np


def is_boundary_curve(record):
    """Identify boundary curves, never all board-surface artwork or fixed points."""
    if record.get('kind') in ('material_point', 'fixed_junction'):
        return False
    label = record.get('label', '').removeprefix('entity/')
    return (record.get('native_kind') in (3, 7)
            or record.get('kind') in ('board_boundary', 'board_course')
            or record.get('entity_id') == 'entity/board.continuous'
            or record.get('canonical_geometry_id', '').startswith('board/continuous')
            or label in ('board.bottom', 'board.top'))


def validate_boundary_definition(record):
    """Current boundary stations cannot opt back into a finite segment chart."""
    domain = record.get('station_domain', {})
    if (record.get('native_kind') != 7 or domain.get('kind') != 'unbounded_periodic'
            or domain.get('hard_segment_bounds') is not False):
        raise ValueError('Current board boundary requires native kind7 and unbounded periodic station')
    for key in ('support', 'finite_support'):
        if key in record and record[key] != [0., 0.]:
            raise ValueError('Current board boundary cannot retain finite segment support')


def normalize_boundary_feature(feature, board, legacy_segment, height_fraction,
                               local_values, core, *, local_indices=None,
                               canonical_entity_id=None):
    """Return a copied feature plus converted *current* local state.

    Caller selects this known boundary feature, so advertisements and arbitrary
    board print polylines cannot be reclassified merely from their surface.
    ``local_indices`` are existing station identities, not observation IDs.
    Shared indices are retained. Merging several old aliases must check that
    every shared station identity has one compatible global initial value.
    """
    if feature.get('surface') != 'board' or feature.get('kind') != 'finite_curve':
        raise ValueError('continuous boundary import requires an identified board curve')
    if type(legacy_segment) is not int or not -2 <= legacy_segment <= 7:
        raise ValueError('unsupported legacy board segment chart')
    if not np.isfinite(height_fraction) or not 0 <= height_fraction <= 1:
        raise ValueError('invalid board height fraction')
    if feature.get('measurement') != 'centreline':
        raise ValueError('current board curves allow only the material-width centreline definition')
    values=np.asarray(local_values,dtype='<f8')
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('legacy station state must be a finite vector')
    indices=None
    if local_indices is not None:
        raw=np.asarray(local_indices)
        if raw.dtype.kind not in 'iu' or raw.shape != values.shape:
            raise ValueError('station identity indices must be an integer vector of identical shape')
        indices=np.ascontiguousarray(raw,dtype='<i8')
        if np.any(indices < 0):
            raise ValueError('curve stations require nonnegative state indices')
        for identity in np.unique(indices):
            if np.ptp(values[indices==identity]) > 1e-12:
                raise ValueError('shared station identity has inconsistent legacy values')
    converted=np.asarray(core.board_stations_from_legacy(board,legacy_segment,values),dtype='<f8')
    result=deepcopy(feature)
    original_branch=result.pop('branch',None)
    result.update(native_kind=7,parameters=[float(height_fraction)],has_station_parameter=True,
        coordinate_kind='unwrapped_global_arc_length',finite_support=[0.,0.],
        station_domain={'kind':'unbounded_periodic','period':'board_perimeter','hard_segment_bounds':False,
                        'support_array_role':'zero_placeholder_not_bounds','support_kind':1},
        canonical_geometry_id=canonical_entity_id or ('board/continuous/bottom' if height_fraction==0 else
                                                     'board/continuous/top' if height_fraction==1 else 'board/continuous'),
        legacy_station_alias={'feature_id':feature['id'],'segment':legacy_segment,
                              'height_fraction':float(height_fraction),'branch':original_branch,
                              'original_finite_support':deepcopy(feature.get('finite_support')),
                              'role':'source_identity_and_initial_coordinate_only_not_active_bounds'},
        new_source_observations=0)
    return {'feature':result,'global_station':np.ascontiguousarray(converted),
            'legacy_local_station':values.copy(),'local_indices':indices,
            'support_kind':np.ones(len(values),dtype=np.uint8),
            'support':np.zeros((len(values),2),dtype='<f8')}


def sample_boundary(feature, board, global_station, core):
    """Use the same native boundary evaluator; no per-segment restriction."""
    domain=feature.get('station_domain',{})
    if feature.get('native_kind') != 7 or domain.get('kind') != 'unbounded_periodic' or domain.get('hard_segment_bounds') is not False:
        raise ValueError('feature does not declare continuous unbounded board station')
    return np.asarray(core.sample_path(board,7,feature['parameters'],[],global_station),dtype=np.float64)
