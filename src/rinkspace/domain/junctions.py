"""Explicit fixed-junction surface relations and native current-scene binding."""
from .contracts import ContractError, Feature, validate_feature_registry


def evaluate_ice_board_junction(feature, features, scene, core, *, path_ids=None):
    """Resolve current parent geometry, then let native establish the junction.

    Parent path IDs bind real scene geometry; labels alone cannot promote a top
    edge or a screen crossing. Candidate scene values are evaluated afresh.
    """
    from ..compat.d78_entities import native_definition, finite_path_support
    from .board_station import validate_boundary_definition
    if feature.kind != 'fixed_junction' or feature.junction_relation != 'ice_board_bottom':
        raise ContractError('Expected an explicit ice_board_bottom fixed junction')
    validate_feature_registry({feature.feature_id: feature,
                              **{p: features[p] for p in feature.parents if p in features}})
    if core is None:
        raise ContractError('ice_board_bottom requires the current native core for numerical validation')
    paths = {p['id']: p for p in scene.get('paths', [])}
    if len(paths) != len(scene.get('paths', [])):
        raise ContractError('Duplicate current scene path identities')
    resolved = {}
    for identity in feature.parents:
        parent = features[identity]
        path_id = (path_ids or {}).get(identity, identity)
        if path_id not in paths:
            raise ContractError('Junction parent lacks its current scene path: '+identity)
        path = paths[path_id]
        if path.get('layer') != parent.surface:
            raise ContractError('Junction parent surface differs from its current scene path')
        kind, parameters = native_definition(scene, path, core)
        if parent.surface == 'ice':
            if kind != 0:
                raise ContractError('ice_board_bottom requires a finite native ice line')
            support = finite_path_support(path)
            if len(support) != 2 or not 0 <= support[0] < support[1] <= 1:
                raise ContractError('Invalid finite ice line support')
            # Native path sampling applies the actual finite model support.
            points = core.sample_path(scene['board'], kind, parameters, [], support)
            resolved['line'] = [value for point in points for value in point]
        else:
            if path.get('label') != 'board.bottom' or kind != 7 or len(parameters) != 1:
                raise ContractError('Junction board parent is not the current continuous board.bottom')
            validate_boundary_definition(path)
            resolved['height_fraction'] = parameters[0]
    return core.ice_board_junction(scene['board'], resolved['line'],
                                   resolved['height_fraction'], int(feature.branch))


def validate_junction_records(scene, records, arrays, rows, *, core=None, print_bindings=None):
    """Shared prepare/load gate for fixed-junction records, including inactive rows."""
    import numpy as np
    by_id = {record['id']: record for record in records}
    if len(by_id) != len(records):
        raise ContractError('Duplicate feature identities')
    for record in records:
        if record.get('junction_relation') is not None and record.get('kind') != 'fixed_junction':
            raise ContractError('Surface relations only apply to fixed junctions')
    junctions = [r for r in records if r.get('kind') == 'fixed_junction']
    if not junctions:
        return {'fixed_junctions': 0, 'ice_board_bottom': 0}
    wanted = {r['id'] for r in junctions}
    wanted.update(p for r in junctions for p in r.get('parents', []))
    if not wanted <= set(by_id):
        raise ContractError('Missing junction parent feature')
    features = {}
    for identity in wanted:
        r = by_id[identity]
        branch = r.get('branch')
        if type(branch) is int:
            branch = str(branch)
        features[identity] = Feature(identity, r.get('kind'), r.get('surface'),
            r.get('identity'), r.get('measurement'), tuple(r.get('parents', ())),
            branch, r.get('has_station_parameter', False), r.get('label'),
            (print_bindings or {}).get(identity), r.get('junction_relation'))
    validate_feature_registry(features)
    crossed = 0
    for r in junctions:
        index = next(i for i, record in enumerate(records) if record['id'] == r['id'])
        selected = np.asarray(rows)[arrays['feature_index'][rows] == index]
        if (np.any(arrays['kind'][selected] != 0) or
                np.any(arrays['local_index'][selected] != -1) or
                np.any(arrays['support'][selected] != 0) or r.get('station_domain') or
                r.get('per_frame_station_parameter', False) or
                np.any(arrays.get('support_kind', np.zeros(len(arrays['pts'])))[selected] != 0)):
            raise ContractError('Fixed junctions cannot acquire per-observation stations')
        feature = features[r['id']]
        if np.any(arrays['active'][selected]):
            if feature.identity != 'confirmed' or any(features[p].identity != 'confirmed' for p in feature.parents):
                raise ContractError('Active junction has unresolved or rejected parents')
        if feature.junction_relation == 'ice_board_bottom':
            evaluate_ice_board_junction(feature, features, scene, core,
                path_ids={identity: by_id[identity].get('path_id', identity) for identity in wanted})
            crossed += 1
    return {'fixed_junctions': len(junctions), 'ice_board_bottom': crossed}
