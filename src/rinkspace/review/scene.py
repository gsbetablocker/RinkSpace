"""Sample the current field and its declared prints through the same native core.

This is a representation sampler, not another scene model or a source reader.
Unlocalized instances stay in the inventory without invented geometry.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from ..compat.d78_entities import sample_entities, native_definition
from ..domain.print_assets import resolve_instance, sample_instance
from ..domain.scene_assets import read_scene_assets


def sample_scene(scene, core, scene_path, *, print_stations=None, parameter_state=None):
    if parameter_state is not None and hasattr(parameter_state, 'apply_scene'):
        scene = parameter_state.apply_scene(scene)
    sampled = sample_entities(scene, core)
    chunks = [sampled['xyz']]
    offset = len(sampled['xyz'])
    inventory = sampled['inventory']
    material_ids = list(inventory['material_ids'])
    path_ids = set(inventory['path_ids'])
    registered, localized, unlocalized = [], [], []
    print_grids = {}
    missing = list(sampled['missing_entities'])
    for reference, assets in read_scene_assets(scene, scene_path):
        instance_ids = reference.get('instance_ids', [reference.get('instance_id')])
        for identity in instance_ids:
            instance, template, size, placement = resolve_instance(assets, identity)
            registered.append(identity)
            if placement['values'] is None or size['values'] is None:
                unlocalized.append(identity)
                missing.append({'id': identity, 'template_id': template['id'],
                                'reason': 'unlocalized_print_instance' if placement['values'] is None
                                          else 'uninitialized_print_size',
                                'geometry_fabricated': False})
                continue
            grids = None if print_stations is None else print_stations.get(identity)
            kwargs = {'parameter_state':parameter_state}
            if grids is not None:
                kwargs['stations_by_path'] = grids
            evaluated = sample_instance(assets, identity, scene['board'], core, **kwargs)
            localized.append(identity)
            entity_id = instance.get('entity_id', 'entity/'+identity.rsplit('/', 1)[-1])
            if entity_id not in material_ids:
                material_ids.append(entity_id)
            print_grids[identity] = {}
            for path in evaluated['paths']:
                path_id = identity+'::'+path['id']
                if path_id in path_ids:
                    raise ValueError('Duplicate sampled scene path: '+path_id)
                path_ids.add(path_id)
                xyz = np.asarray(path['xyz'], dtype=np.float64)
                if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
                    raise ValueError('Invalid native print samples: '+path_id)
                print_grids[identity][path['id']] = list(path['stations'])
                sampled['cuts'].append({
                    'id': path_id, 'entity_id': entity_id,
                    'label': identity.rsplit('/', 1)[-1]+'.'+path['id'].rsplit('/', 1)[-1],
                    'layer': 'advertisement', 'closed': bool(path.get('closed')),
                    'begin': offset, 'end': offset+len(xyz),
                    'kind': 'installed_artwork_polyline', 'support': [0., 1.],
                    'instance_id': identity, 'template_id': template['id'],
                    'template_path_id': path['id'], 'role': path.get('role'),
                })
                chunks.append(xyz)
                offset += len(xyz)
    sampled['xyz'] = np.vstack(chunks)
    sampled['missing_entities'] = missing
    sampled['inventory'] = {
        **inventory, 'material_ids': material_ids, 'material_count': len(material_ids),
        'path_ids': [p['id'] for p in sampled['cuts']], 'path_count': len(sampled['cuts']),
        'registered_print_instance_ids': registered,
        'sampled_print_instance_ids': localized,
        'unlocalized_print_instance_ids': unlocalized,
    }
    sampled['print_sampling_stations'] = print_grids
    sampled['scene_path'] = str(Path(scene_path).resolve())
    sampled['_parameter_state'] = parameter_state
    return sampled


def common_print_stations(first, second):
    """Compare corresponding material locations, even when sizes change.

    The union retains both adaptive drawing grids; it does not interpolate
    projected pixels or turn representation samples into observations.
    """
    if first['inventory'] != second['inventory']:
        raise ValueError('comparison requires the same entity and print-instance inventory')
    grids = {}
    for identity, paths in first['print_sampling_stations'].items():
        other = second['print_sampling_stations'][identity]
        if paths.keys() != other.keys():
            raise ValueError('comparison print topology differs for '+identity)
        grids[identity] = {key: np.union1d(values, other[key]).tolist()
                           for key, values in paths.items()}
    return grids


def sample_review_scene(scene, core, scene_path, *, print_stations=None, parameter_state=None):
    """Add full board display edges only when the model does not contain them."""
    if parameter_state is not None and hasattr(parameter_state, 'apply_scene'):
        scene = parameter_state.apply_scene(scene)
    samples = sample_scene(scene, core, scene_path, print_stations=print_stations, parameter_state=parameter_state)
    existing_heights = set()
    for path in scene['paths']:
        kind, parameters = native_definition(scene, path, core)
        if kind == 7:
            existing_heights.add(float(parameters[0]))
        elif kind == 3 and parameters[0] == -1:
            existing_heights.add(float(parameters[1]))
    derived = []
    for height, name in ((0., 'continuous_board_bottom'), (1., 'continuous_board_top')):
        if height in existing_heights:
            continue
        xyz = np.asarray(core.sample_path(scene['board'], 3, [-1., height], [],
                                         np.linspace(0., 1., 4097)))
        begin = len(samples['xyz'])
        samples['xyz'] = np.vstack((samples['xyz'], xyz))
        samples['cuts'].append({'id': name, 'entity_id': 'continuous_board', 'label': name,
            'layer': 'board', 'closed': True, 'begin': begin, 'end': begin+len(xyz),
            'kind': 'derived_continuous_board_edge', 'support': [0., 1.]})
        derived.append(name)
    samples['inventory']['derived_display_paths'] = derived
    return samples
