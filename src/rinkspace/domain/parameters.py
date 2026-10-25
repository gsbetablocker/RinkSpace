"""Shared scene parameter layout and immutable numeric state.

Static definitions and source observations are not candidate values. All print
instances consume this one ID/offset layout; no advertisement solver lives here.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .scene_assets import read_scene_assets, scene_content_identity

LAYOUT_SCHEMA = 'rinkspace.parameter_layout.v1'
STATE_LAYOUT = 'shared_scene_parameters.v1'


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def compile_scene_parameters(scene, scene_path):
    """Compile present numeric blocks only; never activate observations.

    Board/layout values stay frozen in this task. Print internal geometry and
    existing size/placement coordinates share one vector and active-column map.
    Unlocalized blocks retain None and allocate no invented numeric values.
    """
    from .print_parameters import validate_print_parameters
    blocks, values, identities = [], [], set()

    def add(record, initial, target, *, active=False):
        identity = record['id']
        if identity in identities:
            raise ValueError('Duplicate shared parameter block: '+identity)
        identities.add(identity)
        size = record['size']
        if type(size) is not int or size <= 0:
            raise ValueError('Invalid parameter block size: '+identity)
        indices = record.get('independent_local_indices', list(range(size)))
        if (not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < size for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('Invalid independent parameter coordinates: '+identity)
        available = initial is not None
        enabled = bool(record.get('active', active)) and available
        numeric = None if not available else np.asarray(initial, dtype='<f8')
        if numeric is not None and (numeric.shape != (size,) or not np.isfinite(numeric).all()):
            raise ValueError('Nonfinite/incomplete parameter values: '+identity)
        independent = indices if enabled else []
        block = {key: deepcopy(value) for key, value in record.items()
                 if key not in ('values', 'offset', 'active', 'original_block')}
        block.update(id=identity, size=size, offset=len(values) if available else None,
            available=available, active=enabled, independent_local_indices=independent,
            tangent_size=len(independent), target=deepcopy(target),
            constant_values=([None if i in independent else float(numeric[i]) for i in range(size)]
                             if available else None))
        if available:
            values.extend(numeric.tolist())
        blocks.append(block)

    scene_id = scene.get('scene_id', 'scene')
    if 'board' in scene:
        add(dict(id='scene/'+scene_id+'/board', kind='board_shape', size=len(scene['board'])),
            scene['board'], {'kind':'board'})
    layout = scene.get('shared_layout')
    if layout is not None:
        claimed = set()
        for record in scene.get('parameter_blocks', []):
            start, size = record['offset'], record['size']
            support = set(range(start, start+size))
            if start < 0 or start+size > len(layout) or claimed.intersection(support):
                raise ValueError('Overlapping/out-of-range shared scene parameter block')
            claimed.update(support)
            descriptor = {**record, 'id':'scene/'+scene_id+'/'+record['id'], 'kind':'shared_geometry'}
            add(descriptor, layout[start:start+size], {'kind':'shared_layout', 'offset':start})
        if claimed != set(range(len(layout))):
            raise ValueError('Shared scene layout must register every stored coordinate')
    for reference, assets in read_scene_assets(scene, scene_path):
        validate_print_parameters(assets)
        selected = set(reference.get('instance_ids', [reference.get('instance_id')]))
        allowed_placements = {i['placement_block_id'] for i in assets['instances'] if i['id'] in selected}
        for record in assets.get('parameter_blocks', []):
            descriptor = record
            if record['kind'] == 'print_placement' and record['id'] not in allowed_placements:
                # An asset's complete values contract remains readable in an
                # instance subset. Omitted placements retain existing values
                # (or None), but acquire no active optimization coordinates.
                descriptor = {**record, 'active':False}
            add(descriptor, record.get('values'), {'kind':'print', 'asset_template_ids':
                [t['id'] for t in assets['templates']]}, active=record['kind'].startswith('print_'))
    definition = {'schema':LAYOUT_SCHEMA, 'scene_content_revision':scene_content_identity(scene, scene_path),
        'blocks':blocks, 'size':len(values), 'scope':'shared_scene_geometry',
        'source_observations_changed':False}
    definition['layout_revision'] = _identity(definition)
    return ParameterState(definition, np.asarray(values, dtype='<f8'))


class ParameterState(Mapping):
    """Immutable view of one validated common float64 state vector."""
    def __init__(self, layout, values):
        if not isinstance(layout, dict) or layout.get('schema') != LAYOUT_SCHEMA:
            raise ValueError('Unknown scene parameter layout')
        raw = np.asarray(values)
        if (raw.dtype.str != '<f8' or raw.ndim != 1 or not raw.flags.c_contiguous
                or not np.isfinite(raw).all() or len(raw) != layout.get('size')):
            raise ValueError('State values must be finite contiguous float64 with exact layout size')
        expected = {k:v for k,v in layout.items() if k != 'layout_revision'}
        if layout.get('layout_revision') != _identity(expected):
            raise ValueError('Parameter layout content identity differs')
        registry, claimed = {}, set()
        for block in layout['blocks']:
            identity, size = block.get('id'), block.get('size')
            if not isinstance(identity, str) or not identity or identity in registry:
                raise ValueError('Duplicate/missing state parameter identity')
            if type(size) is not int or size <= 0:
                raise ValueError('Invalid state parameter block size')
            offset = block.get('offset')
            independent = block.get('independent_local_indices')
            if (not isinstance(independent, list) or len(set(independent)) != len(independent)
                    or any(type(i) is not int or not 0 <= i < size for i in independent)
                    or block.get('tangent_size') != len(independent)):
                raise ValueError('Invalid state tangent layout: '+identity)
            if not block.get('available'):
                if offset is not None or block.get('active') or independent:
                    raise ValueError('Unlocalized parameters cannot acquire values or active columns')
            else:
                if type(offset) is not int or offset < 0 or offset+size > len(raw):
                    raise ValueError('State parameter offset is out of bounds')
                support = set(range(offset, offset+size))
                if claimed.intersection(support):
                    raise ValueError('State parameter blocks overlap')
                claimed.update(support)
                constants = block.get('constant_values')
                if not isinstance(constants, list) or len(constants) != size:
                    raise ValueError('State constant-coordinate layout is incomplete')
                for i, value in enumerate(constants):
                    if i in independent:
                        if value is not None:
                            raise ValueError('Active coordinate declared constant')
                    elif value is None or raw[offset+i] != value:
                        raise ValueError('Constrained/frozen parameter coordinate changed: '+identity)
            registry[identity] = deepcopy(block)
        if claimed != set(range(len(raw))):
            raise ValueError('State contains unregistered numeric coordinates')
        self._layout = deepcopy(layout)
        self._values = np.array(raw, dtype='<f8', order='C', copy=True)
        self._values.setflags(write=False)
        def freeze(value):
            if isinstance(value, dict):
                return MappingProxyType({k:freeze(v) for k,v in value.items()})
            if isinstance(value, list):
                return tuple(freeze(v) for v in value)
            return value
        self.blocks = freeze(registry)

    @property
    def layout(self):
        return deepcopy(self._layout)

    @property
    def values(self):
        result = self._values.view()
        result.setflags(write=False)
        return result

    def __iter__(self):
        return iter(self.blocks)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, identity):
        block = self.blocks[identity]
        if not block['available']:
            return None
        return self.values[block['offset']:block['offset']+block['size']]

    def active_columns(self, block_ids=None):
        wanted = set(self.blocks) if block_ids is None else set(block_ids)
        unknown = wanted-set(self.blocks)
        if unknown:
            raise ValueError('Unknown requested active parameter IDs: '+str(sorted(unknown)))
        return np.asarray([b['offset']+i for b in self.blocks.values() if b['id'] in wanted
            and b['active'] for i in b['independent_local_indices']], dtype='<i8')

    def with_blocks(self, replacements):
        """Create a numeric candidate, respecting the compiled active set/gauge."""
        values = self.values.copy()
        for identity, proposed in replacements.items():
            if identity not in self.blocks:
                raise ValueError('Unknown candidate parameter block: '+identity)
            block = self.blocks[identity]
            candidate = np.asarray(proposed, dtype='<f8')
            if not block['available'] or candidate.shape != (block['size'],):
                raise ValueError('Candidate block is unlocalized or has the wrong size: '+identity)
            values[block['offset']:block['offset']+block['size']] = candidate
        return ParameterState(self.layout, values)

    def apply_scene(self, scene):
        """Extract current scene values; geometry remains evaluated by native."""
        result = deepcopy(scene)
        for identity, block in self.blocks.items():
            target = block['target']
            if target['kind'] == 'board':
                result['board'] = self[identity].tolist()
            elif target['kind'] == 'shared_layout':
                start = target['offset']
                result['shared_layout'][start:start+block['size']] = self[identity].tolist()
        return result


def load_parameter_state(state_path, *, expected_scene=None, expected_scene_path=None):
    """Load only the current layout and values; no provenance/history replay."""
    from ..session.dataset import load_array, read_json, relative
    path = Path(state_path).resolve(strict=True)
    state = read_json(path)
    if state.get('schema') != 'rinkspace.state.v1' or state.get('layout_family') != STATE_LAYOUT:
        raise ValueError('Expected a shared scene parameter state')
    layout = read_json(relative(path.parent, state['parameter_layout_ref']))
    if state.get('layout_revision') != layout.get('layout_revision') or state.get('parameter_blocks') != layout.get('blocks'):
        raise ValueError('State ID/offset layout differs from its parameter layout')
    desc = state.get('values_ref')
    if not isinstance(desc, dict):
        raise ValueError('Current state requires a typed values array reference')
    values = load_array(path.parent, desc, verify=True)
    result = ParameterState(layout, values)
    if expected_scene is not None:
        actual = compile_scene_parameters(expected_scene, expected_scene_path)
        if actual.layout != result.layout:
            raise ValueError('State parameter layout differs from the current scene definition')
    return result


def write_parameter_state(root, parameter_state, *, scene_ref='scene.json', prepared_revision='unbound',
                          shots=(), required_shot_ids=(), selected_shot_ids=(), name='state.json'):
    """Serialize values/layout into the same state.v1 used by review/future solve.

    This is a parameter snapshot, not a solver checkpoint or an optimization run.
    Per-observation stations/noise and approval remain owned by prepared inputs.
    """
    from ..session.dataset import write_array, write_json
    from ..path_policy import guard_project_write
    root = Path(root)
    if Path(name).name != name or not name.endswith('.json'):
        raise ValueError('State snapshot name must be one JSON filename')
    values_name = 'state_'+Path(name).stem
    for relative_path in ('parameter_layout.json', 'arrays/'+values_name+'.npy', name):
        guard_project_write(root/relative_path)
    if (root/name).exists() or (root/'arrays'/(values_name+'.npy')).exists():
        raise FileExistsError('State snapshots never overwrite existing numeric candidates')
    layout_path = root/'parameter_layout.json'
    if layout_path.exists():
        existing = json.loads(layout_path.read_text(encoding='utf-8-sig'))
        if existing != parameter_state.layout:
            raise ValueError('State snapshot folder contains a different parameter layout')
    else:
        write_json(layout_path, parameter_state.layout)
    values_ref = write_array(root, values_name, parameter_state.values, '<f8')
    selected, required = list(selected_shot_ids), list(required_shot_ids)
    state = {'schema':'rinkspace.state.v1', 'state_id':'parameters_'+_identity({
        'layout':parameter_state.layout['layout_revision'], 'values':values_ref['sha256']})[:20],
        'layout_family':STATE_LAYOUT, 'layout_revision':parameter_state.layout['layout_revision'],
        'scene_revision':parameter_state.layout['scene_content_revision'],
        'prepared_revision':prepared_revision, 'scene_ref':scene_ref,
        'parameter_layout_ref':'parameter_layout.json', 'parameter_blocks':parameter_state.layout['blocks'],
        'values_ref':values_ref, 'local_values_ref':None, 'local_state_status':'owned_by_prepared_inputs_not_invented',
        'shots':deepcopy(list(shots)), 'required_shot_ids':required, 'selected_shot_ids':selected,
        'scope_status':'complete' if required and set(required)==set(selected) else 'subset',
        'origin':{'kind':'current_scene_parameters_without_optimization'},
        'fit_executed':False, 'cuda_executed':False, 'user_approval':'not_reviewed'}
    write_json(root/name, state)
    return state


class SceneParameterModel:
    """Present scene topology plus one common parameter layout for all prints."""
    def __init__(self, scene, scene_path):
        self.scene, self.scene_path = deepcopy(scene), Path(scene_path).resolve()
        self.initial_state = compile_scene_parameters(scene, scene_path)
        self.assets = read_scene_assets(scene, scene_path)
        self.paths = {}
        for reference, assets in self.assets:
            templates = {t['id']:t for t in assets['templates']}
            for instance in assets['instances']:
                if instance['id'] not in reference['instance_ids']:
                    continue
                for path in templates[instance['template_id']]['paths']:
                    identity = instance['id']+'::'+path['id']
                    if identity in self.paths:
                        raise ValueError('Duplicate compiled print feature path')
                    self.paths[identity] = (assets, instance['id'], path['id'])

    def dependencies(self, feature_path_id):
        from .print_parameters import print_parameter_dependencies
        if feature_path_id not in self.paths:
            raise ValueError('Unregistered print feature path: '+feature_path_id)
        assets, instance_id, path_id = self.paths[feature_path_id]
        return print_parameter_dependencies(assets, instance_id, path_id)

    def evaluate_print(self, feature_path_id, stations, core, *, state=None, jacobian=True):
        from .print_assets import evaluate_print_path
        current = self.initial_state if state is None else state
        if current.layout != self.initial_state.layout:
            raise ValueError('Candidate and compiled scene parameter layouts differ')
        assets, instance_id, path_id = self.paths[feature_path_id]
        result = evaluate_print_path(assets, instance_id, path_id,
            current.apply_scene(self.scene)['board'], core, stations,
            parameter_state=current, jacobian=jacobian)
        result['parameter_columns'] = {identity: self.initial_state.blocks[identity]['offset']+
            np.asarray(self.initial_state.blocks[identity]['independent_local_indices'], dtype='<i8')
            for identity in self.dependencies(feature_path_id) if self.initial_state.blocks[identity]['available']}
        result['layout_revision'] = current.layout['layout_revision']
        return result
